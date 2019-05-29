/******************************************************************************
 The application Pic2Sym approximates images by a
 grid of colored symbols with colored backgrounds.

 Copyrights from the libraries used by the program:
 - (c) 2003 Boost (www.boost.org)
     License: doc/licenses/Boost.lic
     http://www.boost.org/LICENSE_1_0.txt
 - (c) 2015-2016 OpenCV (www.opencv.org)
     License: doc/licenses/OpenCV.lic
     http://opencv.org/license/
 - (c) 1996-2002, 2006 The FreeType Project (www.freetype.org)
     License: doc/licenses/FTL.txt
     http://git.savannah.gnu.org/cgit/freetype/freetype2.git/plain/docs/FTL.TXT
 - (c) 1997-2002 OpenMP Architecture Review Board (www.openmp.org)
   (c) Microsoft Corporation (implementation for OpenMP C/C++ v2.0 March 2002)
     See: https://msdn.microsoft.com/en-us/library/8y6825x5.aspx
 - (c) 1995-2017 zlib software (Jean-loup Gailly and Mark Adler - www.zlib.net)
     License: doc/licenses/zlib.lic
     http://www.zlib.net/zlib_license.html


 (c) 2016-2019 Florin Tulba <florintulba@yahoo.com>

 This program is free software: you can use its results,
 redistribute it and/or modify it under the terms of the GNU
 Affero General Public License version 3 as published by the
 Free Software Foundation.

 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 GNU Affero General Public License for more details.

 You should have received a copy of the GNU Affero General Public License
 along with this program ('agpl-3.0.txt').
 If not, see: http://www.gnu.org/licenses/agpl-3.0.txt .
 *****************************************************************************/

#include "precompiled.h"

#include "gridBarsFilter.h"
#include "pixMapSymBase.h"
#include "symFilterCache.h"
#include "warnings.h"

#pragma warning(push, 0)

#include <set>

#include <opencv2/imgproc/imgproc.hpp>

#pragma warning(pop)

using namespace std;
using namespace cv;

extern template class set<double>;

static const Point defAnchor(-1, -1);

/**
Determines if the vertical / horizontal profile of the glyph contains only
several patterns, since glyph lines have a quite constant profile, apart from
any encountered corners.

The number of this pattern changes needs to be extremely small on all the
margins of an imaginary cross overlapped on the glyph. Central part could accept
more patterns.
*/
static bool acceptableProfile(
    const Mat& narrowGlyph,     ///< bounding box (BB) region of the glyph
    const SymFilterCache& sfc,  ///< precomputed values
    unsigned crossClearance,    ///< size of the margins of the imaginary cross
    unsigned firstRowColBB,     ///< row/col where BB starts
    unsigned lastRowColBB,      ///< row/col where BB ends
    const Mat& projSums,        ///< row/col projection sums
    Mat (Mat::*fMatRowCol)(int) const  ///< one of &Mat::row/col
    ) noexcept {
  unsigned diffsBeforeCenter = 0U, diffsCentralRegion = 0U,
           diffsAfterCenter = 0U, firstNonEmptyRowColBB = 0U,
           lastNonEmptyRowColBB =
               0U,      // first/last relevant line / col relative to BB
      rowColProj = 0U;  // row / column within horizontal / vertical projections
                        // of
                        // the glyph
  for (lastNonEmptyRowColBB = lastRowColBB,
      rowColProj = lastNonEmptyRowColBB + firstRowColBB;
       0. == projSums.at<double>((int)rowColProj);
       --rowColProj, --lastNonEmptyRowColBB) {
  }
  for (firstNonEmptyRowColBB = 0U, rowColProj = firstRowColBB;
       0. == projSums.at<double>((int)rowColProj);
       ++rowColProj, ++firstNonEmptyRowColBB) {
  }

  static constexpr double FactorTolUnder = .9,
                          FactorTolAbove = 1. / FactorTolUnder;

  Mat prevData = (narrowGlyph.*fMatRowCol)((int)firstNonEmptyRowColBB),
      prevDataInfLim = prevData * FactorTolUnder,
      prevDataSupLim = prevData * FactorTolAbove;
  int cnzPrevData = countNonZero(prevData);

  for (unsigned rc = firstNonEmptyRowColBB + 1U; rc <= lastNonEmptyRowColBB;
       ++rc) {
    ++rowColProj;
    const Mat thisRowCol = (narrowGlyph.*fMatRowCol)((int)rc);
    const int cnzThisRowCol = countNonZero(thisRowCol);
    if (cnzPrevData == cnzThisRowCol) {
      MatConstIterator_<unsigned char> itThisRowCol =
          thisRowCol.begin<unsigned char>();
      const MatConstIterator_<unsigned char> itThisRowColEnd =
          thisRowCol.end<unsigned char>();
      MatConstIterator_<unsigned char>
          itPrevDataInfLim = prevDataInfLim.begin<unsigned char>(),
          itPrevDataSupLim = prevDataSupLim.begin<unsigned char>();
      bool quiteTheSamePattern = true;
      while (itThisRowCol != itThisRowColEnd) {
        if (const unsigned char cellVal = *itThisRowCol;
            cellVal < *itPrevDataInfLim || cellVal > *itPrevDataSupLim) {
          quiteTheSamePattern = false;
          break;
        }
        ++itThisRowCol;
        ++itPrevDataInfLim;
        ++itPrevDataSupLim;
      }
      if (quiteTheSamePattern)
        continue;
    }

    // There should be at most 2 patterns above/below or to the left/right of
    // the center area and a lot more patterns in the central region
    if (rowColProj < crossClearance) {
      if (++diffsBeforeCenter > 1U)
        return false;

      // Even when different, the 2 rows/columns must have non-zero elements in
      // same columns/rows
      Mat intersMin;
      (cv::min)(thisRowCol, prevData, intersMin);
      if (countNonZero(intersMin) < cnzPrevData)
        return false;

    } else if (rowColProj >= sfc.getSzU() - crossClearance) {
      if (++diffsAfterCenter > 1U)
        return false;

      // Even when different, the 2 rows/columns must have non-zero elements in
      // same columns/rows
      Mat intersMin;
      (cv::min)(thisRowCol, prevData, intersMin);
      if (countNonZero(intersMin) < cnzPrevData)
        return false;

    } else {
      if (++diffsCentralRegion > 9U)
        return false;
    }

    // Update cnz, prevData, inferior & superior limits for future compares
    prevData = thisRowCol;
    cnzPrevData = cnzThisRowCol;
    prevDataInfLim = thisRowCol * FactorTolUnder;
    prevDataSupLim = thisRowCol * FactorTolAbove;
  }

  return true;
}

GridBarsFilter::GridBarsFilter(
    unique_ptr<ISymFilter> nextFilter_ /* = nullptr*/) noexcept
    : TSymFilter(1U, "grid-like symbols", move(nextFilter_)) {}

#pragma warning(disable : WARN_THROWS_ALTHOUGH_NOEXCEPT)
bool GridBarsFilter::checkProjectionForGridSymbols(const Mat& sums) noexcept(
    !UT) {
  const MatConstIterator_<double> itFirst = sums.begin<double>(),
                                  itEnd = sums.end<double>();
  set<double> uniqueVals(itFirst, itEnd);
  if (uniqueVals.size() == 1)
    return true;  // Pattern is: A+   (case (c))

  uniqueVals.erase(0.);  // Erase the 0, if present

  auto itUniqueVals = uniqueVals.crbegin();  // reverse iterator !!
  double A = *itUniqueVals++, t1 = -1., t2 = -1.;
  const auto uniqueValsCount = uniqueVals.size();
  switch (uniqueValsCount) {
    case 1:  // Pattern is: 0* A+ 0*  (case (c))
      break;
    case 2:  // Pattern is: 0* t1* A+ t1* 0*   (case (c))
      t1 = *itUniqueVals;
      break;
    case 3:  // Pattern is 0* t2+ t1+ A+ 0* (case (a))   or    0* A+ t1+ t2+ 0*
             // (case (b))
      t2 = *itUniqueVals++;
      t1 = *itUniqueVals;
      break;
    case 0:
      [[fallthrough]];
    default:
      return false;  // 1 .. 3 possible values: t1 < t2 < A
  }

  vector<double> vals(itFirst, itEnd);
  enum class State { Read0s, ReadT2, ReadT1, ReadA };
  bool metA = false;
  State state{State::Read0s};
  for (auto val : vals) {
    switch (state) {
      case State::Read0s:
        if (val == 0.)
          continue;  // stay in state Read0s
        if (metA)
          return false;  // last 0s cannot be followed by t1/t2/A

        if (val == A) {
          state = State::ReadA;
          metA = true;
        } else if (val == t2) {
          state = State::ReadT2;
        } else {
          state = State::ReadT1;
        }
        break;

      case State::ReadT2:
        if (val == t2)
          continue;  // stay in state ReadT2
        if (val == A)
          return false;  // from T2 it's possible to accept only 0 and T1

        if (val == 0.) {
          if (!metA)
            return false;         // case (a) T1 is expected here
          state = State::Read0s;  // case (b)
        } else {                  // val == t1 here
          if (metA)
            return false;         // case (b) 0 is expected here
          state = State::ReadT1;  // case (a)
        }
        break;

      case State::ReadT1:
        if (val == t1)
          continue;  // stay in state ReadT1
        if (val == A) {
          if (metA)
            return false;  // case (c) 0 is expected here
          state = State::ReadA;
          metA = true;
        } else if (val == t2) {
          if (!metA)
            return false;  // case (a) A is expected here
          state = State::ReadT2;
        } else {  // val == 0 here
          if (!metA)
            return false;  // case (a) A is expected here
          if (t2 != -1)
            return false;  // case (b) t2 is expected here
          state = State::Read0s;
        }
        break;

      case State::ReadA:
        if (val == A)
          continue;  // stay in state ReadA
        if (val == t2)
          return false;  // 0 and t1 are the only allowed values here

        if (val == 0.) {
          state = State::Read0s;
        } else {
          state = State::ReadT1;
        }
        break;

      default:
        THROW_WITH_VAR_MSG("Handling of state " + to_string((int)state) +
                               " not implemented yet!",
                           logic_error);
    }
  }

  return metA;
}
#pragma warning(default : WARN_THROWS_ALTHOUGH_NOEXCEPT)

bool GridBarsFilter::isDisposable(const IPixMapSym& pms,
                                  const SymFilterCache& sfc) noexcept {
  if (!isEnabled())
    return false;

  // At least one side of the bounding box needs to be larger than half sz
  if (max(pms.getRows(), pms.getCols()) < (sfc.getSzU() >> 1))
    return false;

  Mat narrowGlyph = pms.asNarrowMat();
  Mat glyphBin = (narrowGlyph > 0U);

  const unsigned crossClearance = (unsigned)max(1, int(sfc.getSzU() / 3U - 1U)),
                 crossWidth =
                     sfc.getSzU() -
                     (crossClearance << 1);  // more than 1/3 from font size
  const int minPixelsCenter =
                int(crossWidth - 1U) |
                1,  // crossWidth when odd, or crossWidth-1 when even
      minPixelsBranch = int((crossClearance << 1) / 3),   // 2/3*crossClearance
      minPixels = 2 * minPixelsBranch + minPixelsCenter;  // center + 2 branches

  // Don't consider fonts with less pixels than necessary to obtain a grid bar
  if (countNonZero(glyphBin) < minPixels)
    return false;

  // Exclude glyphs that touch pixels outside the main cross
  const Mat glyph = pms.toMat(sfc.getSzU());
  const Range topOrLeft(0, (int)crossClearance),
      rightOrBottom(int(sfc.getSzU() - crossClearance), (int)sfc.getSzU()),
      center((int)crossClearance, int(crossClearance + crossWidth));
  if (countNonZero(Mat(glyph, topOrLeft, topOrLeft)) > 0)
    return false;
  if (countNonZero(Mat(glyph, topOrLeft, rightOrBottom)) > 0)
    return false;
  if (countNonZero(Mat(glyph, rightOrBottom, topOrLeft)) > 0)
    return false;
  if (countNonZero(Mat(glyph, rightOrBottom, rightOrBottom)) > 0)
    return false;

  // Grid bars also need some pixels in the center (>= minPixelsCenter)
  if (countNonZero(Mat(glyph, center, center)) < minPixelsCenter)
    return false;

  // On each end of the imaginary cross, there should be either 0 pixels or >=
  // minPixelsBranch and there have to be at least 2 branches.
  int cnz = 0, branchesCount = 0;
  if ((cnz = countNonZero(Mat(glyph, topOrLeft, center))) >= minPixelsBranch)
    ++branchesCount;
  else if (cnz > 0)
    return false;
  if ((cnz = countNonZero(Mat(glyph, center, topOrLeft))) >= minPixelsBranch)
    ++branchesCount;
  else if (cnz > 0)
    return false;
  if ((cnz = countNonZero(Mat(glyph, rightOrBottom, center))) >=
      minPixelsBranch)
    ++branchesCount;
  else if (cnz > 0)
    return false;
  if ((cnz = countNonZero(Mat(glyph, center, rightOrBottom))) >=
      minPixelsBranch)
    ++branchesCount;
  else if (cnz > 0)
    return false;
  if (branchesCount < 2)
    return false;

  // Making sure glyphBin is entitled to represent narrowGlyph
  if (!acceptableProfile(narrowGlyph, sfc, crossClearance,
                         sfc.getSzU() - 1U - pms.getTop(), pms.getRows() - 1U,
                         pms.getRowSums(), &Mat::row))
    return false;
  if (!acceptableProfile(narrowGlyph, sfc, crossClearance, pms.getLeft(),
                         pms.getCols() - 1U, pms.getColSums(), &Mat::col))
    return false;

  // Closing the space between any parallel lines of the grid symbol
  static const Scalar BlackFrame(0U);

  const int maskSz = max(3, (((int)sfc.getSzU() >> 2) | 1)),
            frameSz = maskSz >> 1;
  const Mat mask(maskSz, maskSz, CV_8UC1, Scalar(1U));
  Mat glyphBinAux(pms.getRows() + 2 * frameSz, pms.getCols() + 2 * frameSz,
                  CV_8UC1, Scalar(0U)),
      destRegion(glyphBinAux, Range(frameSz, pms.getRows() + frameSz),
                 Range(frameSz, pms.getCols() + frameSz)),
      closedGlyphBin;
  glyphBin.copyTo(destRegion);
  morphologyEx(glyphBinAux, closedGlyphBin, MORPH_CLOSE, mask, defAnchor, 1,
               BORDER_CONSTANT, BlackFrame);

  // Analyze the horizontal and vertical sum projections of closedGlyphBin
  Mat projSum;
  closedGlyphBin.convertTo(closedGlyphBin, CV_64FC1, INV_255);
  cv::reduce(closedGlyphBin, projSum, 0, REDUCE_SUM);
  if (!checkProjectionForGridSymbols(projSum))
    return false;

  cv::reduce(closedGlyphBin, projSum, 1, REDUCE_SUM);
  if (!checkProjectionForGridSymbols(projSum))
    return false;

  return true;
}
