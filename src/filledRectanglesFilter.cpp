/******************************************************************************
 The application Pic2Sym approximates images by a
 grid of colored symbols with colored backgrounds.

 Copyrights from the libraries used by the program:
 - (c) 2003-2021 Boost (www.boost.org)
     License: doc/licenses/Boost.lic
     http://www.boost.org/LICENSE_1_0.txt
 - (c) 2015-2021 OpenCV (www.opencv.org)
     License: doc/licenses/OpenCV.lic
     http://opencv.org/license/
 - (c) 1996-2021 The FreeType Project (www.freetype.org)
     License: doc/licenses/FTL.txt
     http://git.savannah.gnu.org/cgit/freetype/freetype2.git/plain/docs/FTL.TXT
 - (c) 1997-2021 OpenMP Architecture Review Board (www.openmp.org)
   (c) Microsoft Corporation (implementation for OpenMP C/C++ v2.0 March 2002)
     See: https://msdn.microsoft.com/en-us/library/8y6825x5.aspx
 - (c) 1995-2021 zlib software (Jean-loup Gailly and Mark Adler - www.zlib.net)
     License: doc/licenses/zlib.lic
     http://www.zlib.net/zlib_license.html
 - (c) 2015-2021 Microsoft Guidelines Support Library - github.com/microsoft/GSL
     License: doc/licenses/MicrosoftGSL.lic
     https://raw.githubusercontent.com/microsoft/GSL/main/LICENSE


 (c) 2016-2021 Florin Tulba <florintulba@yahoo.com>

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
// This keeps precompiled.h first; Otherwise header sorting might move it

#include "filledRectanglesFilter.h"

#include "pixMapSymBase.h"
#include "symFilterCache.h"

#pragma warning(push, 0)

#include <opencv2/imgproc/imgproc.hpp>

#pragma warning(pop)

using namespace std;
using namespace cv;
using namespace gsl;

namespace pic2sym {

SYM_FILTER_DEFINE_IS_ENABLED(FilledRectanglesFilter)

namespace syms::inline filter {

/**
Analyzes a horizontal / vertical projection (reduction sum) of the
glyph, checking for clues of rectangular blocks: a projection with one /
several adjacent indices holding the maximum value.

@param sums the projection (reduction sum) of the glyph in horizontal /
vertical direction
@param sideLen height / width of glyph's bounding box
@param countOfMaxSums [out] determined side length of a possible
rectangle: [1..sideLen]

@return true if the projection denotes a valid uncut rectangular block
*/
bool FilledRectanglesFilter::checkProjectionForFilledRectangles(
    const Mat& sums,
    unsigned sideLen,
    int& countOfMaxSums) noexcept {
  static const Mat structuringElem{1, 3, CV_8U, Scalar{1.}};

  double maxVSums;
  minMaxIdx(sums, nullptr, &maxVSums);

  // these should be the white rows/columns
  const Mat sumsOnMax{sums == maxVSums};
  countOfMaxSums = countNonZero(sumsOnMax);  // 1..sideLen
  if (countOfMaxSums == (int)sideLen)
    // the white rows/columns are consecutive, for sure
    return true;

  if (countOfMaxSums == 1)
    return true;  // a single white row/column is a (degenerate)
                  // rectangle

  if (countOfMaxSums == 2) {
    // pad sumsOnMax and then dilate it
    Mat paddedSumsOnMax{1, sumsOnMax.cols + 2, CV_8U, Scalar{}};
    sumsOnMax.copyTo((const Mat&)Mat{paddedSumsOnMax, Range::all(),
                                     Range{1, sumsOnMax.cols + 1}});
    dilate(paddedSumsOnMax, paddedSumsOnMax, structuringElem);
    if (countNonZero(paddedSumsOnMax) - countOfMaxSums > 2)
      return false;  // there was at least one gap, so dilation filled
                     // more than 2 pixels
  } else {           // countOfMaxSums is [3..sideLen)
    erode(sumsOnMax, sumsOnMax, structuringElem);
    if (countOfMaxSums - countNonZero(sumsOnMax) > 2)
      return false;  // there was at least one gap, so erosion teared
                     // down more than 2 pixels
  }
  return true;
}

FilledRectanglesFilter::FilledRectanglesFilter(
    unique_ptr<ISymFilter> nextFilter_ /* = nullptr*/) noexcept
    : TSymFilter{0U, "filled rectangles", move(nextFilter_)} {}

bool FilledRectanglesFilter::isDisposable(const IPixMapSym& pms,
                                          const SymFilterCache&) noexcept {
  if (!isEnabled())
    return false;

  const Mat narrowGlyph{pms.asNarrowMat()};
  double brightestVal;
  minMaxIdx(narrowGlyph, nullptr, &brightestVal);
  const int brightestPixels{
      countNonZero(narrowGlyph == narrow_cast<unsigned char>(brightestVal))};
  if (brightestPixels < 3)
    return false;  // don't report really small areas

  // Sides of a possible filled rectangle
  int rowsWithMaxSum{};
  int colsWithMaxSum{};

  // Analyze the horizontal and vertical projections of pms, looking for
  // signs of rectangle symbols
  if (!checkProjectionForFilledRectangles(pms.getRowSums(), pms.getRows(),
                                          rowsWithMaxSum))
    return false;
  if (!checkProjectionForFilledRectangles(pms.getColSums(), pms.getCols(),
                                          colsWithMaxSum))
    return false;
  if (brightestPixels != colsWithMaxSum * rowsWithMaxSum)
    return false;  // rectangle's area should be the product of its
                   // sides

  return true;
}

}  // namespace syms::inline filter
}  // namespace pic2sym
