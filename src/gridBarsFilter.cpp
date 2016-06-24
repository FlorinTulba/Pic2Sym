/****************************************************************************************
 The application Pic2Sym approximates images by a
 grid of colored symbols with colored backgrounds.

 This file belongs to the Pic2Sym project.

 Copyrights from the libraries used by the program:
 - (c) 2015 Boost (www.boost.org)
   License: <http://www.boost.org/LICENSE_1_0.txt>
            or doc/licenses/Boost.lic
 - (c) 2015 The FreeType Project (www.freetype.org)
   License: <http://git.savannah.gnu.org/cgit/freetype/freetype2.git/plain/docs/FTL.TXT>
	        or doc/licenses/FTL.txt
 - (c) 2015 OpenCV (www.opencv.org)
   License: <http://opencv.org/license.html>
            or doc/licenses/OpenCV.lic
 - (c) 1997-2002 OpenMP Architecture Review Board (www.openmp.org)
   (c) Microsoft Corporation (Visual C++ implementation for OpenMP C/C++ Version 2.0 March 2002)
   See: <https://msdn.microsoft.com/en-us/library/8y6825x5(v=vs.140).aspx>
 
 (c) 2016 Florin Tulba <florintulba@yahoo.com>

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
 If not, see <http://www.gnu.org/licenses/agpl-3.0.txt>.
 ****************************************************************************************/

#include "gridBarsFilter.h"
#include "pixMapSym.h"
#include "symFilterCache.h"
#include "misc.h"

#include <set>

#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

static const Point defAnchor(-1, -1);
static const double INV_255 = 1./255;

/**
Determines if the vertical / horizontal profile of the glyph contains only several patterns, since
glyph lines have a quite constant profile, apart from any encountered corners.

The number of this pattern changes needs to be extremely small on all the margins of an imaginary
cross overlapped on the glyph. Central part could accept more patterns.
*/
static bool acceptableProfile(const Mat &narrowGlyph,	///< bounding box (BB) region of the glyph
							  const PixMapSym &pms,		///< whole PixMapSym object
							  const SymFilterCache &sfc,///< precomputed values
							  unsigned crossClearance,	///< size of the margins of the imaginary cross
							  unsigned firstRowColBB,	///< row/col where BB starts
							  unsigned lastRowColBB,	///< row/col where BB ends
							  const Mat &projSums,		///< row/col projection sums
							  Mat(Mat::*fMatRowCol)(int) const ///< one of &Mat::row/col
							  ) {
	Mat prevData;
	double prevSums = 0.;
	unsigned diffsBeforeCenter = 0U, diffsCentralRegion = 0U, diffsAfterCenter = 0U,
		firstNonEmptyRowColBB, lastNonEmptyRowColBB, // first/last relevant line / col relative to BB
		rowColProj; // row / column within horizontal / vertical projections of the glyph
	for(lastNonEmptyRowColBB = lastRowColBB, rowColProj = lastNonEmptyRowColBB + firstRowColBB;
		0. == projSums.at<double>(rowColProj); --rowColProj, --lastNonEmptyRowColBB);
	for(firstNonEmptyRowColBB = 0U, rowColProj = firstRowColBB;
		0. == projSums.at<double>(rowColProj); ++rowColProj, ++firstNonEmptyRowColBB);
	prevSums = projSums.at<double>(rowColProj);
	prevData = (narrowGlyph.*fMatRowCol)(firstNonEmptyRowColBB);

	for(unsigned rc = firstNonEmptyRowColBB + 1U; rc <= lastNonEmptyRowColBB; ++rc) {
		const Mat thisRowCol = (narrowGlyph.*fMatRowCol)(rc);
		const double thisSums = projSums.at<double>(++rowColProj);
		if(thisSums == prevSums) {
			/* equal(BOUNDS_FOR_ITEM_TYPE(thisRowCol, const unsigned char),
										stdext::checked_array_iterator<MatIterator_<const unsigned char>>
										(prevData.begin<const unsigned char>(), prevData.total()))
			would generate a read from address 0 in Debug mode */
			auto itThisRowCol = thisRowCol.begin<unsigned char>(),
				itThisRowColEnd = thisRowCol.end<unsigned char>();
			auto itPrevData = prevData.begin<unsigned char>();
			bool notEqual = false;
			while(itThisRowCol != itThisRowColEnd) {
				if(*itThisRowCol++ != *itPrevData++) {
					notEqual = true;
					break;
				}
			}
			if(!notEqual) continue;
		}

		// There should be at most 2 patterns above/below or to the left/right of the center area and
		// at most 4 patterns in the central region
		if(rowColProj < crossClearance) {
			if(++diffsBeforeCenter > 1U) return false;

			// All elements from thisRowCol should be >= than the corresponding ones from prevData
			if(thisSums <= prevSums) return false;
			for(unsigned cr = 0, lim = (unsigned)thisRowCol.total(); cr < lim; ++cr)
				if(thisRowCol.at<unsigned char>(cr) < prevData.at<unsigned char>(cr))
					return false;
		} else if(rowColProj >= sfc.szU-crossClearance) {
			if(++diffsAfterCenter > 1U) return false;

			// All elements from thisRowCol should be <= than the corresponding ones from prevData
			if(thisSums >= prevSums) return false;
			for(unsigned cr = 0, lim = (unsigned)thisRowCol.total(); cr < lim; ++cr)
				if(thisRowCol.at<unsigned char>(cr) > prevData.at<unsigned char>(cr))
					return false;
		} else {
			if(++diffsCentralRegion > 4U) return false;
		}

		prevSums = thisSums;
		prevData = thisRowCol;
	}

	return true;
}

GridBarsFilter::GridBarsFilter(unique_ptr<ISymFilter> nextFilter_/* = nullptr*/) :
		TSymFilter(2U, "grid-like symbols", std::move(nextFilter_)) {}

/**
Checks if sums might be the projection of a grid bar symbol.
Allowed patterns:
a) 0* t2+ t1+ A+ 0*
b) 0* A+ t1+ t2+ 0*	(reverse of (a))
c) 0* t1* A+ t1* 0*

where:
- t1 &lt; t2 - thicknesses of projected lines (which were parallel to the projection plan)
- A (&gt;t2&gt;t1) - peak generated by the projection of a line perpendicular to the projection plan.
*/
bool GridBarsFilter::checkProjectionForGridSymbols(const Mat &sums) {
	const auto itFirst = sums.begin<double>(), itEnd = sums.end<double>();
	set<double> uniqueVals(itFirst, itEnd);
	if(uniqueVals.size() == 1)
		return true; // Pattern is: A+   (case (c))

	uniqueVals.erase(0.); // Erase the 0, if present

	auto itUniqueVals = uniqueVals.crbegin(); // reverse iterator !!
	double A = *itUniqueVals++, t1 = -1., t2 = -1.;
	const auto uniqueValsCount = uniqueVals.size();
	switch(uniqueValsCount) {
		case 1: // Pattern is: 0* A+ 0*  (case (c))
			break;
		case 2: // Pattern is: 0* t1* A+ t1* 0*   (case (c))
			t1 = *itUniqueVals;
			break;
		case 3: // Pattern is 0* t2+ t1+ A+ 0* (case (a))   or    0* A+ t1+ t2+ 0* (case (b))
			t2 = *itUniqueVals++;
			t1 = *itUniqueVals;
			break;
		case 0:
		default:
			return false; // 1 .. 3 possible values: t1 < t2 < A
	}

	vector<double> vals(itFirst, itEnd);
	typedef enum { Read0s, ReadT2, ReadT1, ReadA } State;
	bool metA = false;
	State state = Read0s;
	for(auto val : vals) {
		switch(state) {
			case Read0s:
				if(val == 0.) continue; // stay in state Read0s
				if(metA) return false; // last 0s cannot be followed by t1/t2/A

				if(val == A) { state = ReadA; metA = true; } else if(val == t2) { state = ReadT2; } else { state = ReadT1; }
				break;

			case ReadT2:
				if(val == t2) continue; // stay in state ReadT2
				if(val == A) return false; // from T2 it's possible to accept only 0 and T1

				if(val == 0.) {
					if(!metA) return false; // case (a) T1 is expected here
					state = Read0s; // case (b)
				} else { // val == t1 here
					if(metA) return false; // case (b) 0 is expected here
					state = ReadT1; // case (a)
				}
				break;

			case ReadT1:
				if(val == t1) continue; // stay in state ReadT1
				if(val == A) {
					if(metA) return false; // case (c) 0 is expected here
					state = ReadA; metA = true;
				} else if(val == t2) {
					if(!metA) return false; // case (a) A is expected here
					state = ReadT2;
				} else { // val == 0 here
					if(!metA) return false; // case (a) A is expected here
					if(t2 != -1) return false;  // case (b) t2 is expected here
					state = Read0s;
				}
				break;

			case ReadA:
				if(val == A) continue; // stay in state ReadA
				if(val == t2) return false; // 0 and t1 are the only allowed values here

				if(val == 0.) { state = Read0s; } else { state = ReadT1; }
				break;

			default:; // Unreachable
		}
	}

	return metA;
}

bool GridBarsFilter::isDisposable(const PixMapSym &pms, const SymFilterCache &sfc) {
	// At least one side of the bounding box needs to be larger than half sz
	if(max(pms.rows, pms.cols) < (sfc.szU>>1)) return false;

	Mat narrowGlyph = pms.asNarrowMat();
	Mat glyphBin = (narrowGlyph > 0U), closedGlyphBin;
	const int brightestPixels = countNonZero(glyphBin);
	if(brightestPixels < 7) return false; // don't report really small areas

	// Exclude also glyphs that touch pixels outside the main cross
	const Mat glyph = pms.toMat(sfc.szU);
	const unsigned crossClearance = sfc.szU/3,
		crossWidth = sfc.szU - (crossClearance << 1); // at least 1/3 from font size
	const Range topOrLeft(0, crossClearance), rightOrBottom(sfc.szU-crossClearance, sfc.szU);
	if(countNonZero(Mat(glyph, topOrLeft, topOrLeft)) > 0) return false;
	if(countNonZero(Mat(glyph, topOrLeft, rightOrBottom)) > 0) return false;
	if(countNonZero(Mat(glyph, rightOrBottom, topOrLeft)) > 0) return false;
	if(countNonZero(Mat(glyph, rightOrBottom, rightOrBottom)) > 0) return false;

	// Making sure glyphBin entirely represents original narrowGlyph
	narrowGlyph.setTo(255U, narrowGlyph >= 250U); // allow profiles to be less sensitive
	if(!acceptableProfile(narrowGlyph, pms, sfc, crossClearance, sfc.szU - 1U - pms.top, pms.rows - 1U,
		pms.rowSums, &Mat::row)) return false;
	if(!acceptableProfile(narrowGlyph, pms, sfc, crossClearance, pms.left, pms.cols - 1U,
		pms.colSums, &Mat::col)) return false;

	// Closing the space between any parallel lines of the grid symbol
	static const Scalar BlackFrame(0U);
	const int maskSz = max(3, (((int)sfc.szU>>2) | 1)), frameSz = maskSz>>1;
	const Mat mask(maskSz, maskSz, CV_8UC1, Scalar(1U));
	Mat glyphBinAux(pms.rows + 2*frameSz, pms.cols + 2*frameSz, CV_8UC1, Scalar(0U)),
		destRegion(glyphBinAux, Range(frameSz, pms.rows+frameSz), Range(frameSz, pms.cols+frameSz));
	glyphBin.copyTo(destRegion);
	morphologyEx(glyphBinAux, closedGlyphBin, MORPH_CLOSE, mask,
				 defAnchor, 1, BORDER_CONSTANT, BlackFrame);

	// Analyze the horizontal and vertical sum projections of closedGlyphBin
	Mat projSum;
	closedGlyphBin.convertTo(closedGlyphBin, CV_64FC1, INV_255);
	reduce(closedGlyphBin, projSum, 0, REDUCE_SUM);
	if(!checkProjectionForGridSymbols(projSum))
		return false;

	reduce(closedGlyphBin, projSum, 1, REDUCE_SUM);
	if(!checkProjectionForGridSymbols(projSum))
		return false;

	return true;
}