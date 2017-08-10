/************************************************************************************************
 The application Pic2Sym approximates images by a
 grid of colored symbols with colored backgrounds.

 Copyrights from the libraries used by the program:
 - (c) 2016 Boost (www.boost.org)
		License: <http://www.boost.org/LICENSE_1_0.txt>
			or doc/licenses/Boost.lic
 - (c) 2015 OpenCV (www.opencv.org)
		License: <http://opencv.org/license.html>
            or doc/licenses/OpenCV.lic
 - (c) 2015 The FreeType Project (www.freetype.org)
		License: <http://git.savannah.gnu.org/cgit/freetype/freetype2.git/plain/docs/FTL.TXT>
	        or doc/licenses/FTL.txt
 - (c) 1997-2002 OpenMP Architecture Review Board (www.openmp.org)
   (c) Microsoft Corporation (Visual C++ implementation for OpenMP C/C++ Version 2.0 March 2002)
		See: <https://msdn.microsoft.com/en-us/library/8y6825x5(v=vs.140).aspx>
 - (c) 1995-2013 zlib software (Jean-loup Gailly and Mark Adler - see: www.zlib.net)
		License: <http://www.zlib.net/zlib_license.html>
            or doc/licenses/zlib.lic
 
 (c) 2016, 2017 Florin Tulba <florintulba@yahoo.com>

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
 ***********************************************************************************************/

#include "filledRectanglesFilter.h"
#include "symFilterCache.h"
#include "pixMapSymBase.h"
#include "misc.h"

#pragma warning ( push, 0 )

#include <opencv2/imgproc/imgproc.hpp>

#pragma warning ( pop )

using namespace std;
using namespace cv;

/**
Analyzes a horizontal / vertical projection (reduction sum) of the glyph,
checking for clues of rectangular blocks: a projection with one / several
adjacent indices holding the maximum value.

@param sums the projection (reduction sum) of the glyph in horizontal / vertical direction
@param sideLen height / width of glyph's bounding box
@param countOfMaxSums [out] determined side length of a possible rectangle: [1..sideLen]

@return true if the projection denotes a valid uncut rectangular block
*/
bool FilledRectanglesFilter::checkProjectionForFilledRectangles(const Mat &sums, unsigned sideLen, int &countOfMaxSums) {
#pragma warning ( disable : WARN_THREAD_UNSAFE )
	static const Mat structuringElem(1, 3, CV_8U, Scalar(1U));
#pragma warning ( default : WARN_THREAD_UNSAFE )

	double maxVSums;
	minMaxIdx(sums, nullptr, &maxVSums);
	const Mat sumsOnMax = (sums==maxVSums); // these should be the white rows/columns
	countOfMaxSums = countNonZero(sumsOnMax); // 1..sideLen
	if(countOfMaxSums == (int)sideLen) // the white rows/columns are consecutive, for sure
		return true;

	if(countOfMaxSums == 1)
		return true; // a single white row/column is a (degenerate) rectangle

	if(countOfMaxSums == 2) {
		// pad sumsOnMax and then dilate it
		Mat paddedSumsOnMax(1, sumsOnMax.cols+2, CV_8U, Scalar(0U));
		sumsOnMax.copyTo((const Mat&)Mat(paddedSumsOnMax, Range::all(), Range(1, sumsOnMax.cols+1)));
		dilate(paddedSumsOnMax, paddedSumsOnMax, structuringElem);
		if(countNonZero(paddedSumsOnMax) - countOfMaxSums > 2)
			return false; // there was at least one gap, so dilation filled more than 2 pixels
	} else { // countOfMaxSums is [3..sideLen)
		erode(sumsOnMax, sumsOnMax, structuringElem);
		if(countOfMaxSums - countNonZero(sumsOnMax) > 2)
			return false; // there was at least one gap, so erosion teared down more than 2 pixels
	}
	return true;
}

FilledRectanglesFilter::FilledRectanglesFilter(unique_ptr<ISymFilter> nextFilter_/* = nullptr*/) :
		TSymFilter(0U, "filled rectangles", std::move(nextFilter_)) {}

bool FilledRectanglesFilter::isDisposable(const IPixMapSym &pms, const SymFilterCache&) {
	if(!isEnabled())
		THROW_WITH_CONST_MSG(__FUNCTION__ " should be called only for enabled filters!", logic_error);

	const Mat narrowGlyph = pms.asNarrowMat();
	double brightestVal;
	minMaxIdx(narrowGlyph, nullptr, &brightestVal);
	const int brightestPixels = countNonZero(narrowGlyph == (unsigned char)brightestVal);
	if(brightestPixels < 3)
		return false; // don't report really small areas

	int rowsWithMaxSum, colsWithMaxSum; // sides of a possible filled rectangle

	// Analyze the horizontal and vertical projections of pms, looking for signs of rectangle symbols
	if(!checkProjectionForFilledRectangles(pms.getRowSums(), pms.getRows(), rowsWithMaxSum))
		return false;
	if(!checkProjectionForFilledRectangles(pms.getColSums(), pms.getCols(), colsWithMaxSum))
		return false;
	if(brightestPixels != colsWithMaxSum * rowsWithMaxSum)
		return false; // rectangle's area should be the product of its sides

	return true;
}