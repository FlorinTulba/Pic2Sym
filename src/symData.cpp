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

#include "symData.h"
#include "structuralSimilarity.h"
#include "blur.h"
#include "misc.h"

#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

SymData::SymData(unsigned long code_, size_t symIdx_, double minVal_, double diffMinMax_, double pixelSum_,
				 const Point2d &mc_, const MatArray &symAndMasks_) :
	code(code_), symIdx(symIdx_), minVal(minVal_), diffMinMax(diffMinMax_),
	pixelSum(pixelSum_), mc(mc_), symAndMasks(symAndMasks_) {}

SymData::SymData() : symAndMasks({ { Mat(), Mat(), Mat(), Mat(), Mat(), Mat(), Mat() } }) {}

SymData::SymData(const SymData &other) : code(other.code), symIdx(other.symIdx),
		minVal(other.minVal), diffMinMax(other.diffMinMax),
		pixelSum(other.pixelSum), mc(other.mc), symAndMasks(other.symAndMasks) {}

SymData::SymData(SymData &&other) : SymData(other) {
	for(int i = 0; i < SymData::MATRICES_COUNT; ++i)
		const_cast<Mat&>(other.symAndMasks[i]).release();
}

SymData& SymData::operator=(const SymData &other) {
	if(this != &other) {
#define REPLACE_FIELD(Field, Type) \
		const_cast<Type&>(Field) = other.Field

		REPLACE_FIELD(code, unsigned long);
		REPLACE_FIELD(symIdx, size_t);
		REPLACE_FIELD(minVal, double);
		REPLACE_FIELD(diffMinMax, double);
		REPLACE_FIELD(pixelSum, double);
		REPLACE_FIELD(mc, Point2d);
		REPLACE_FIELD(code, unsigned long);
		for(int i = 0; i < SymData::MATRICES_COUNT; ++i)
			REPLACE_FIELD(symAndMasks[i], Mat);

#undef REPLACE_FIELD
	}

	return *this;
}

SymData& SymData::operator=(SymData &&other) {
	operator=(other);

	if(this != &other) {
		for(int i = 0; i < SymData::MATRICES_COUNT; ++i)
			const_cast<Mat&>(other.symAndMasks[i]).release();
	}

	return *this;
}

void SymData::computeFields(const Mat &glyph, Mat &fgMask, Mat &bgMask, Mat &edgeMask,
							Mat &groundedGlyph, Mat &blurOfGroundedGlyph, Mat &varianceOfGroundedGlyph,
							double &minVal, double &maxVal) {
	// constants for foreground / background thresholds
	// 1/255 = 0.00392, so 0.004 tolerates pixels with 1 brightness unit less / more than ideal
	// STILL_BG was set to 0, as there are font families with extremely similar glyphs.
	// When Unit Testing shouldn't identify exactly each glyph, STILL_BG might be > 0.
	// But testing on 'BPmonoBold.ttf' does tolerate such larger values (0.025, for instance).
	extern const double SymData_computeFields_STILL_BG;					// darkest shades
	static const double STILL_FG = 1. - SymData_computeFields_STILL_BG;	// brightest shades

	minMaxIdx(glyph, &minVal, &maxVal);
	groundedGlyph = (minVal==0. ? glyph : (glyph - minVal)); // min val on 0

	fgMask = (glyph >= (minVal + STILL_FG * (maxVal-minVal)));
	bgMask = (glyph <= (minVal + SymData_computeFields_STILL_BG * (maxVal-minVal)));

	// Storing a blurred version of the grounded glyph for structural similarity match aspect
	StructuralSimilarity::supportBlur.process(groundedGlyph, blurOfGroundedGlyph);

	// edgeMask selects all pixels that are not minVal, nor maxVal
	inRange(glyph, minVal+EPS, maxVal-EPS, edgeMask);

	// Storing also the variance of the grounded glyph for structural similarity match aspect
	// Actual varianceOfGroundedGlyph is obtained in the subtraction after the blur
	StructuralSimilarity::supportBlur.process(groundedGlyph.mul(groundedGlyph), varianceOfGroundedGlyph);

	varianceOfGroundedGlyph -= blurOfGroundedGlyph.mul(blurOfGroundedGlyph);
}
