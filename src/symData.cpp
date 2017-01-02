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

#include "symData.h"
#include "structuralSimilarity.h"
#include "blur.h"
#include "misc.h"

#pragma warning ( push, 0 )

#include <opencv2/imgproc/imgproc.hpp>

#pragma warning ( pop )

using namespace std;
using namespace cv;

/*
SymData_computeFields_STILL_BG and STILL_FG from below are constants for foreground / background thresholds.

1/255 = 0.00392, so 0.004 tolerates pixels with 1 brightness unit less / more than ideal
STILL_BG was set to 0, as there are font families with extremely similar glyphs.
When Unit Testing shouldn't identify exactly each glyph, STILL_BG might be > 0.
But testing on 'BPmonoBold.ttf' does tolerate such larger values (0.025, for instance).
*/
extern const double SymData_computeFields_STILL_BG;					// darkest shades
static const double STILL_FG = 1. - SymData_computeFields_STILL_BG;	// brightest shades

extern const double EPSp1();

SymData::SymData(const Mat &negSym_, unsigned long code_, size_t symIdx_, double minVal_, double diffMinMax_, 
				 double avgPixVal_, const Point2d &mc_, const MatArray &masks_, bool removable_/* = false*/) :
	code(code_), symIdx(symIdx_), minVal(minVal_), diffMinMax(diffMinMax_),
	avgPixVal(avgPixVal_), mc(mc_), negSym(negSym_), removable(removable_), masks(masks_) {}

SymData::SymData(unsigned long code_/* = ULONG_MAX*/, size_t symIdx_/* = 0U*/,
				 double avgPixVal_/* = 0.*/, const cv::Point2d &mc_/* = Point2d(.5, .5)*/) :
	code(code_), symIdx(symIdx_), avgPixVal(avgPixVal_), mc(mc_) {}

SymData::SymData(const cv::Point2d &mc_, double avgPixVal_) : avgPixVal(avgPixVal_), mc(mc_) {}

SymData::SymData(const SymData &other) : code(other.code), symIdx(other.symIdx),
		minVal(other.minVal), diffMinMax(other.diffMinMax),
		avgPixVal(other.avgPixVal), mc(other.mc),
		negSym(other.negSym), removable(other.removable), masks(other.masks) {}

SymData::SymData(SymData &&other) : SymData(other) {
	other.negSym.release();
		for(auto &m : other.masks)
			m.release();
}

SymData& SymData::operator=(const SymData &other) {
	if(this != &other) {
#define REPLACE_FIELD(Field) \
		Field = other.Field

		REPLACE_FIELD(code);
		REPLACE_FIELD(symIdx);
		REPLACE_FIELD(minVal);
		REPLACE_FIELD(diffMinMax);
		REPLACE_FIELD(avgPixVal);
		REPLACE_FIELD(mc);
		REPLACE_FIELD(negSym);
		REPLACE_FIELD(removable);

		for(int i = 0; i < SymData::MATRICES_COUNT; ++i)
			REPLACE_FIELD(masks[(size_t)i]);

#undef REPLACE_FIELD
	}

	return *this;
}

SymData& SymData::operator=(SymData &&other) {
	operator=(other);

	if(this != &other) {
		other.negSym.release();

		for(auto &m : other.masks)
			m.release();
	}

	return *this;
}

void SymData::computeFields(const Mat &glyph, Mat &fgMask, Mat &bgMask, Mat &edgeMask,
							Mat &groundedGlyph, Mat &blurOfGroundedGlyph, Mat &varianceOfGroundedGlyph,
							double &minVal, double &diffMinMax, bool forTinySym) {
	double maxVal;
	minMaxIdx(glyph, &minVal, &maxVal);
	assert(maxVal < EPSp1()); // ensures diffMinMax, groundedGlyph and blurOfGroundedGlyph are within 0..1

	diffMinMax = maxVal - minVal;
	groundedGlyph = (minVal==0. ? glyph.clone() : (glyph - minVal)); // min val on 0

	fgMask = (glyph >= (minVal + STILL_FG * diffMinMax));
	bgMask = (glyph <= (minVal + SymData_computeFields_STILL_BG * diffMinMax));

	// Storing a blurred version of the grounded glyph for structural similarity match aspect
	StructuralSimilarity::supportBlur.process(groundedGlyph, blurOfGroundedGlyph, forTinySym);

	// edgeMask selects all pixels that are not minVal, nor maxVal
	inRange(glyph, minVal+EPS, maxVal-EPS, edgeMask);

	// Storing also the variance of the grounded glyph for structural similarity match aspect
	// Actual varianceOfGroundedGlyph is obtained in the subtraction after the blur
	StructuralSimilarity::supportBlur.process(groundedGlyph.mul(groundedGlyph),
											  varianceOfGroundedGlyph, forTinySym);

	varianceOfGroundedGlyph -= blurOfGroundedGlyph.mul(blurOfGroundedGlyph);
}
