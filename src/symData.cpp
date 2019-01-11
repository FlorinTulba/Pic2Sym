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
 If not, see <http://www.gnu.org/licenses/agpl-3.0.txt>.
 ***********************************************************************************************/

#include "symData.h"
#include "pixMapSymBase.h"
#include "structuralSimilarity.h"
#include "blurBase.h"
#include "misc.h"

using namespace std;
using namespace cv;

unsigned SymData::VERSION_FROM_LAST_IO_OP = UINT_MAX;

SymData::SymData(const Mat &negSym_, const cv::Mat &symMiu0_,
				 unsigned long code_, size_t symIdx_, 
				 double minVal_, double diffMinMax_, 
				 double avgPixVal_, double normSymMiu0_,
				 const Point2d &mc_, 
				 const MatArray &masks_, 
				 bool removable_/* = false*/) :
	code(code_), symIdx(symIdx_), minVal(minVal_), diffMinMax(diffMinMax_),
	avgPixVal(avgPixVal_), normSymMiu0(normSymMiu0_), mc(mc_),
	negSym(negSym_), symMiu0(symMiu0_),
	removable(removable_), masks(masks_) {}

SymData::SymData(const IPixMapSym &pms, unsigned sz, bool forTinySym) :
		code(pms.getSymCode()), symIdx(pms.getSymIdx()),
		avgPixVal(pms.getAvgPixVal()),
		mc(pms.getMc()),
		negSym(pms.toMat(sz, true)),
		removable(pms.isRemovable()) {
	computeFields(pms.toMatD01(sz), *this, forTinySym);
}

SymData::SymData(unsigned long code_/* = ULONG_MAX*/, size_t symIdx_/* = 0ULL*/,
				 double avgPixVal_/* = 0.*/, const Point2d &mc_/* = Point2d(.5, .5)*/) :
	code(code_), symIdx(symIdx_), avgPixVal(avgPixVal_), mc(mc_) {}

SymData::SymData(const Point2d &mc_, double avgPixVal_) : avgPixVal(avgPixVal_), mc(mc_) {}

SymData::SymData(const SymData &other) : code(other.code), symIdx(other.symIdx),
		minVal(other.minVal), diffMinMax(other.diffMinMax),
		avgPixVal(other.avgPixVal), normSymMiu0(other.normSymMiu0), mc(other.mc),
		negSym(other.negSym), symMiu0(other.symMiu0),
		removable(other.removable), masks(other.masks) {}

// Delegating constructors for virtual inheritance triggers this warning in VS2013:
// https://connect.microsoft.com/VisualStudio/feedback/details/774986/codename-milan-delegating-constructors-causes-warning-c4100-initvbases-unreferenced-formal-parameter-w4-when-derived-class-uses-virtual-inheritance
#pragma warning( disable : WARN_UNREF_FORMAL_PARAM )
SymData::SymData(SymData &&other) : SymData(other) {
	other.negSym.release();
	other.symMiu0.release();
	for(Mat &m : other.masks)
		m.release();
}
#pragma warning( default : WARN_UNREF_FORMAL_PARAM )

SymData& SymData::operator=(const SymData &other) {
	if(this != &other) {
#define REPLACE_FIELD(Field) \
		Field = other.Field

		REPLACE_FIELD(code);
		REPLACE_FIELD(symIdx);
		REPLACE_FIELD(minVal);
		REPLACE_FIELD(diffMinMax);
		REPLACE_FIELD(avgPixVal);
		REPLACE_FIELD(normSymMiu0);
		REPLACE_FIELD(mc);
		REPLACE_FIELD(negSym);
		REPLACE_FIELD(symMiu0);
		REPLACE_FIELD(removable);

		for(int i = 0; i < ISymData::MATRICES_COUNT; ++i)
			REPLACE_FIELD(masks[(size_t)i]);

#undef REPLACE_FIELD
	}

	return *this;
}

SymData& SymData::operator=(SymData &&other) {
	operator=(other);

	if(this != &other) {
		other.negSym.release();
		other.symMiu0.release();

		for(Mat &m : other.masks)
			m.release();
	}

	return *this;
}

const Point2d& SymData::getMc() const { return mc; }
const Mat& SymData::getNegSym() const { return negSym; }
const Mat& SymData::getSymMiu0() const { return symMiu0; }
double SymData::getNormSymMiu0() const { return normSymMiu0; }
const ISymData::MatArray& SymData::getMasks() const { return masks; }
size_t SymData::getSymIdx() const { return symIdx; }
#ifdef UNIT_TESTING
double SymData::getMinVal() const { return minVal; }
#endif // UNIT_TESTING defined
double SymData::getDiffMinMax() const { return diffMinMax; }
double SymData::getAvgPixVal() const { return avgPixVal; }
unsigned long SymData::getCode() const { return code; }
bool SymData::isRemovable() const { return removable; }

bool SymData::olderVersionDuringLastIO() {
	return VERSION_FROM_LAST_IO_OP < VERSION;
}

void SymData::computeSymMiu0Related(const cv::Mat &glyph, double miu, SymData &sd) {
	sd.symMiu0 = glyph - miu;
	sd.normSymMiu0 = norm(sd.symMiu0, NORM_L2);
}

void SymData::computeFields(const cv::Mat &glyph, SymData &sd, bool forTinySym) {
	extern const double EPSp1();

	computeSymMiu0Related(glyph, sd.avgPixVal, sd);

	/*
	SymData_computeFields_STILL_BG and STILL_FG from below are constants for foreground / background thresholds.

	1/255 = 0.00392, so 0.004 tolerates pixels with 1 brightness unit less / more than ideal
	STILL_BG was set to 0, as there are font families with extremely similar glyphs.
	When Unit Testing shouldn't identify exactly each glyph, STILL_BG might be > 0.
	But testing on 'BPmonoBold.ttf' does tolerate such larger values (0.025, for instance).
	*/
	extern const double SymData_computeFields_STILL_BG;					// darkest shades

#pragma warning( disable : WARN_THREAD_UNSAFE )
	static const double STILL_FG = 1. - SymData_computeFields_STILL_BG;	// brightest shades
#pragma warning( default : WARN_THREAD_UNSAFE )

	double minVal, maxVal;
	minMaxIdx(glyph, &minVal, &maxVal);
	assert(maxVal < EPSp1()); // ensures diffMinMax, groundedGlyph and blurOfGroundedGlyph are within 0..1

	sd.minVal = minVal;
	const double diffMinMax = sd.diffMinMax = maxVal - minVal;
	const Mat groundedGlyph = sd.masks[GROUNDED_SYM_IDX] =
		(minVal==0. ? glyph.clone() : (glyph - minVal)); // min val on 0

	sd.masks[FG_MASK_IDX] = (glyph >= (minVal + STILL_FG * diffMinMax));
	sd.masks[BG_MASK_IDX] = (glyph <= (minVal + SymData_computeFields_STILL_BG * diffMinMax));

	// Storing a blurred version of the grounded glyph for structural similarity match aspect
	Mat blurOfGroundedGlyph;
	StructuralSimilarity::supportBlur.process(groundedGlyph, blurOfGroundedGlyph, forTinySym);
	sd.masks[BLURRED_GR_SYM_IDX] = blurOfGroundedGlyph;
	// edgeMask selects all pixels that are not minVal, nor maxVal
	inRange(glyph, minVal+EPS, maxVal-EPS, sd.masks[EDGE_MASK_IDX]);

	// Storing also the variance of the grounded glyph for structural similarity match aspect
	// Actual varianceOfGroundedGlyph is obtained in the subtraction after the blur
	Mat blurOfGroundedGlyphSquared;
	StructuralSimilarity::supportBlur.process(groundedGlyph.mul(groundedGlyph),
											  blurOfGroundedGlyphSquared, forTinySym);

	sd.masks[VARIANCE_GR_SYM_IDX] =
		blurOfGroundedGlyphSquared - blurOfGroundedGlyph.mul(blurOfGroundedGlyph);
}
