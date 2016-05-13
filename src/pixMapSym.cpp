/****************************************************************************************
 The application Pic2Sym approximates images by a
 grid of colored symbols with colored backgrounds.

 This file was created on 2016-4-9
 and belongs to the Pic2Sym project.

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

#include "pixMapSym.h"
#include "misc.h"

#include <numeric>

using namespace std;
using namespace cv;

/// Minimal glyph shifting and cropping or none to fit the bounding box
static void fitGlyphToBox(const FT_Bitmap &bm, const FT_BBox &bb,
						  int leftBound, int topBound, int sz,
						  int& rows_, int& cols_, int& left_, int& top_,
						  int& diffLeft, int& diffRight, int& diffTop, int& diffBottom) {
	rows_ = (int)bm.rows; cols_ = (int)bm.width;
	left_ = leftBound; top_ = topBound;

	diffLeft = left_ - bb.xMin; diffRight = bb.xMax - (left_ + cols_ - 1);
	if(diffLeft < 0) { // cropping left_ side?
		if(cols_+diffLeft > 0 && cols_ > sz) // not shiftable => crop
			cols_ -= min(cols_-sz, -diffLeft);
		left_ = bb.xMin;
	}
	diffLeft = 0;

	if(diffRight < 0) { // cropping right side?
		if(cols_+diffRight > 0 && cols_ > sz) // not shiftable => crop
			cols_ -= min(cols_-sz, -diffRight);
		left_ = bb.xMax - cols_ + 1;
	}

	diffTop = bb.yMax - top_; diffBottom = (top_ - rows_ + 1) - bb.yMin;
	if(diffTop < 0) { // cropping top_ side?
		if(rows_+diffTop > 0 && rows_ > sz) // not shiftable => crop
			rows_ -= min(rows_-sz, -diffTop);
		top_ = bb.yMax;
	}
	diffTop = 0;

	if(diffBottom < 0) { // cropping bottom side?
		if(rows_+diffBottom > 0 && rows_ > sz) // not shiftable => crop
			rows_ -= min(rows_-sz, -diffBottom);
		top_ = bb.yMin + rows_ - 1;
	}

	assert(top_>=bb.yMin && top_<=bb.yMax);
	assert(left_>=bb.xMin && left_<=bb.xMax);
	assert(rows_>=0 && rows_<=sz);
	assert(cols_>=0 && cols_<=sz);
}

PixMapSym::PixMapSym(unsigned long symCode_,		// the symbol code
					 const FT_Bitmap &bm,			// the bitmap to process
					 int leftBound, int topBound,	// initial position of the symbol
					 int sz, double sz2,				// font size and squared of it
					 const Mat &consec,				// vector of consecutive values 0 .. sz-1
					 const Mat &revConsec,			// vector of consecutive values sz-1 .. 0
					 const FT_BBox &bb) :			// the bounding box to fit
					 symCode(symCode_) {
	int rows_, cols_, left_, top_, diffLeft, diffRight, diffTop, diffBottom;
	fitGlyphToBox(bm, bb, leftBound, topBound, sz, // input params
				  rows_, cols_, left_, top_, diffLeft, diffRight, diffTop, diffBottom); // output params

	rows = (unsigned char)rows_;
	cols = (unsigned char)cols_;

	if(rows_ > 0 && cols_ > 0) {
		pixels.resize(rows_ * cols_);
		for(int r = 0U; r<rows_; ++r) // copy a row at a time
			memcpy_s(&pixels[r*cols_], (rows_-r)*cols_,
			&bm.buffer[(r-diffTop)*bm.pitch - diffLeft],
			cols_);
	}

	// Considering a bounding box sz x sz with coordinates 0,0 -> (sz-1),(sz-1)
	left_ -= bb.xMin;
	top_ = (sz-1) - (bb.yMax-top_);

	left = (unsigned char)left_;
	top = (unsigned char)top_;

	glyphSum = computeGlyphSum(rows, cols, pixels);
	mc = computeMc((unsigned)sz, pixels, rows, cols, left, top, glyphSum, consec, revConsec);
}

PixMapSym::PixMapSym(const PixMapSym &other) :
		symCode(other.symCode),
		glyphSum(other.glyphSum),
		mc(other.mc),
		rows(other.rows), cols(other.cols), left(other.left), top(other.top),
		pixels(other.pixels) {}

PixMapSym::PixMapSym(PixMapSym &&other) : // required by some vector manipulations
		symCode(other.symCode),
		glyphSum(other.glyphSum),
		mc(other.mc),
		rows(other.rows), cols(other.cols), left(other.left), top(other.top),
		pixels(move(other.pixels)) {}

PixMapSym& PixMapSym::operator=(PixMapSym &&other) {
	if(this != &other) {
		symCode = other.symCode;
		glyphSum = other.glyphSum;
		mc = other.mc;
		rows = other.rows; cols = other.cols;
		left = other.left; top = other.top;
		pixels = move(other.pixels);
	}
	return *this;
}

bool PixMapSym::operator==(const PixMapSym &other) const {
	if(this == &other || symCode == other.symCode)
		return true;

	return
		glyphSum == other.glyphSum &&
		left == other.left && top == other.top &&
		rows == other.rows && cols == other.cols &&
		mc == other.mc &&
		equal(CBOUNDS(pixels), cbegin(other.pixels));
}

double PixMapSym::computeGlyphSum(unsigned char rows_, unsigned char cols_,
								  const vector<unsigned char> &pixels_) {
	if(rows_ == 0U || cols_ == 0U)
		return 0.;

	return  *sum(Mat(1, (int)rows_*cols_, CV_8UC1, (void*)pixels_.data())).val / 255.;
}

const Point2d PixMapSym::computeMc(unsigned sz, const vector<unsigned char> &pixels_,
								   unsigned char rows_, unsigned char cols_,
								   unsigned char left_, unsigned char top_,
								   double glyphSum_, const Mat &consec, const Mat &revConsec) {
	if(rows_ == 0U || cols_ == 0U || glyphSum_ < .9/(255U*sz*sz)) {
		const double centerCoord = (sz-1U)/2.;
		return Point2d(centerCoord, centerCoord);
	}

	const Mat glyph((int)rows_, (int)cols_, CV_8UC1, (void*)pixels_.data());
	Mat sumPerColumn, sumPerRow;

	reduce(glyph, sumPerColumn, 0, CV_REDUCE_SUM, CV_64F); // sum all rows
	reduce(glyph, sumPerRow, 1, CV_REDUCE_SUM, CV_64F); // sum all columns

	const double sumX = sumPerColumn.dot(Mat(consec, Range::all(), Range(left_, left_+cols_))),
		sumY = sumPerRow.dot(Mat(revConsec, Range(top_+1-rows_, top_+1)));

	return Point2d(sumX, sumY) / (255. * glyphSum_);
}

unsigned PmsCont::getFontSz() const {
	if(!ready) {
		cerr<<__FUNCTION__  " cannot be called before setAsReady"<<endl;
		throw logic_error(__FUNCTION__  " cannot be called before setAsReady");
	}

	return fontSz;
}

unsigned PmsCont::getBlanksCount() const {
	if(!ready) {
		cerr<<__FUNCTION__ " cannot be called before setAsReady"<<endl;
		throw logic_error(__FUNCTION__ " cannot be called before setAsReady");
	}

	return blanks;
}

unsigned PmsCont::getDuplicatesCount() const {
	if(!ready) {
		cerr<<__FUNCTION__ " cannot be called before setAsReady"<<endl;
		throw logic_error(__FUNCTION__ " cannot be called before setAsReady");
	}

	return duplicates;
}

double PmsCont::getCoverageOfSmallGlyphs() const {
	if(!ready) {
		cerr<<__FUNCTION__ " cannot be called before setAsReady"<<endl;
		throw logic_error(__FUNCTION__ " cannot be called before setAsReady");
	}

	return coverageOfSmallGlyphs;
}

const vector<const PixMapSym>& PmsCont::getSyms() const {
	if(!ready) {
		cerr<<__FUNCTION__ " cannot be called before setAsReady"<<endl;
		throw logic_error(__FUNCTION__ " cannot be called before setAsReady");
	}

	return syms;
}

void PmsCont::reset(unsigned fontSz_/* = 0U*/, unsigned symsCount/* = 0U*/) {
	ready = false;
	fontSz = fontSz_;
	blanks = duplicates = 0U;
	sz2 = (double)fontSz_ * fontSz_;
	coverageOfSmallGlyphs = 0.;

	syms.clear();
	if(symsCount != 0U)
		syms.reserve(symsCount);

	consec = Mat(1, fontSz_, CV_64FC1);
	revConsec.release();

	iota(consec.begin<double>(), consec.end<double>(), (double)0.);
	flip(consec, revConsec, 1);
	revConsec = revConsec.t();
}

void PmsCont::appendSym(FT_ULong c, FT_GlyphSlot g, FT_BBox &bb) {
	if(ready) {
		cerr<<"Cannot " __FUNCTION__ " after setAsReady without reset-ing"<<endl;
		throw logic_error("Cannot " __FUNCTION__ " after setAsReady without reset-ing");
	}

	const FT_Bitmap b = g->bitmap;
	const unsigned height = b.rows, width = b.width;
	if(height==0U || width==0U) { // skip Space character
		++blanks;
		return;
	}

	const PixMapSym pmc(c, g->bitmap, g->bitmap_left, g->bitmap_top,
						(int)fontSz, sz2, consec, revConsec, bb);
	if(pmc.glyphSum < EPS || sz2 - pmc.glyphSum < EPS) // discard disguised Space characters
		++blanks;
	else {
		for(const auto &prevPmc : syms)
			if(prevPmc == pmc) {
				++duplicates;
				return;
			}
		syms.push_back(move(pmc));
	}
}

void PmsCont::setAsReady() {
	if(ready)
		return;

	// Determine below max box coverage for smallest glyphs from the kept symsSet.
	// This will be used to favor using larger glyphs when this option is selected.
	extern const double PmsCont_SMALL_GLYPHS_PERCENT;
	const auto smallGlyphsQty = (long)round(syms.size() * PmsCont_SMALL_GLYPHS_PERCENT);
	nth_element(syms.begin(), next(syms.begin(), smallGlyphsQty), syms.end(),
				[] (const PixMapSym &first, const PixMapSym &second) -> bool {
		return first.glyphSum < second.glyphSum;
	});
	coverageOfSmallGlyphs = next(syms.begin(), smallGlyphsQty)->glyphSum / sz2;

	sort(BOUNDS(syms), // just to appear familiar while visualizing the cmap
		 [] (const PixMapSym &first, const PixMapSym &second) -> bool {
		return first.symCode < second.symCode;
	});

	ready = true;
}
