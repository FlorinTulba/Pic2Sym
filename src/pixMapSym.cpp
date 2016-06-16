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

#include "presentCmap.h"
#include "pixMapSym.h"
#include "misc.h"

#include <numeric>

#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

extern const double MinAreaRatioForUncutBlocksBB;
extern const double MinCoveredPixelsRatioForSmallUnreadableSyms;
extern const double MinWhiteAreaForUncutBlocksBB;
extern const double MinAreaRatioForUnreadableSymsBB;
extern const double StillForegroundThreshold;
extern const double ForegroundThresholdDelta;
extern const double MinBulkinessForUnreadableSyms;
extern const double MinAvgBrightnessForUnreadableSmallSyms;

namespace {
	/// Minimal glyph shifting and cropping or none to fit the bounding box
	void fitGlyphToBox(const FT_Bitmap &bm, const FT_BBox &bb,
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

	/**
	Analyzes a horizontal / vertical projection (reduction sum) of the glyph,
	checking for clues of uncut rectangular blocks: a projection with several (at least 2)
	adjacent indices holding the maximum value.

	@param sums the projection (reduction sum) of the glyph in horizontal / vertical direction
	@param sideLen height / width of glyph's bounding box
	@param countOfMaxSums [out] determined side length of a possible rectangle: [1..sideLen]
	
	@return true if the projection denotes a valid uncut rectangular block
	*/
	bool checkProjectionForUncutBlock(const Mat &sums, unsigned sideLen, int &countOfMaxSums) {
		static const Mat structuringElem(1, 3, CV_8U, Scalar(1U));

		double maxVSums;
		minMaxIdx(sums, nullptr, &maxVSums);
		const Mat sumsOnMax = (sums==maxVSums); // these should be the white rows/columns
		countOfMaxSums = countNonZero(sumsOnMax); // 1..sideLen
		if(countOfMaxSums == (int)sideLen) // the white rows/columns are consecutive, for sure
			return true;

		if(countOfMaxSums == 1)
			return false; // a single white row/column isn't an uncut block

		if(countOfMaxSums == 2) {
			// pad sumsOnMax and then dilate it
			Mat paddedSumsOnMax(1, sumsOnMax.cols+2, CV_8U, Scalar(0U));
			sumsOnMax.copyTo(Mat(paddedSumsOnMax, Range::all(), Range(1, sumsOnMax.cols+1)));
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

	/// Determines whether pmc is an uncut block symbol
	bool isAnUncutBlock(const PixMapSym &pmc, unsigned height, unsigned width, unsigned bbArea,
						 double sz2) {
		// uncut blocks usually are large (~>25% of square)
		if(bbArea < MinAreaRatioForUncutBlocksBB * sz2) 
			return false;

		const Mat narrowGlyph = pmc.asNarrowMat();
		const int whitePixels = countNonZero(narrowGlyph == 255U);
		if(whitePixels < MinWhiteAreaForUncutBlocksBB * bbArea)
			return false; // white pixels don't cover enough from the bounding box

		int rowsWithMaxSum, colsWithMaxSum; // sides of a possible uncut block

		// Analyze the horizontal and vertical projections of pmc, looking for signs of uncut blocks
		if(!checkProjectionForUncutBlock(pmc.rowSums, height, rowsWithMaxSum)
		   || !checkProjectionForUncutBlock(pmc.colSums, width, colsWithMaxSum)
		   || whitePixels != colsWithMaxSum * rowsWithMaxSum) // rectangle's area should be the product of its sides
			return false;


		// By now, the symbol is for sure a white rectangle

		// Now simply avoid to report symbols like '|', '_' and '-' as uncut blocks
		// by recognizing that their thickness is usually less than 1/2.7 from their length
		double lengthOverThickness = (double)colsWithMaxSum / rowsWithMaxSum;
		if(colsWithMaxSum < rowsWithMaxSum)
			lengthOverThickness = 1./lengthOverThickness;
		return lengthOverThickness < 2.7; // for a smaller ratio, the sides are more balanced
	}

	/**
	Determines if symbol pmc appears unreadable.

	Best approach would take size 50 of the glyph and compare its features with the ones found in the
	current font size version of the symbol. However, this would still produce some mislabeled cases
	and besides, humans do normally recognize many symbols even when some of their features are missing.

	Supervised machine learning would be the ideal solution here, since:
	- humans can label corner-cases
	- font sizes are 7..50
	- there are lots of mono-spaced font families, each with several encodings,
	each with several styles (Regular, Italic, Bold) and each with 100..30000 symbols

	A basic criteria for selecting unreadable symbols is avoiding the ones with compact
	rectangular / elliptical areas, larger than 20 - 25% of the side of the enclosing square.
	This would work for most well-known glyphs, but it fails for solid glyphs.

	Apart from the ugly, various sizes rectangular monolithic glyphs, there are some interesting
	solid symbols which could be preserved: filled triangles, circles, playing cards suits, smilies,
	dices, arrows a.o.

	The current implementation is a compromise surprising the fact that smaller fonts are
	progressively less readable.
	*/
	bool isAnUnreadableSym(const PixMapSym &pmc, unsigned fontSz, unsigned bbArea, double sz2) {
		// Usually, fonts of size >= 20 are quite readable
		if(fontSz >= 20)
			return false;

		// Usually unreadable syms are not small
		if(bbArea < MinAreaRatioForUnreadableSymsBB * sz2)
			return false;

/*
		// Fonts smaller than 10, who cover > MinCoveredPixelsRatioForSmallUnreadableSyms% are not that readable
		if(fontSz < 10) {
			const bool isPixelSumLarge = 255.*pmc.glyphSum > sz2 * MinAvgBrightnessForUnreadableSmallSyms;
			if(isPixelSumLarge)
				return true;

			const int nonZero = countNonZero(pmc.asNarrowMat());
			return nonZero > MinCoveredPixelsRatioForSmallUnreadableSyms * sz2;
		}
*/

		static const Point defAnchor(-1, -1);
		const int winSideSE = (fontSz > 15U) ? 5 : 3; // masks need to be larger for larger fonts
		const Size winSE(winSideSE, winSideSE);

		Mat glyph = pmc.toMatD01(fontSz), glyphBinAux;

		// adaptiveThreshold has some issues on uniform areas, so here's a customized implementation
		boxFilter(glyph, glyphBinAux, -1, winSE, defAnchor, true, BORDER_CONSTANT);
		double toSubtract = StillForegroundThreshold;
		if(fontSz < 15U) { // for small fonts, thresholds should be much lower, to encode perceived pixel agglomeration
			const double delta = (15U-fontSz) * ForegroundThresholdDelta/255.;
			toSubtract += delta;
		}

		// lower the threshold, but keep all values positive
		glyphBinAux -= toSubtract;
		glyphBinAux.setTo(0., glyphBinAux < 0.);
		glyphBinAux = glyph > glyphBinAux;

		// pad the thresholded matrix with 0-s all around, to allow distanceTransform consider the borders as 0-s
		const int frameSz = (int)fontSz + 2;
		const Range innerFrame(1, (int)fontSz + 1);
		Mat glyphBin = Mat(frameSz, frameSz, CV_8UC1, Scalar(0U)), glyph32f;
		glyphBinAux.copyTo(Mat(glyphBin, innerFrame, innerFrame));
		distanceTransform(glyphBin, glyph32f, DIST_L2, DIST_MASK_PRECISE);
		
		double maxValAllowed = 1.9; // just below 2 (2 means there are sections at least 3 pixels thick)
		if(fontSz > 10U)
			maxValAllowed = fontSz/5.; // larger symbols have their core in a thicker shell (>2)

		Mat symCore = glyph32f > maxValAllowed;
		const unsigned notAllowedCount = (unsigned)countNonZero(symCore);
		if(notAllowedCount > fontSz - 6U)
			return true; // limit the count of symCore pixels

		double maxVal; // largest symbol depth
		minMaxIdx(glyph32f, nullptr, &maxVal, nullptr, nullptr, symCore);

		return maxVal - maxValAllowed > 1.; // limit the depth of the symCore
	}
} // anonymous namespace

PixMapSym::PixMapSym(unsigned long symCode_,		// the symbol code
					 const FT_Bitmap &bm,			// the bitmap to process
					 int leftBound, int topBound,	// initial position of the symbol
					 int sz,						// font size
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

	computeMcAndGlyphSum((unsigned)sz, pixels, rows, cols, left, top, consec, revConsec,
						 mc, glyphSum, &colSums, &rowSums);
}

PixMapSym::PixMapSym(const PixMapSym &other) :
		symCode(other.symCode),
		glyphSum(other.glyphSum),
		mc(other.mc),
		rows(other.rows), cols(other.cols), left(other.left), top(other.top),
		pixels(other.pixels),
		rowSums(other.rowSums), colSums(other.colSums),
		removable(other.removable) {}

PixMapSym::PixMapSym(PixMapSym &&other) : // required by some vector manipulations
		symCode(other.symCode),
		glyphSum(other.glyphSum),
		mc(other.mc),
		rows(other.rows), cols(other.cols), left(other.left), top(other.top),
		pixels(move(other.pixels)),
		rowSums(other.rowSums), colSums(other.colSums),
		removable(other.removable) {
}

PixMapSym& PixMapSym::operator=(PixMapSym &&other) {
	if(this != &other) {
		symCode = other.symCode;
		glyphSum = other.glyphSum;
		mc = other.mc;
		rows = other.rows; cols = other.cols;
		left = other.left; top = other.top;
		pixels = move(other.pixels);
		rowSums = other.rowSums;
		colSums = other.colSums;
		removable = other.removable;
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

Mat PixMapSym::asNarrowMat() const {
	return Mat((int)rows, (int)cols, CV_8UC1, (void*)pixels.data());
}

Mat PixMapSym::toMat(unsigned fontSz, bool inverse/* = false*/) const {
	Mat result((int)fontSz, (int)fontSz, CV_8UC1, Scalar(inverse ? 255U : 0U));

	const int firstRow = (int)fontSz-(int)top-1;
	Mat region(result,
			   Range(firstRow, firstRow+(int)rows),
			   Range((int)left, (int)(left+cols)));
	const Mat pmsData = inverse ? (255U - asNarrowMat()) : asNarrowMat();
	pmsData.copyTo(region);

	return result;
}

Mat PixMapSym::toMatD01(unsigned fontSz) const {
	static const double INV_255 = 1./255;

	Mat result((int)fontSz, (int)fontSz, CV_64FC1, Scalar(0.));

	const int firstRow = (int)fontSz-(int)top-1;
	Mat region(result,
			   Range(firstRow, firstRow+(int)rows),
			   Range((int)left, (int)(left+cols)));

	Mat pmsData = asNarrowMat();
	pmsData.convertTo(pmsData, CV_64FC1, INV_255); // convert to double
	pmsData.copyTo(region);

	return result;
}

void PixMapSym::computeMcAndGlyphSum(unsigned sz, const vector<unsigned char> &pixels_,
									 unsigned char rows_, unsigned char cols_,
									 unsigned char left_, unsigned char top_,
									 const Mat &consec, const Mat &revConsec,
									 Point2d &mc, double &glyphSum,
									 Mat *colSums/* = nullptr*/, Mat *rowSums/* = nullptr*/) {
	const int diagsCount = (int)(sz<<1) | 1;
	const double centerCoord = (sz-1U)/2.;
	const Point2d center(centerCoord, centerCoord);
	if(colSums) *colSums = Mat::zeros(1, sz, CV_64FC1);
	if(rowSums) *rowSums = Mat::zeros(1, sz, CV_64FC1);
	if(rowSums) *rowSums = Mat::zeros(1, sz, CV_64FC1);

	if(rows_ == 0U || cols_ == 0U) {
		mc = center; glyphSum = 0.;
		return;
	}

	const Mat glyph((int)rows_, (int)cols_, CV_8UC1, (void*)pixels_.data());
	Mat sumPerColumn, sumPerRow;

	reduce(glyph, sumPerColumn, 0, CV_REDUCE_SUM, CV_64F); // sum all rows
	glyphSum = *sum(sumPerColumn).val / 255.;
	
	extern const unsigned Settings_MAX_FONT_SIZE;
	static const double NO_PIXELS_SET_THRESHOLD = .9/(255U*Settings_MAX_FONT_SIZE*Settings_MAX_FONT_SIZE);
	if(glyphSum < NO_PIXELS_SET_THRESHOLD) {
		mc = center; glyphSum = 0.;
		return;
	}

	reduce(glyph, sumPerRow, 1, CV_REDUCE_SUM, CV_64F); // sum all columns
	Range leftRange((int)left_, (int)(left_+cols_)), topRange((int)(top_-rows_)+1, (int)top_+1);
	if(rowSums) {
		Mat sumPerRowTransposed = sumPerRow.t()/255.,
			destRegion(*rowSums, Range::all(), topRange);
		sumPerRowTransposed.copyTo(destRegion);
	}
	if(colSums) {
		Mat destRegion(*colSums, Range::all(), leftRange);
		Mat(sumPerColumn/255.).copyTo(destRegion);
	}

	const double sumX = sumPerColumn.dot(Mat(consec, Range::all(), leftRange)),
				sumY = sumPerRow.dot(Mat(revConsec, topRange));

	mc = Point2d(sumX, sumY) / (255. * glyphSum);
}

PmsCont::PmsCont(const IPresentCmap &cmapViewUpdater_) :
		// Some methods from IPresentCmap are non-const => removing const:
		cmapViewUpdater(cmapViewUpdater_) {}

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

unsigned PmsCont::getUncutBlocksCount() const {
	if(!ready) {
		cerr<<__FUNCTION__ " cannot be called before setAsReady"<<endl;
		throw logic_error(__FUNCTION__ " cannot be called before setAsReady");
	}

	return uncutBlocks;
}

unsigned PmsCont::getUnreadableCount() const {
	if(!ready) {
		cerr<<__FUNCTION__ " cannot be called before setAsReady"<<endl;
		throw logic_error(__FUNCTION__ " cannot be called before setAsReady");
	}

	return unreadable;
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
	blanks = duplicates = uncutBlocks = unreadable = 0U;
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

	extern const bool PreserveRemovableSymbolsForExamination;

	const FT_Bitmap b = g->bitmap;
	const unsigned height = b.rows, width = b.width;

	// Skip Space characters
	if(height==0U || width==0U) {
		++blanks;
		return;
	}

	const PixMapSym pmc(c, g->bitmap, g->bitmap_left, g->bitmap_top,
						(int)fontSz, consec, revConsec, bb);
	if(pmc.glyphSum < EPS || sz2 - pmc.glyphSum < EPS) { // discard disguised Space characters
		++blanks;
		return;
	}

	// Exclude duplicates, as well
	for(const auto &prevPmc : syms)
		if(prevPmc == pmc) {
			++duplicates;
			return;
		}

	const unsigned bbArea = height * width;

	// Initially I considered removing only these ugly, bulky, rectangular symbols.
	if(isAnUncutBlock(pmc, height, width, bbArea, sz2)) {
		++uncutBlocks;
		if(!PreserveRemovableSymbolsForExamination)
			return;
		const_cast<PixMapSym&>(pmc).removable = true;
		goto labelAppendPmcToSyms;
	}

	// Later I thought that unreadable glyphs would just annoy the user when inspecting the approximated image
	if(isAnUnreadableSym(pmc, fontSz, bbArea, sz2)) {
		++unreadable;
		if(!PreserveRemovableSymbolsForExamination)
			return;
		const_cast<PixMapSym&>(pmc).removable = true;
		goto labelAppendPmcToSyms; // just in case more tests will appear inbetween
	}

	// Add here any other filters to be applied on the charmap
	
labelAppendPmcToSyms:
	syms.push_back(move(pmc));

	const_cast<IPresentCmap&>(cmapViewUpdater).display1stPageIfFull(syms);
}

void PmsCont::setAsReady() {
	if(ready)
		return;

	sort(BOUNDS(syms), // just to appear familiar while visualizing the cmap
		 [] (const PixMapSym &first, const PixMapSym &second) {
		return first.glyphSum < second.glyphSum;
	});

	// Determine below max box coverage for smallest glyphs from the kept symsSet.
	// This will be used to favor using larger glyphs when this option is selected.
	extern const double PmsCont_SMALL_GLYPHS_PERCENT;
	const auto smallGlyphsQty = (long)round(syms.size() * PmsCont_SMALL_GLYPHS_PERCENT);
	coverageOfSmallGlyphs = next(syms.begin(), smallGlyphsQty)->glyphSum / sz2;

	ready = true;
}
