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

#include "pixMapSym.h"
#include "misc.h"

using namespace std;
using namespace cv;

extern const double INV_255();

namespace {
	const Point2d center(.5, .5);
	const double OneMinEPS = 1. - EPS;

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

} // anonymous namespace

PixMapSym::PixMapSym(unsigned long symCode_,		// the symbol code
					 size_t symIdx_,				// symbol index within cmap
					 const FT_Bitmap &bm,			// the bitmap to process
					 int leftBound, int topBound,	// initial position of the symbol
					 int sz,						// font size
					 double maxGlyphSum,			// max sum of a glyph's pixels
					 const Mat &consec,				// vector of consecutive values 0 .. sz-1
					 const Mat &revConsec,			// vector of consecutive values sz-1 .. 0
					 const FT_BBox &bb) :			// the bounding box to fit
		symCode(symCode_), symIdx(symIdx_) {
	int rows_, cols_, left_, top_, diffLeft, diffRight, diffTop, diffBottom;
	fitGlyphToBox(bm, bb, leftBound, topBound, sz, // input params
				  rows_, cols_, left_, top_, diffLeft, diffRight, diffTop, diffBottom); // output params

	knownSize(rows_, cols_);

	if(rows_ > 0 && cols_ > 0) {
		pixels.resize(size_t(rows_ * cols_));
		for(int r = 0U; r<rows_; ++r) // copy a row at a time
			memcpy_s(&pixels[size_t(r*cols_)], size_t((rows_-r)*cols_),
					&bm.buffer[size_t((r-diffTop)*bm.pitch - diffLeft)],
					(size_t)cols_);
	}

	// Considering a bounding box sz x sz with coordinates 0,0 -> (sz-1),(sz-1)
	left_ -= bb.xMin;
	top_ = (sz-1) - (bb.yMax-top_);

	knownPosition(top_, left_);

	computeMcAndAvgPixVal((unsigned)sz, maxGlyphSum, pixels, rows, cols, left, top, consec, revConsec,
						  mc, avgPixVal, &colSums, &rowSums);
}

PixMapSym::PixMapSym(const PixMapSym &other) :
		symCode(other.symCode),
		symIdx(other.symIdx),
		avgPixVal(other.avgPixVal),
		mc(other.mc),
		rows(other.rows), cols(other.cols), left(other.left), top(other.top),
		pixels(other.pixels),
		rowSums(other.rowSums), colSums(other.colSums),
		removable(other.removable) {}

PixMapSym::PixMapSym(PixMapSym &&other) : // required by some vector manipulations
		symCode(other.symCode),
		symIdx(other.symIdx),
		avgPixVal(other.avgPixVal),
		mc(other.mc),
		rows(other.rows), cols(other.cols), left(other.left), top(other.top),
		pixels(move(other.pixels)),
		rowSums(other.rowSums), colSums(other.colSums),
		removable(other.removable) {
	other.rowSums.release();
	other.colSums.release();
}

PixMapSym& PixMapSym::operator=(PixMapSym &&other) {
	if(this != &other) {
		symCode = other.symCode;
		symIdx = other.symIdx;
		avgPixVal = other.avgPixVal;
		mc = other.mc;
		rows = other.rows; cols = other.cols;
		left = other.left; top = other.top;
		pixels = move(other.pixels);
		rowSums = other.rowSums; other.rowSums.release();
		colSums = other.colSums; other.colSums.release();
		removable = other.removable;
	}
	return *this;
}

bool PixMapSym::operator==(const PixMapSym &other) const {
	if(this == &other || symCode == other.symCode || symIdx == other.symIdx)
		return true;

	return
		avgPixVal == other.avgPixVal &&
		left == other.left && top == other.top &&
		rows == other.rows && cols == other.cols &&
		mc == other.mc &&
		equal(CBOUNDS(pixels), cbegin(other.pixels));
}

void PixMapSym::knownSize(int rows_, int cols_) {
	rows = (unsigned char)rows_;
	cols = (unsigned char)cols_;
}

void PixMapSym::knownPosition(int top_, int left_) {
	top = (unsigned char)top_;
	left = (unsigned char)left_;
}

const Point2d& PixMapSym::getMc() const { return mc; }
const Mat& PixMapSym::getColSums() const { return colSums; }
const Mat& PixMapSym::getRowSums() const { return rowSums; }
size_t PixMapSym::getSymIdx() const { return symIdx; }
double PixMapSym::getAvgPixVal() const { return avgPixVal; }
unsigned long PixMapSym::getSymCode() const { return symCode; }
unsigned char PixMapSym::getRows() const { return rows; }
unsigned char PixMapSym::getCols() const { return cols; }
unsigned char PixMapSym::getLeft() const { return left; }
unsigned char PixMapSym::getTop() const { return top; }
bool PixMapSym::isRemovable() const { return removable; }
void PixMapSym::setRemovable(bool removable_/* = true*/) { removable = removable_; }

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
	Mat result((int)fontSz, (int)fontSz, CV_64FC1, Scalar(0.));

	const int firstRow = (int)fontSz-(int)top-1;
	Mat region(result,
			   Range(firstRow, firstRow+(int)rows),
			   Range((int)left, (int)(left+cols)));

	Mat pmsData = asNarrowMat();
	pmsData.convertTo(pmsData, CV_64FC1, INV_255()); // convert to double
	pmsData.copyTo(region);

	return result;
}

void PixMapSym::computeMcAndAvgPixVal(unsigned sz, double maxGlyphSum,
									  const vector<unsigned char> &pixels_,
									  unsigned char rows_, unsigned char cols_,
									  unsigned char left_, unsigned char top_,
									  const Mat &consec, const Mat &revConsec,
									  Point2d &mc, double &avgPixVal,
									  Mat *colSums/* = nullptr*/, Mat *rowSums/* = nullptr*/) {
	const double szM1 = sz - 1.;

	if(colSums) *colSums = Mat::zeros(1, (int)sz, CV_64FC1);
	if(rowSums) *rowSums = Mat::zeros(1, (int)sz, CV_64FC1);

	if(rows_ == 0U || cols_ == 0U) {
		mc = center; avgPixVal = 0.;
		return;
	}

	const Mat glyph((int)rows_, (int)cols_, CV_8UC1, (void*)pixels_.data());
	Mat sumPerColumn, sumPerRow;

	reduce(glyph, sumPerColumn, 0, CV_REDUCE_SUM, CV_64F); // sum all rows
	const double glyphSum = *sum(sumPerColumn).val;
	avgPixVal = glyphSum / maxGlyphSum;
	
	// Checking if the glyph with non-empty bounding box contains only zeros, or only ones (a Blank)
	if(avgPixVal < EPS || avgPixVal > OneMinEPS) {
		mc = center;
		return;
	}

	reduce(glyph, sumPerRow, 1, CV_REDUCE_SUM, CV_64F); // sum all columns
	Range leftRange((int)left_, (int)(left_+cols_)), topRange((int)(sz-top_)-1, (int)(sz+rows_-top_)-1);
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

	mc = Point2d(sumX, sumY) / (glyphSum * szM1);
}
