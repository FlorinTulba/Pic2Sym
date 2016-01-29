/**********************************************************
 Project:     Pic2Sym
 File:        fontEngine.cpp

 Author:      Florin Tulba
 Created on:  2015-12-20
 
 Copyright (c) 2016 Florin Tulba
 **********************************************************/

#include "fontEngine.h"
#include "misc.h"
#include "controller.h"

#include <sstream>
#include <set>
#include <algorithm>
#include <numeric>

#include <boost/filesystem/operations.hpp>
#include FT_TRUETYPE_IDS_H

using namespace std;
using namespace cv;
using namespace boost::filesystem;
using namespace boost::bimaps;

namespace {
	/*
	adjustScaling enforces sz as vertical size and determines an optimal horizontal size,
	so that most symbols will widen enough to fill more of the drawing square,
	while preserving the designer's placement.

	The parameters <bb> and <symsCount> return the estimated future bounding box and
	respectively the count of glyphs within selected charmap.

	The returned pair will contain the scaling factors to remember.
	*/
	pair<double, double>
				adjustScaling(FT_Face face, unsigned sz, FT_BBox &bb, unsigned &symsCount) {
		vector<double> vTop, vBottom, vLeft, vRight, vHeight, vWidth;
		FT_Size_RequestRec  req;
		req.type = FT_SIZE_REQUEST_TYPE_REAL_DIM; // FT_SIZE_REQUEST_TYPE_BBOX, ...
		req.height = sz<<6; // 26.6 format
		req.width = req.height; // initial check for square drawing board
		req.horiResolution = req.vertResolution = 72U; // 72dpi is set by default by higher-level methods
		FT_Error error = FT_Request_Size(face, &req);
		if(error != FT_Err_Ok) {
			cerr<<"Couldn't set font size: "<<sz<<"  Error: "<<error<<endl;
			throw invalid_argument("Couldn't set font size!");
		}
		symsCount = 0U;
		FT_UInt idx;
		FT_ULong c = FT_Get_First_Char(face, &idx);
		while(idx != 0) { // Assess all glyphs of this charmap
			++symsCount;
			FT_Load_Char(face, c, FT_LOAD_RENDER);
			FT_GlyphSlot g = face->glyph;
			FT_Bitmap b = g->bitmap;

			unsigned height = b.rows, width = b.width;
			vHeight.push_back(height); vWidth.push_back(width);
			int left = g->bitmap_left, right = g->bitmap_left+width - 1,
				top = g->bitmap_top, bottom = top - (int)height + 1;
			vLeft.push_back(left); vRight.push_back(right);
			vTop.push_back(top); vBottom.push_back(bottom);
			c = FT_Get_Next_Char(face, c, &idx);
		}

		// Compute some means and standard deviations
		Vec<double, 1> avgHeight, sdHeight, avgWidth, sdWidth, avgTop, sdTop, avgBottom, sdBottom, avgLeft, sdLeft, avgRight, sdRight;
		meanStdDev(Mat(1, symsCount, CV_64FC1, vHeight.data()), avgHeight, sdHeight);
		meanStdDev(Mat(1, symsCount, CV_64FC1, vWidth.data()), avgWidth, sdWidth);
		meanStdDev(Mat(1, symsCount, CV_64FC1, vTop.data()), avgTop, sdTop);
		meanStdDev(Mat(1, symsCount, CV_64FC1, vBottom.data()), avgBottom, sdBottom);
		meanStdDev(Mat(1, symsCount, CV_64FC1, vLeft.data()), avgLeft, sdLeft);
		meanStdDev(Mat(1, symsCount, CV_64FC1, vRight.data()), avgRight, sdRight);

		double kv = 1., kh = 1.; // 1. means a single standard deviation => ~68% of the data

		// Enlarge factors, forcing the average width + lateral std. devs
		// to fit the width of the drawing square.
		double factorH = sz / ( *avgWidth.val + kh*( *sdLeft.val + *sdRight.val)),
			factorV = sz / ( *avgHeight.val + kv*( *sdTop.val + *sdBottom.val)); // Similar story vertically

		// Computing new height & width
		req.height = (FT_ULong)floor(factorV * req.height);
		req.width = (FT_ULong)floor(factorH * req.width);

		FT_Request_Size(face, &req); // reshaping the fonts to better fill the drawing square

		// Positioning the Bounding box to best cover the estimated future position & size of the symbols
		double yMin = factorV * ( *avgBottom.val - *sdBottom.val), // current bottom scaled by factorV
			yMax = factorV * ( *avgTop.val + *sdTop.val),			// top
			yDiff2 = (yMax-yMin+1-sz)/2.,	// the difference to divide equally between top & bottom
			xMin = factorH * ( *avgLeft.val - *sdLeft.val),		// current left scaled by factorH
			xMax = factorH * ( *avgRight.val + *sdRight.val),		// right
			xDiff2 = (xMax-xMin+1-sz)/2.;	// the difference to divide equally between left & right

		// distributing the differences
		yMin += yDiff2; yMax -= yDiff2;
		xMin += xDiff2; xMax -= xDiff2;

		// ensure yMin <= 0 (should be at most the baseline y coord, which is 0)
		if(yMin > 0) {
			yMax -= yMin;
			yMin = 0;
		}

		bb.xMin = (FT_Pos)round(xMin); bb.xMax = (FT_Pos)round(xMax);
		bb.yMin = (FT_Pos)round(yMin); bb.yMax = (FT_Pos)round(yMax);

		return make_pair(factorH, factorV);
	}

	/*
	Computing the mass center (mc) of a given glyph and its background.
	*/	
	const Point2d computeMc(unsigned sz, unsigned char *data, 
					unsigned char rows, unsigned char cols,
					unsigned char left, unsigned char top,
					double glyphSum, const Mat &consec, const Mat &revConsec) {
		if(rows == 0U || cols == 0U) {
			double centerCoord = (sz-1U)/2.;
			return Point2d(centerCoord, centerCoord);
		}

		const Mat glyph((int)rows, (int)cols, CV_8UC1, (void*)data);
		Mat sumPerColumn, sumPerRow;

		reduce(glyph, sumPerColumn, 0, CV_REDUCE_SUM, CV_64F); // sum all rows
		reduce(glyph, sumPerRow, 1, CV_REDUCE_SUM, CV_64F); // sum all columns

		double sumX = sumPerColumn.dot(Mat(consec, Range::all(), Range(left, left+cols))),
			sumY = sumPerRow.t().dot(Mat(revConsec, Range::all(), Range(top+1-rows, top+1)));

		return Point2d(sumX, sumY) / (255. * glyphSum);
	}

	// Minimal glyph shifting and cropping or none to fit the bounding box
	void fitGlyphToBox(const FT_Bitmap &bm, const FT_BBox &bb,
					   int leftBound, int topBound,
					   int& sz, int& rows_, int& cols_, int& left_, int& top_,
					   int& diffLeft, int& diffRight, int& diffTop, int& diffBottom) {
		sz = bb.yMax-bb.yMin+1;
		assert(sz > 0 && bb.xMax-bb.xMin+1 == sz);

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

	/*
	appendSym puts valid glyphs into vector <syms>. Space (empty / full) glyphs are invalid.
	Adds a new mapping between glyph's index and the position it entered <syms>.
	Updates the count of Blanks.
	*/
	void appendSym(FT_ULong c, FT_GlyphSlot g, FT_BBox &bb, vector<PixMapSym> &syms,
					unsigned &blanks, unsigned &duplicates, const double sz2) {
		static const double EPS = 1e-6;

		PixMapSym pmc(c, g->bitmap, g->bitmap_left, g->bitmap_top, bb);
		if(pmc.glyphSum < EPS || sz2 - pmc.glyphSum < EPS) // discard disguised Space characters
			++blanks;
		else {
			for(auto &prevPmc : syms)
				if(prevPmc == pmc) {
					++duplicates;
					return;
				}
			syms.emplace_back(move(pmc));
		}
	}

	// Creates a bimap using initializer_list. Needed in 'encodingsMap' below
	template <typename L, typename R>
	bimap<L, R> make_bimap(initializer_list<typename bimap<L, R>::value_type> il) {
		return bimap<L, R>(BOUNDS(il));
	}

	// Holds the mapping between encodings codes and their corresponding names
	// Returns non-const just to allow accessing the map with operator[].
	bimap<FT_Encoding, string>& encodingsMap() {

		// Defines pairs like { FT_ENCODING_ADOBE_STANDARD, "ADOBE_STANDARD" }
		#define enc(encValue) { encValue, string(#encValue).substr(12) }

		static bimap<FT_Encoding, string> encMap(
			make_bimap<FT_Encoding, string>({ // known encodings
				enc(FT_ENCODING_NONE),
				enc(FT_ENCODING_UNICODE),
				enc(FT_ENCODING_MS_SYMBOL),
				enc(FT_ENCODING_ADOBE_LATIN_1),
				enc(FT_ENCODING_OLD_LATIN_2),
				enc(FT_ENCODING_SJIS),
				enc(FT_ENCODING_GB2312),
				enc(FT_ENCODING_BIG5),
				enc(FT_ENCODING_WANSUNG),
				enc(FT_ENCODING_JOHAB),
				enc(FT_ENCODING_ADOBE_STANDARD),
				enc(FT_ENCODING_ADOBE_EXPERT),
				enc(FT_ENCODING_ADOBE_CUSTOM),
				enc(FT_ENCODING_APPLE_ROMAN)
			}));

		#undef enc

		return encMap;
	}

} // anonymous namespace

PixMapSym::PixMapSym(unsigned long symCode_,				// the symbol code
						const FT_Bitmap &bm,			// the bitmap to process
						int leftBound, int topBound,	// initial position of the symbol
						const FT_BBox &bb) :			// the bounding box to fit
			symCode(symCode_) {
	int sz, rows_, cols_, left_, top_, diffLeft, diffRight, diffTop, diffBottom;
	fitGlyphToBox(bm, bb, leftBound, topBound, // input params
				  sz, rows_, cols_, left_, top_, diffLeft, diffRight, diffTop, diffBottom); // output params

	if(rows_ > 0 && cols_ > 0) {
		int amount = rows_ * cols_;
		data = new unsigned char[amount];
		for(int r = 0U; r<rows_; ++r) // copy a row at a time
			memcpy_s(data+r*cols_, (rows_-r)*cols_,
						&bm.buffer[(r-diffTop)*bm.pitch - diffLeft],
						cols_);
		glyphSum = *sum(Mat(1, amount, CV_8UC1, data)).val / 255.;
	}
	negGlyphSum = (double)sz*sz - glyphSum;

	// Considering a bounding box sz x sz with coordinates 0,0 -> (sz-1),(sz-1)
	left_ -= bb.xMin;
	top_ = (sz-1) - (bb.yMax-top_);
	assert(top_>=0 && top_<sz);
	
	left = (unsigned char)left_;
	top = (unsigned char)top_;
	rows = (unsigned char)rows_;
	cols = (unsigned char)cols_;

	Mat consec(1, sz, CV_64FC1), revConsec;
	iota(consec.begin<double>(), consec.end<double>(), (double)0.);
	flip(consec, revConsec, 1);

	mc = computeMc((unsigned)sz, data, rows, cols, left, top, glyphSum, consec, revConsec);
}

PixMapSym::PixMapSym(PixMapSym &&other) : // required by some vector manipulations
		symCode(other.symCode),
		glyphSum(other.glyphSum), negGlyphSum(other.negGlyphSum),
		mc(other.mc),
		rows(other.rows), cols(other.cols), left(other.left), top(other.top),
		data(other.data) {
	other.data = nullptr;
}

PixMapSym& PixMapSym::operator=(PixMapSym &&other) {
	if(this != &other) {
		symCode = other.symCode;
		glyphSum = other.glyphSum; negGlyphSum = other.negGlyphSum;
		mc = other.mc;
		rows = other.rows; cols = other.cols;
		left = other.left; top = other.top;
		delete[] data; data = other.data; other.data = nullptr;
	}
	return *this;
}

PixMapSym::~PixMapSym() {
	delete[] data;
}

bool PixMapSym::operator==(const PixMapSym &other) const {
	if(this == &other || symCode == other.symCode)
		return true;

	return
		glyphSum == other.glyphSum &&
		left == other.left && top == other.top &&
		rows == other.rows && cols == other.cols &&
		mc == other.mc &&
		memcmp(data, other.data, rows*cols*sizeof(decltype(*data))) == 0;
}

// Smallest 10% of all glyphs are considered small
const double FontEngine::SMALL_GLYPHS_PERCENT = 0.1;

FontEngine::FontEngine(Controller &ctrler_) : ctrler(ctrler_), syms(), fontSz(0U) {
	FT_Error error = FT_Init_FreeType(&library);
	if(error != FT_Err_Ok) {
		cerr<<"Couldn't initialize FreeType! Error: "<<error<<endl;
		throw runtime_error("Couldn't initialize FreeType!");
	}
}

FontEngine::~FontEngine() {
	FT_Done_Face(face);
	FT_Done_FreeType(library);
}

bool FontEngine::checkFontFile(const path &fontPath, FT_Face &face_) const {
	if(!exists(fontPath)) {
		cerr<<"No such file: "<<fontPath<<endl;
		return false;
	}
	FT_Error error = FT_New_Face(library, fontPath.string().c_str(), 0, &face_);
	if(error != FT_Err_Ok) {
		cerr<<"Invalid font file: "<<fontPath<<"  Error: "<<error<<endl;
		return false;
	}
/*
	// Some faces not providing this flag 'squeeze' basic ASCII characters to the left of the square

	if(!FT_IS_FIXED_WIDTH(face_)) {
		cerr<<"The font file "<<fontPath<<" isn't a fixed-width (monospace) font! Flags: 0x"<<hex<<face_->face_flags<<dec<<endl;
		return false;
	}
*/

	if(!FT_IS_SCALABLE(face_)) {
		cerr<<"The font file "<<fontPath<<" isn't a scalable font!"<<endl;
		return false;
	}

	return true;
}

bool FontEngine::setNthUniqueEncoding(unsigned idx) {
	if(face == nullptr) {
		cerr<<"No Font yet! Please select one first and then call setNthUniqueEncoding!"<<endl;
		throw logic_error("setNthUniqueEncoding called before selecting a font.");
	}

	if(idx == encodingIndex)
		return true; // same encoding

	if(idx >= uniqueEncodings())
		return false;

	FT_Error error =
		FT_Set_Charmap(face, face->charmaps[next(uniqueEncs.right.begin(), idx)->first]);
	if(error != FT_Err_Ok) {
		cerr<<"Couldn't set new cmap! Error: "<<error<<endl;
		return false;
	}

	encodingIndex = idx;
	encoding = encodingsMap().left.find(face->charmap->encoding)->second;
	fontSz = 0;

	return true;
}

bool FontEngine::setEncoding(const string &encName) {
	if(face == nullptr) {
		cerr<<"No Font yet! Please select one first and then call setEncoding!"<<endl;
		throw logic_error("setEncoding called before selecting a font.");
	}

	if(encName.compare(encoding) == 0)
		return true; // same encoding

	auto &encMap = encodingsMap().right; // encodingName->FT_Encoding
	auto itEncName = encMap.find(encName);
	if(encMap.end() == itEncName) {
		cerr<<"Unknown encoding "<<encName<<endl;
		return false;
	}

	auto itEnc = uniqueEncs.left.find(itEncName->second); // FT_Encoding->uniqueIndices
	if(uniqueEncs.left.end() == itEnc) {
		cerr<<"Current font doesn't contain encoding "<<encName<<endl;
		return false;
	}

	encodingIndex = UINT_MAX;
	return setNthUniqueEncoding(itEnc->second);
}

void FontEngine::setFace(FT_Face face_, const string &fontFile_/* = ""*/) {
	if(face_ == nullptr) {
		cerr<<"Trying to set a NULL face!"<<endl;
		throw invalid_argument("Can't provide a NULL face as parameter!");
	}
	if(face != nullptr) {
		if(strcmp(face->family_name, face_->family_name)==0 &&
		   strcmp(face->style_name, face_->style_name)==0)
			return; // same face

		FT_Done_Face(face);
	}

	fontSz = 0U;
	syms.clear();
	uniqueEncs.clear();
	face = face_;

	for(int i = 0, charmapsCount = face->num_charmaps; i<charmapsCount; ++i)
		uniqueEncs.insert(
			bimap<FT_Encoding, unsigned>::value_type(face->charmaps[i]->encoding, i));

	encodingIndex = UINT_MAX;
	setNthUniqueEncoding(0U);

	if(!fontFile_.empty())
		fontFile = fontFile_;
}

bool FontEngine::newFont(const string &fontFile_) {
	FT_Face face_;
	path fontPath(absolute(fontFile_));
	if(!checkFontFile(fontPath, face_))
		return false;
	
	setFace(face_, fontPath.string());
	return true;
}

void FontEngine::setFontSz(unsigned fontSz_) {
	if(fontSz == fontSz_)
		return;

	if(face == nullptr) {
		cerr<<"Please use FontEngine::newFont before calling FontEngine::setFontSz!"<<endl;
		throw logic_error("FontEngine::setFontSz called before FontEngine::newFont!");
	}

	if(!Config::isFontSizeOk(fontSz_)) {
		cerr<<"Invalid font size ("<<fontSz_<<") for FontEngine::setFontSz!"<<endl;
		throw invalid_argument("Invalid font size for FontEngine::setFontSz!");
	}

	vector<tuple<FT_ULong, double, double>> toResize;
	double sz = fontSz = fontSz_, factorH, factorV;
	const double sz2 = (double)fontSz * fontSz;
	unsigned symsCount, blanks = 0U, duplicates = 0U;
	FT_BBox bb;
	FT_UInt idx;
	FT_Size_RequestRec  req;
	req.type = FT_SIZE_REQUEST_TYPE_REAL_DIM;
	req.horiResolution = req.vertResolution = 72U;

	syms.clear();
	double progress = 0.;
	ctrler.reportGlyphProgress(progress);
	// 2% for this stage, no reports
	tie(factorH, factorV) = adjustScaling(face, fontSz, bb, symsCount);
	syms.reserve(symsCount);

	cout<<"The current charmap contains "<<symsCount<<" symbols"<<endl;

	// 90% for this stage, report every 5%
	FT_ULong tick = (FT_ULong)round((symsCount*5.)/90);

	// Store the pixmaps of the symbols that fit the bounding box already or by shifting.
	// Preserve the symbols that don't feet to resize them first, then add them too to pixmaps.
	for(FT_ULong c = FT_Get_First_Char(face, &idx), i = (FT_ULong)round((symsCount*2.)/90);
				idx != 0;  c=FT_Get_Next_Char(face, c, &idx), ++i) {
		if(i % tick == 0)
			ctrler.reportGlyphProgress(progress+=.05);
		FT_Load_Char(face, c, FT_LOAD_RENDER);
		FT_GlyphSlot g = face->glyph;
		FT_Bitmap b = g->bitmap;
		unsigned height = b.rows, width = b.width;
		if(height==0U || width==0U) { // skip Space character
			++blanks;
			continue;
		}

		if(width > fontSz || height > fontSz)
			toResize.emplace_back(c, max(1., height/sz), max(1., width/sz));
		else
			appendSym(c, g, bb, syms, blanks, duplicates, sz2);
	}

	// 7% for this stage, report every 5% => report 95% at 3/7 from it
	// Resize symbols which didn't fit initially
	int i = 0, reportI = (int)(3*toResize.size()/7);
	for(auto &item : toResize) {
		if(i++ == reportI)
			ctrler.reportGlyphProgress(.95);

		FT_ULong c;
		double hRatio, vRatio;
		tie(c, vRatio, hRatio) = item;
		req.height = (FT_ULong)floor(factorV * (fontSz<<6) / vRatio);
		req.width = (FT_ULong)floor(factorH * (fontSz<<6) / hRatio);
		FT_Error error = FT_Request_Size(face, &req);
		FT_Load_Char(face, c, FT_LOAD_RENDER);
		appendSym(c, face->glyph, bb, syms, blanks, duplicates, sz2);
	}

	if(blanks != 0U)
		cout<<"Removed "<<blanks<<" Space characters from symsSet!"<<endl;
	if(duplicates != 0U)
		cout<<"Removed "<<duplicates<<" duplicates from symsSet!"<<endl;

	// 1% for this stage, no more reports
	// Determine below max box coverage for smallest glyphs from the kept symsSet.
	// This will be used to favor using larger glyphs when this option is selected.
	auto smallGlyphsQty = (long)round(syms.size() * SMALL_GLYPHS_PERCENT);
	nth_element(syms.begin(), next(syms.begin(), smallGlyphsQty), syms.end(),
				[] (const PixMapSym &first, const PixMapSym &second) -> bool {
					return first.glyphSum < second.glyphSum;
				});
	coverageOfSmallGlyphs = next(syms.begin(), smallGlyphsQty)->glyphSum / sz2;
	sort(BOUNDS(syms), // just to appear familiar while visualizing the cmap
		 [] (const PixMapSym &first, const PixMapSym &second) -> bool {
			return first.symCode < second.symCode;
		});

	ctrler.reportGlyphProgress(1.);

#ifdef _DEBUG
	cout<<toResize.size()<<" symbols were resized twice."<<endl<<endl;

	for(auto &item : toResize)
		cout<<get<0>(item)<<", ";
	cout<<endl<<endl;

	cout<<"Resulted Bounding box: "<<bb.yMin<<","<<bb.xMin<<" -> "<<bb.yMax<<","<<bb.xMax<<endl<<endl;

	cout<<"Symbolrs considered small cover at most "<<
		fixed<<setprecision(2)<<100.*coverageOfSmallGlyphs<<"% of the box"<<endl<<endl;

#endif // _DEBUG
}

const string& FontEngine::getEncoding(unsigned *pEncodingIndex/* = nullptr*/) const {
	if(face == nullptr) {
		cerr<<"Font Encoding not ready yet! Please do all the configurations first!"<<endl;
		throw logic_error("getEncoding called before the completion of configuration.");
	}
	if(pEncodingIndex != nullptr)
		*pEncodingIndex = encodingIndex;

	return encoding;
}

unsigned FontEngine::uniqueEncodings() const {
	if(face == nullptr) {
		cerr<<"No Font yet! Please select one first and then call uniqueEncodings!"<<endl;
		throw logic_error("fontFileName called before selecting a font.");
	}
	return (unsigned)uniqueEncs.size();
}

const vector<PixMapSym>& FontEngine::symsSet() const {
	if(face == nullptr || fontSz == 0U) {
		cerr<<"symsSet not ready yet! Please do all the configurations first!"<<endl;
		throw logic_error("symsSet called before selecting a font.");
	}
	return syms;
}

double FontEngine::smallGlyphsCoverage() const {
	if(face == nullptr || fontSz == 0U) {
		cerr<<"smallGlyphsCoverage not ready yet! Please do all the configurations first!"<<endl;
		throw logic_error("smallGlyphsCoverage called before selecting a font.");
	}
	return coverageOfSmallGlyphs;
}

const string& FontEngine::fontFileName() const {
	return fontFile; // don't throw if empty; simply denote that the user didn't select a font yet
}

FT_String* FontEngine::getFamily() const {
	if(face == nullptr) {
		cerr<<"No Font yet! Please select one first and then call getFamily!"<<endl;
		throw logic_error("getFamily called before selecting a font.");
	}
	return face->family_name;
}

FT_String* FontEngine::getStyle() const {
	if(face == nullptr) {
		cerr<<"No Font yet! Please select one first and then call getStyle!"<<endl;
		throw logic_error("getStyle called before selecting a font.");
	}
	return face->style_name;
}
