/**********************************************************
 Project:     Pic2Sym
 File:        fontEngine.cpp

 Author:      Florin Tulba
 Created on:  2015-12-20
 
 Copyright (c) 2016 Florin Tulba
 **********************************************************/

#include "fontEngine.h"
#include "misc.h"

#include <fstream>
#include <sstream>
#include <set>
#include <map>
#include <algorithm>
#include <cassert>

#include FT_TRUETYPE_IDS_H

using namespace std;
using namespace cv;
using namespace boost::filesystem;

namespace {
#ifdef _DEBUG
	// Reads the character to represent. Its code can be entered, as well.
	FT_ULong inputChar() {
		string ch;
		while(ch.empty()) {
			cout<<"Enter the char or its decimal code(single digit codes should be prefixed by 0) : ";
			getline(cin, ch);
		}

		FT_ULong c;
		if(ch.length() == 1)
			c = (FT_ULong)ch[0];
		else
			istringstream(ch)>>c;

		return c;
	}

	// Display character's data normally or within a bounding box of given size.
	void displayChar(unsigned char *data,
							int rows, int cols, int pitch,
							int left = 0, int top = 0,
							unsigned sz = 0U) {
		static const string SHADING_CHARS = " .:;+cuoLCSOG8$@";

		string topChunk, bottomChunk,	// padding above and below glyph rectangle
			prefix, suffix;			// padding to the left/right of the glyph rectangle
		if(sz != 0U) {
			assert(left>=0);
			int blankAfterCount = (int)sz - (left + cols), sz1 = (int)sz - 1;
			assert(blankAfterCount>=0);

			string hBorder(sz, '-'), blankLine(sz, ' '),
				blankBefore(left, ' '), blankAfter(blankAfterCount, ' ');

			ostringstream oss;
			oss<<(char)218<<hBorder<<(char)191<<endl;
			for(int r = top; r<sz1; ++r)
				oss<<'|'<<blankLine<<'|'<<endl;
			topChunk = oss.str(); // padding above glyph rectangle

			oss.str(""); oss.clear();
			oss<<'|'<<blankBefore;
			prefix = oss.str(); // padding to the left of the glyph rectangle

			oss.str(""); oss.clear();
			oss<<blankAfter<<'|';
			suffix = oss.str(); // padding to the right of the glyph rectangle

			oss.str(""); oss.clear();
			for(int r = 0; r <= top-rows; ++r)
				oss<<'|'<<blankLine<<'|'<<endl;
			oss<<(char)192<<hBorder<<(char)217<<endl;
			bottomChunk = oss.str(); // padding below glyph rectangle
		}

		cout<<topChunk; // above padding
		for(int r = 0, ii = 0; r < rows; ++r, ii = r*pitch) {
			cout<<prefix; // left padding
			for(int c = 0; c<cols; ++c)
				cout<<SHADING_CHARS[((unsigned)data[ii++])>>4];
			cout<<suffix<<endl; // right padding
		}
		cout<<bottomChunk<<endl; // bottom padding
	}
#endif // _DEBUG

	/*
	adjustScaling enforces sz as vertical size and determines an optimal horizontal size,
	so that most characters will widen enough to fill more of the drawing square,
	while preserving the designer's placement.

	The parameters:
	- current face
	- drawing square size

	The returned tuple will contain:
	- the estimated future bounding box
	- the scaling factors to remember
	- the count of glyphs within current charmap
	*/
	tuple<FT_BBox, double, double, unsigned>
		adjustScaling(FT_Face face, unsigned sz) {
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
		unsigned count = 0U;
		FT_UInt idx;
		FT_ULong c = FT_Get_First_Char(face, &idx);
		while(idx != 0) { // Assess all glyphs of this charmap
			++count;
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
		meanStdDev(Mat_<double>(1, count, vHeight.data()), avgHeight, sdHeight);
		meanStdDev(Mat_<double>(1, count, vWidth.data()), avgWidth, sdWidth);
		meanStdDev(Mat_<double>(1, count, vTop.data()), avgTop, sdTop);
		meanStdDev(Mat_<double>(1, count, vBottom.data()), avgBottom, sdBottom);
		meanStdDev(Mat_<double>(1, count, vLeft.data()), avgLeft, sdLeft);
		meanStdDev(Mat_<double>(1, count, vRight.data()), avgRight, sdRight);

		double kv = 1., kh = 1.; // 1. means a single standard deviation => ~68% of the data

		// Enlarge factors, forcing the average width + lateral std. devs
		// to fit the width of the drawing square.
		double factorH = sz / ( *avgWidth.val + kh*( *sdLeft.val + *sdRight.val)),
			factorV = sz / ( *avgHeight.val + kv*( *sdTop.val + *sdBottom.val)); // Similar story vertically

		// Computing new height & width
		req.height = (FT_ULong)floor(factorV * req.height);
		req.width = (FT_ULong)floor(factorH * req.width);

		FT_Request_Size(face, &req); // reshaping the fonts to better fill the drawing square

		// Positioning the Bounding box to best cover the estimated future position & size of the chars
		double yMin = factorV * ( *avgBottom.val - *sdBottom.val), // current bottom scaled by factorV
			yMax = factorV * ( *avgTop.val + *sdTop.val),			// top
			yDiff2 = (yMax-yMin+1-sz)/2.,	// the difference to divide equally between top & bottom
			xMin = factorH * ( *avgLeft.val - *sdLeft.val),		// current left scaled by factorH
			xMax = factorH * ( *avgRight.val + *sdRight.val),		// right
			xDiff2 = (xMax-xMin+1-sz)/2.;	// the difference to divide equally between left & right

		// distributing the differences
		yMin += yDiff2; yMax -= yDiff2;
		xMin += xDiff2; xMax -= xDiff2;

		// ensure yMin <= 0
		if(yMin > 0) {
			yMax -= yMin;
			yMin = 0;
		}

		return make_tuple(
			FT_BBox {				// estimated future bounding box
			(FT_Pos)round(xMin), (FT_Pos)round(yMin),
			(FT_Pos)round(xMax), (FT_Pos)round(yMax) },
			factorH, factorV,		// scaling factors to remember
			count);					// count of glyphs within current charmap
	}

	// Computing center of gravity (cog)
	const Point2d computeCog(unsigned sz, unsigned char *data,
							 unsigned char rows, unsigned char cols,
							 unsigned char left, unsigned char top,
							 unsigned long cachedSum) {
		// x : sum columns, weight each such sum by the column index => total sum
		// Total sum has to be then divided by cachedSum => x cog
		vector<unsigned long> sums((size_t)sz, 0UL);
		for(unsigned r = 0U, idx = 0; r<sz; ++r)
			for(unsigned c = 0U; c<sz; ++c)
				sums[c] += data[idx++];

		double totalX = 0.;
		for(unsigned char c = left; c < left+cols; ++c)
			totalX += (double)c * sums[c-left];

		// y : sum rows, weight each such sum by the row index => total sum
		// Total sum has to be then divided by cachedSum => y cog
		sums.assign((size_t)sz, 0UL);
		for(unsigned r = 0U, idx = 0; r<sz; ++r)
			for(unsigned c = 0U; c<sz; ++c)
				sums[r] += data[idx++];

		double totalY = 0.;
		for(unsigned char bottom = top+1-rows, r = bottom; r <= top; ++r)
			totalY += (double)r * sums[r-bottom];

		return Point2d(totalX, totalY) / (double)cachedSum;
	}

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
} // anonymous namespace

PixMapChar::PixMapChar(unsigned long charCode,		// the character code
								   const FT_Bitmap &bm,			// the bitmap to process
								   int leftBound, int topBound,	// initial position of the character
								   const FT_BBox &bb) :			// the bounding box to fit

			chCode(charCode) {

	int sz, rows_, cols_, left_, top_, diffLeft, diffRight, diffTop, diffBottom;
	fitGlyphToBox(bm, bb, leftBound, topBound, // input params
				  sz, rows_, cols_, left_, top_, diffLeft, diffRight, diffTop, diffBottom); // output params

	if(rows_ > 0 && cols_ > 0) {
		data = new unsigned char[rows_ * cols_];
		for(int r = 0U; r<rows_; ++r) // copy a row at a time
			memcpy_s(data+r*cols_, (rows_-r)*cols_,
						&bm.buffer[(r-diffTop)*bm.pitch - diffLeft],
						cols_);
		for(int i = 0, lim = rows_*cols_; i<lim; ++i)
			cachedSum += (unsigned long)data[i];
	}

	// Considering a bounding box sz x sz with coordinates 0,0 -> (sz-1),(sz-1)
	left_ -= bb.xMin;
	top_ = (sz-1) - (bb.yMax-top_);
	assert(top_>=0 && top_<sz);
	
	left = (unsigned char)left_;
	top = (unsigned char)top_;
	rows = (unsigned char)rows_;
	cols = (unsigned char)cols_;

	cachedCog = computeCog((unsigned)sz, data, rows, cols, left, top, cachedSum);
}

PixMapChar::PixMapChar(PixMapChar &&other) : // required by some vector manipulations
		chCode(other.chCode), cachedSum(other.cachedSum), cachedCog(other.cachedCog),
		rows(other.rows), cols(other.cols), left(other.left), top(other.top),
		data(other.data) {
	other.data = nullptr;
}

PixMapChar::~PixMapChar() { delete[] data; }

FontEngine::FontEngine() : chars(), fontSz(0U), dirty(true) {
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

bool FontEngine::checkFontFile(const string &fName, FT_Face &face_) const {
	FT_Error error = FT_New_Face(library, fName.c_str(), 0, &face_);
	if(error != FT_Err_Ok) {
		cerr<<"Invalid font file: "<<fName<<"  Error: "<<error<<endl;
		return false;
	}
/*
	// Some faces not providing this flag squeeze basic ASCII characters to the left of the square

	if(!FT_IS_FIXED_WIDTH(face_)) {
		cerr<<"The font file "<<fName<<" isn't a fixed-width (monospace) font! Flags: 0x"<<hex<<face_->face_flags<<dec<<endl;
		return false;
	}
*/

	if(!FT_IS_SCALABLE(face_)) {
		cerr<<"The font file "<<fName<<" isn't a scalable font!"<<endl;
		return false;
	}

	return true;
}

void FontEngine::ready() {
	if(dirty && face != nullptr && fontSz != 0U && !chars.empty()) {
		ostringstream oss;
		oss<<face->family_name<<'_'<<face->style_name<<'_'<<fontSz;
		if(encoding.length() > 0)
			oss<<'_'<<encoding;
		id = oss.str();
		dirty = false;
	}
}

// used within modifier methods below to ensure dirty-ready mechanism
#define CHARSET_ALTERATION(block) \
	dirty = true; \
	block \
	ready()

void FontEngine::selectEncoding() {
	if(face == nullptr) {
		cerr<<"Please use FontEngine::setFace before calling FontEngine::selectEncoding!"<<endl;
		throw logic_error("FontEngine::selectEncoding called before FontEngine::setFace!");
	}
	int charmapsCount = face->num_charmaps;
	if(charmapsCount > 1) {
#define enc(encValue) {encValue, string(#encValue).substr(12)}
		static map<FT_Encoding, string> encMap { // known encodings
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
		};
#undef enc
		auto selectedCmap = face->charmap;
		set<FT_Encoding> encodings;
		set<int> newCmaps;
		for(int i = 0; i<charmapsCount; ++i) {
			auto cmap = face->charmaps[i];
			if(selectedCmap == cmap ||
			   selectedCmap->encoding == cmap->encoding ||
			   encodings.end() != encodings.find(cmap->encoding))
				continue;
			newCmaps.insert(i);
			encodings.insert(cmap->encoding);
		}
		if(newCmaps.empty())
			return;

		cout<<"The encoding in use is "<<encMap[selectedCmap->encoding]<<'.'<<endl;
		cout<<"There are "<<newCmaps.size()<<" other charmaps available:"<<endl;
		for(int i : newCmaps)
			cout<<setw(2)<<i<<") "<<encMap[face->charmaps[i]->encoding]<<endl;

		cout<<endl;
		if(!boolPrompt("Keeping current charmap?")) {
			string ans;
			int idxCmap = -1;
			if(newCmaps.size() > 1) {
				cout<<endl;
				for(auto itEndNewCmaps = newCmaps.end(); newCmaps.find(idxCmap)==itEndNewCmaps;) {
					cout<<"Enter the index of the new cmap: ";
					getline(cin, ans);
					istringstream(ans)>>idxCmap;
				}
			} else idxCmap = *newCmaps.begin();
			FT_Error error = FT_Set_Charmap(face, face->charmaps[idxCmap]);
			if(error != FT_Err_Ok) {
				cerr<<"Couldn't set new cmap! Error: "<<error<<endl;
				throw runtime_error("Couldn't set new cmap!");
			}

			CHARSET_ALTERATION(
				encoding = encMap[face->charmap->encoding];
				fontSz = 0U;
				chars.clear();
			);
			cout<<"Set encoding "<<encoding<<endl;
		}
	}
}

void FontEngine::setFace(FT_Face &face_) {
	if(face != nullptr) {
		if(strcmp(face->family_name, face_->family_name)==0 && strcmp(face->style_name, face_->style_name)==0)
			return; // same face

		FT_Done_Face(face);
	}
	CHARSET_ALTERATION(
		face = face_;
		encoding = "";
		fontSz = 0U;
		chars.clear();
	);
}

void FontEngine::setFontSz(unsigned fontSz_) {
	if(!dirty && fontSz == fontSz_)
		return;

	if(face == nullptr) {
		cerr<<"Please use FontEngine::setFace before calling FontEngine::setFontSz!"<<endl;
		throw logic_error("FontEngine::setFontSz called before FontEngine::setFace!");
	}

	CHARSET_ALTERATION(SINGLE_ARG(
		double sz = fontSz = fontSz_;
		const unsigned long sz2_255 = (unsigned long)fontSz * fontSz * 255UL;
		map<FT_ULong, size_t> code2Pixmap;
		double factorH, factorV;
		unsigned charsCount;
		FT_BBox bb;

		chars.clear();
		tie(bb, factorH, factorV, charsCount) = adjustScaling(face, fontSz);
		chars.reserve(charsCount);

		cout<<"The current charmap contains "<<charsCount<<" characters"<<endl;

		// Store the pixmaps of the characters that fit the bounding box already or by shifting.
		// Preserve the characters that don't feet to resize them first, then add them too to pixmaps.
		vector<tuple<FT_ULong, double, double>> toResize;
		FT_UInt idx;
		unsigned blanks = 0U;
		for(FT_ULong c=FT_Get_First_Char(face, &idx);  idx != 0;  c=FT_Get_Next_Char(face, c, &idx)) {
			FT_Load_Char(face, c, FT_LOAD_RENDER);
			FT_GlyphSlot g = face->glyph;
			FT_Bitmap b = g->bitmap;
			unsigned height = b.rows, width = b.width;
			if(height==0U || width==0U) { // skip Space character
				++blanks;
				continue;
			}
			FT_Pos leftBound = g->bitmap_left, rightBound = leftBound+(int)width-1,
				topBound = g->bitmap_top, bottomBound = topBound-(int)height+1;

			if(width > fontSz || height > fontSz) {
				toResize.emplace_back(c, max(1., height/sz), max(1., width/sz));
			} else {
				chars.emplace_back(c, b, leftBound, topBound, bb);
				if(chars.back().cachedSum % sz2_255 != 0) // discard disguised Space chars
					code2Pixmap[c] = chars.size()-1U;
				else {
					chars.pop_back();
					++blanks;
				}
			}
		}

		// Resize characters which didn't fit initially
		FT_Size_RequestRec  req;
		req.type = FT_SIZE_REQUEST_TYPE_REAL_DIM; // FT_SIZE_REQUEST_TYPE_BBOX;
		req.horiResolution = req.vertResolution = 72U;
		for(auto &item : toResize) {
			FT_ULong c;
			double hRatio, vRatio;
			tie(c, vRatio, hRatio) = item;
			req.height = (FT_ULong)floor(factorV * (fontSz<<6) / vRatio);
			req.width = (FT_ULong)floor(factorH * (fontSz<<6) / hRatio);
			FT_Error error = FT_Request_Size(face, &req);
			FT_Load_Char(face, c, FT_LOAD_RENDER);
			FT_GlyphSlot g = face->glyph;
			FT_Bitmap b = g->bitmap;
			chars.emplace_back(c, b, g->bitmap_left, g->bitmap_top, bb);
			if(chars.back().cachedSum % sz2_255 != 0) // discard disguised Space chars
				code2Pixmap[c] = chars.size()-1U;
			else {
				chars.pop_back();
				++blanks;
			}
		}

		if(blanks != 0U)
			cout<<"Removed "<<blanks<<" Space characters from charset!"<<endl;
	)); // CHARSET_ALTERATION(SINGLE_ARG(

#ifdef _DEBUG
	cout<<toResize.size()<<" characters were resized twice."<<endl<<endl;

	for(auto &item : toResize)
		cout<<get<0>(item)<<", ";
	cout<<endl<<endl;

	cout<<"Resulted Bounding box: "<<bb.yMin<<","<<bb.xMin<<" -> "<<bb.yMax<<","<<bb.xMax<<endl<<endl;

	req.height = (FT_ULong)round(factorV * (fontSz<<6));
	req.width = (FT_ULong)round(factorH * (fontSz<<6));
	FT_Error error = FT_Request_Size(face, &req);

	cout<<"You may now inspect the characters. Leave by entering the Space character followed by Enter."<<endl;
	for(FT_ULong c = 0U; c!=32U;) {
		c = inputChar();

		FT_Load_Char(face, c, FT_LOAD_RENDER);

		FT_GlyphSlot g = face->glyph;
		FT_Bitmap b = g->bitmap;

		cout<<"Bitmap for "<<c<<" is "<<b.rows<<"x"<<b.width<<":"<<endl<<endl;
		displayChar(b.buffer, b.rows, b.width, b.pitch);
		cout<<endl<<"Bottom row="<<g->bitmap_top-(int)b.rows+1<<" ; Left column="<<g->bitmap_left<<endl<<endl;

		if(code2Pixmap.find(c) != code2Pixmap.end()) {
			auto &p = chars[code2Pixmap[c]];
			displayChar(p.data, p.rows, p.cols, p.cols, p.left, p.top, fontSz);
		}
	}
#endif
}

#undef CHARSET_ALTERATION

void FontEngine::generateCharmapCharts(const path& destdir) const {
	if(dirty || chars.empty()) {
		cerr<<"No charts can be generated! Please do all the configurations first!"<<endl;
		return;
	}

	static const unsigned ROWS = 600U, COLS = 800U, PITCH = (COLS+3)&-4, dataSz = ROWS*PITCH,
		GRAYS_COUNT = 256U;

	unsigned cellSz = fontSz+1U, idx = 0U;
	FT_Bitmap bm { ROWS, COLS, PITCH, new unsigned char[dataSz],
		GRAYS_COUNT, FT_PIXEL_MODE_GRAY,
		(unsigned char)0, nullptr }; // fields not used
	

	FT_BBox bb { 0, 0, fontSz-1, fontSz-1 };
	auto pixIt = chars.begin(), pixEnd = chars.end();
	while(pixIt!=pixEnd) {
		memset(bm.buffer, 255, dataSz);

		// Create grid
		for(unsigned r = 0U; r<ROWS; ++r)
			for(unsigned c = fontSz; c<COLS; c += cellSz)
				bm.buffer[r*PITCH+c] = 200U;

		for(unsigned r = fontSz; r<ROWS; r += cellSz)
			for(unsigned c = 0U; c<COLS; ++c)
				bm.buffer[r*PITCH+c] = 200U;

		// Fill the chart
		for(unsigned r = 0U; pixIt!=pixEnd && r<ROWS-cellSz; r += cellSz) {
			for(unsigned c = 0U; pixIt!=pixEnd && c<COLS-cellSz; c += cellSz, ++pixIt) {
				auto &p = *pixIt;
				int blankBeforeCount = p.left - bb.xMin + c;

				for(int rr = bb.yMax-p.top, ii = 0; rr < bb.yMax-p.top+p.rows; ++rr) {
					auto pos = (rr+r)*PITCH+blankBeforeCount;
					for(int cc = 0; cc<p.cols; ++cc)
						bm.buffer[pos++] = 255-p.data[ii++];
				}
			}
		}

		// write this chart page to file
		ostringstream oss;
		oss<<face->family_name<<'_'<<face->style_name;
		if(encoding.length() > 0)
			oss<<'_'<<encoding;
		oss<<'_'<<idx++<<".pgm";
		path temp(destdir); temp /= oss.str();
		ofstream out_gray(temp.c_str(), ios::binary);
		out_gray << "P5 " << COLS << " " << ROWS << " 255\n";
		out_gray.write((const char *)bm.buffer, dataSz);
	}

	delete[] bm.buffer;
}

const vector<PixMapChar>& FontEngine::charset() const {
	if(dirty) {
		cerr<<"Charset not ready yet! Please do all the configurations first!"<<endl;
		throw logic_error("charset called before the completion of configuration.");
	}
	return chars;
}

const string& FontEngine::fontId() const {
	if(dirty && !id.empty()) {
		cerr<<"Id not ready yet! Please do all the configurations first!"<<endl;
		throw logic_error("fontId called before the completion of configuration.");
	}
	return id;
}