/**********************************************************
 Project:     Pic2Sym
 File:        fontEngine.h

 Author:      Florin Tulba
 Created on:  2015-12-20
 
 Copyright (c) 2016 Florin Tulba
 **********************************************************/

#ifndef H_FONT_ENGINE
#define H_FONT_ENGINE

#include <string>
#include <vector>

#include <boost/filesystem.hpp>
#include <opencv2/core.hpp>
#include <ft2build.h>
#include FT_FREETYPE_H

/*
PixMapChar holds the representation('data') of character 'chCode'.
Expects characters that mostly fit the Bounding Box provided to the constructor.
That is the characters are either already within the BBox,
or their size needs to be at most a few pixels more than the BBox (they'll get trimmed).

The height('rows'), width('cols') are also recorded.
The bitmap pitch is transformed into width, so 'data' is continuous.
Fields 'left' and 'top' indicate the position of the top-left corner within the drawing square.
*/
struct PixMapChar {
	unsigned long chCode = 0UL;	// character code
	double glyphSum = 0.;		// sum of the pixel values divided by 255
	double negGlyphSum = 0.;	// sum of the pixel values divided by 255 for negated glyph
	cv::Point2d cogFg;			// center of gravity of the glyph
	cv::Point2d cogBg;			// center of gravity of the background of the glyph

	unsigned char rows = 0U, cols = 0U;	// dimensions of 'data' rectangle from below 
	unsigned char left = 0U, top = 0U;	// position within the drawing square
	unsigned char *data = nullptr;		// 256-shades of gray rectangle describing the character
	// (top-down, left-right traversal)

	/*
	Processes a FT_Bitmap object to store a faithful representation of the charCode
	drawn within a bounding box (BBox).

	The characters must be:
	- either already within the BBox,
	- or their size needs to be at most a few pixels more than the BBox (they'll get trimmed).

	The other characters need resizing before feeding the PixMapChar.

	After shifting the character inside, the regions still lying outside are trimmed.

	Any bitmap pitch is removed => storing data without gaps.
	*/
	PixMapChar(unsigned long charCode,		// the character code
			   const FT_Bitmap &bm,			// the bitmap to process
			   int leftBound, int topBound,	// initial position of the character
			   const FT_BBox &bb);			// the bounding box to fit
	PixMapChar(const PixMapChar&) = delete;
	PixMapChar(PixMapChar &&other);			// required by some vector manipulations
	~PixMapChar();

	void operator=(const PixMapChar&) = delete;

	PixMapChar& operator=(PixMapChar &&other); // needed when nth_element applies to vector<PixMapChar>
};

/*
FontEngine class wraps some necessary FreeType functionality.
*/
class FontEngine final {
	static const double SMALL_GLYPHS_PERCENT; // percentage of total glyphs considered small

	FT_Library library = nullptr;	// the FreeType lib
	FT_Face face		= nullptr;	// a loaded font

	std::string encoding;			// selected charmap (cmap)
	std::string id;					// the id of the current font
	std::vector<PixMapChar> chars;	// data for each character within current charmap
	
	double coverageOfSmallGlyphs;	// max ratio of glyph area / containing area for small chars
	unsigned fontSz = 0U;			// bounding box size
	bool dirty = true;				// flag raised for new face/encoding/size; lowered by ready

	void ready();					// toggles dirty when the glyphs are ready to use

public:
	FontEngine();
	~FontEngine();

	/*
	checkFontFile Validates a new font file.
	When fName is valid, face_ parameter will return the successfully loaded font.
	*/
	bool checkFontFile(const std::string &fName, FT_Face &face_) const;
	void setFace(FT_Face &face_);		// Installs a new font
	void selectEncoding();				// Setting a different charmap if available
	void setFontSz(unsigned fontSz_);	// Sets the desired font height in pixels

	// Generate several charts with the glyphs from current charmap
	void generateCharmapCharts(const boost::filesystem::path& destdir) const;
	
	const std::vector<PixMapChar>& charset() const;	// get the charset
	double smallGlyphsCoverage() const;				// get coverageOfSmallGlyphs
	const std::string& fontId() const;				// current font id
	const std::string& getEncoding() const;			// get encoding
};

#endif
