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

#include <boost/filesystem/path.hpp>
#include <boost/bimap/bimap.hpp>
#include <opencv2/core.hpp>
#include <ft2build.h>
#include FT_FREETYPE_H

class Controller; // data & views manager

/*
PixMapSym holds the representation('pixels') of symbol 'symCode'.
Expects symbols that mostly fit the Bounding Box provided to the constructor.
That is the symbols are either already within the BBox,
or their size needs to be at most a few pixels more than the BBox (they'll get trimmed).

The height('rows'), width('cols') are also recorded.
The bitmap pitch is transformed into width, so 'pixels' is continuous.
Fields 'left' and 'top' indicate the position of the top-left corner within the drawing square.
*/
struct PixMapSym final {
	unsigned long symCode = 0UL;	// symbol code
	double glyphSum = 0.;			// sum of the pixel values divided by 255
	cv::Point2d mc;					// glyph's mass center

	unsigned char rows = 0U, cols = 0U;	// dimensions of 'pixels' rectangle from below 
	unsigned char left = 0U, top = 0U;	// position within the drawing square
	std::vector<unsigned char> pixels;	// 256-shades of gray rectangle describing the character
	// (top-down, left-right traversal)

	/*
	Processes a FT_Bitmap object to store a faithful representation of the symCode
	drawn within a bounding box (BBox).

	The symbols must be:
	- either already within the BBox,
	- or their size needs to be at most a few pixels more than the BBox (they'll get trimmed).

	The other symbols need resizing before feeding the PixMapSym.

	After shifting the symbol inside, the regions still lying outside are trimmed.

	Any bitmap pitch is removed => storing pixels without gaps.
	*/
	PixMapSym(unsigned long symCode,		// the symbol code
			   const FT_Bitmap &bm,			// the bitmap to process
			   int leftBound, int topBound,	// initial position of the symbol
			   int sz, double sz2,			// font size and squared of it
			   const cv::Mat &consec,		// vector of consecutive values 0 .. sz-1
			   const cv::Mat &revConsec,	// vector of consecutive values sz-1 .. 0
			   const FT_BBox &bb);			// the bounding box to fit
	PixMapSym(const PixMapSym&);
	PixMapSym(PixMapSym&&);
	PixMapSym& operator=(PixMapSym &&other); // needed when nth_element applies to vector<PixMapSym>

	void operator=(const PixMapSym&) = delete;

	bool operator==(const PixMapSym &other) const; // useful to detect duplicates

	/*
	Computes the sum of the pixel values divided by 255.
	Static to Unit Test it easier.
	*/
	static double computeGlyphSum(unsigned char rows_, unsigned char cols_,
								  const std::vector<unsigned char> &pixels_);

	/*
	Computing the mass center (mc) of a given glyph and its background.
	Static to Unit Test it easier.
	*/
	static const cv::Point2d computeMc(unsigned sz, const std::vector<unsigned char> &data,
								unsigned char rows, unsigned char cols,
								unsigned char left, unsigned char top,
								double glyphSum, const cv::Mat &consec, const cv::Mat &revConsec);
};

// Convenience container to hold PixMapSym-s of same size
class PmsCont final {
	static const double SMALL_GLYPHS_PERCENT; // percentage of total glyphs considered small

	bool ready = false;				// is container ready to provide useful data?
	unsigned fontSz = 0U;			// bounding box size
	std::vector<const PixMapSym> syms;	// data for each symbol within current charmap
	unsigned blanks = 0U;			// how many Blank characters were within the charmap
	unsigned duplicates = 0U;		// how many duplicate symbols were within the charmap
	double coverageOfSmallGlyphs;	// max ratio for small symbols of glyph area / containing area

	// Precomputed entities during reset
	cv::Mat consec, revConsec;		// vectors of consecutive values 0..fontSz-1 and reversed
	double sz2;						// fontSz^2

public:
	bool isReady() const { return ready; }
	unsigned getFontSz() const;
	unsigned getBlanksCount() const;
	unsigned getDuplicatesCount() const;
	double getCoverageOfSmallGlyphs() const;
	const std::vector<const PixMapSym>& getSyms() const;

	// clears & prepares container for new entries
	void reset(unsigned fontSz_ = 0U, unsigned symsCount = 0U);

	/*
	appendSym puts valid glyphs into vector <syms>. Space (empty / full) glyphs are invalid.
	Updates the count of blanks & duplicates.
	*/
	void appendSym(FT_ULong c, FT_GlyphSlot g, FT_BBox &bb);

	void setAsReady(); // No other symbols to append. Statistics can be now computed
};

// FontEngine class wraps some necessary FreeType functionality.
class FontEngine final {
	const Controller &ctrler;		// data & views manager

	FT_Library library	= nullptr;	// the FreeType lib
	FT_Face face		= nullptr;	// a loaded font

	std::string fontFile;			// path to the current font file
	std::string encoding;			// name of selected charmap (cmap)
	unsigned encodingIndex = 0U;	// the index of the selected cmap within face's charmaps array
	boost::bimaps::bimap<FT_Encoding, unsigned> uniqueEncs;	// indices for each unique Encoding within cmaps array

	PmsCont symsCont;				// Container with the PixMapSym-s of current charmap

	/*
	checkFontFile Validates a new font file.
	When fName is valid, face_ parameter will return the successfully loaded font.
	*/
	bool checkFontFile(const boost::filesystem::path &fontPath, FT_Face &face_) const;
	void setFace(FT_Face face_, const std::string &fontFile_/* = ""*/);	// Installs a new font

public:
	FontEngine(const Controller &ctrler_);
	~FontEngine();

	bool newFont(const std::string &fontFile_);		// Tries to use the font from <fontFile_>
	void setFontSz(unsigned fontSz_);				// Sets the desired font height in pixels
	bool setEncoding(const std::string &encName);	// Sets an encoding by name
	bool setNthUniqueEncoding(unsigned idx);		// Switches to nth unique encoding

	const std::vector<const PixMapSym>& symsSet() const;	// get the symsSet
	double smallGlyphsCoverage() const;				// get coverageOfSmallGlyphs

	const std::string& fontFileName() const;		// font name provided by Font Dialog
	unsigned uniqueEncodings() const;				// Returns the count of unique encodings
	const std::string& getEncoding(unsigned *pEncodingIndex = nullptr) const; // get encoding
	FT_String* getFamily() const;					// get font family
	FT_String* getStyle() const;					// get font style
};

#endif
