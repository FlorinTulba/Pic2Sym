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

#ifndef H_PIXMAP_SYM
#define H_PIXMAP_SYM

#include "symFilterBase.h"

#include <vector>
#include <map>
#include <memory>

#include <opencv2/core.hpp>
#include <ft2build.h>
#include FT_FREETYPE_H

/**
PixMapSym holds the representation('pixels') of symbol 'symCode'.

Expects symbols that mostly fit the Bounding Box provided to the constructor.
That is the symbols are either already within the BBox,
or their size needs to be at most a few pixels more than the BBox (they'll get trimmed).

The height('rows'), width('cols') are also recorded.
The bitmap pitch is transformed into width, so 'pixels' is continuous.
Fields 'left' and 'top' indicate the position of the top-left corner within the drawing square.
*/
struct PixMapSym {
	unsigned long symCode = 0UL;	///< symbol code
	size_t symIdx = 0U;				///< symbol index within cmap
	double avgPixVal = 0.;			///< average of the pixel values divided by 255
	
	/// glyph's mass center (coordinates are within a unit-square: 0..1 x 0..1)
	cv::Point2d mc;

	unsigned char rows = 0U, cols = 0U;	// dimensions of 'pixels' rectangle from below 
	unsigned char left = 0U, top = 0U;	// position within the drawing square

	/// 256-shades of gray rectangle describing the character (top-down, left-right traversal)
	std::vector<unsigned char> pixels;

	cv::Mat colSums; ///< row with the sums of the pixels of each column of the symbol (each pixel in 0..1)
	cv::Mat rowSums; ///< row with the sums of the pixels of each row of the symbol (each pixel in 0..1)

	bool removable = false;	///< when set to true, the symbol will appear as marked (inversed) in the cmap viewer

	/**
	Processes a FT_Bitmap object to store a faithful representation of the symCode
	drawn within a bounding box (BBox).

	The symbols must be:
	- either already within the BBox,
	- or their size needs to be at most a few pixels more than the BBox (they'll get trimmed).

	The other symbols need resizing before feeding the PixMapSym.

	After shifting the symbol inside, the regions still lying outside are trimmed.

	Any bitmap pitch is removed => storing pixels without gaps.
	*/
	PixMapSym(unsigned long symCode,	///< the symbol code
			  size_t symIdx_,			///< symbol index within cmap
			  const FT_Bitmap &bm,		///< the bitmap to process
			  int leftBound,			///< initial position of the symbol considered from the left
			  int topBound,				///< initial position of the symbol considered from the top
			  int sz,					///< font size
			  double maxGlyphSum,		///< max sum of a glyph's pixels
			  const cv::Mat &consec,	///< vector of consecutive values 0 .. sz-1
			  const cv::Mat &revConsec,	///< vector of consecutive values sz-1 .. 0
			  const FT_BBox &bb			///< the bounding box to fit
			  );
	PixMapSym(const PixMapSym&);
	PixMapSym(PixMapSym&&);
	PixMapSym& operator=(PixMapSym &&other); ///< needed when nth_element applies to vector<PixMapSym>

	void operator=(const PixMapSym&) = delete;

	bool operator==(const PixMapSym &other) const; ///< useful to detect duplicates

	cv::Mat asNarrowMat() const;			///< a matrix with the symbol within its tight bounding box

	/// the symbol within a square of fontSz x fontSz, either as is, or inversed
	cv::Mat toMat(unsigned fontSz, bool inverse = false) const;

	/// Conversion PixMapSym .. Mat of type double with range [0..1] instead of [0..255]
	cv::Mat toMatD01(unsigned fontSz) const;

	/**
	Computing the mass center (mc) and average pixel value (divided by 255) of a given symbol.
	When the parameters colSums, rowSums are not nullptr, the corresponding sum is returned.

	It's static for easier Unit Testing.
	*/
	static void computeMcAndAvgPixVal(unsigned sz, double maxGlyphSum, const std::vector<unsigned char> &data,
									  unsigned char rows, unsigned char cols, unsigned char left, unsigned char top,
									  const cv::Mat &consec, const cv::Mat &revConsec,
									  cv::Point2d &mc, double &avgPixVal,
									  cv::Mat *colSums = nullptr, cv::Mat *rowSums = nullptr);

#ifdef UNIT_TESTING
	/// Constructs a PixMapSym in Unit Testing mode
	PixMapSym(const std::vector<unsigned char> &data, ///< values of the symbol's pixels
			  const cv::Mat &consec,	///< vector of consecutive values 0 .. sz-1
			  const cv::Mat &revConsec	///< vector of consecutive values sz-1 .. 0
			  );
#endif
};

struct IPresentCmap; // forward declaration

/// Convenience container to hold PixMapSym-s of same size
class PmsCont {
protected:
	bool ready = false;				///< is container ready to provide useful data?
	unsigned fontSz = 0U;			///< bounding box size
	std::vector<const PixMapSym> syms;	///< data for each symbol within current charmap
	unsigned blanks = 0U;			///< how many Blank characters were within the charmap
	unsigned duplicates = 0U;		///< how many duplicate symbols were within the charmap
	
	double maxGlyphSum;				///< max sum of a glyph's pixels
	double coverageOfSmallGlyphs;	///< max ratio for small symbols of glyph area / containing area

	// Precomputed entities during reset
	cv::Mat consec;					///< vector of consecutive values 0..fontSz-1
	cv::Mat revConsec;				///< consec reversed

	const IPresentCmap &cmapViewUpdater;	///< updates Cmap View as soon as there are enough symbols for 1 page

	/// member that allows setting filters to detect symbols with undesired features
	std::unique_ptr<ISymFilter> symFilter = std::make_unique<DefSymFilter>();
	std::map<unsigned, unsigned> removableSymsByCateg; ///< associations: filterId - count of detected syms

public:
	PmsCont(const IPresentCmap &cmapViewUpdater_);

	bool isReady() const { return ready; }
	unsigned getFontSz() const;
	unsigned getBlanksCount() const;
	unsigned getDuplicatesCount() const;
	const std::map<unsigned, unsigned>& getRemovableSymsByCateg() const;
	double getCoverageOfSmallGlyphs() const;
	const std::vector<const PixMapSym>& getSyms() const;

	/// clears & prepares container for new entries
	void reset(unsigned fontSz_ = 0U, unsigned symsCount = 0U);

	/**
	appendSym puts valid glyphs into vector 'syms'.

	Space (empty / full) glyphs are invalid.
	Also updates the count of blanks & duplicates and of any filtered out symbols.
	*/
	void appendSym(FT_ULong c, size_t symIdx, FT_GlyphSlot g, FT_BBox &bb, SymFilterCache &sfc);

	void setAsReady(); ///< No other symbols to append. Statistics can be now computed
};

#endif