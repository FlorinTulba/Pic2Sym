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

#ifndef H_PIXMAP_SYM
#define H_PIXMAP_SYM

#include "pixMapSymBase.h"

#pragma warning ( push, 0 )

#include <ft2build.h>
#include FT_FREETYPE_H

#pragma warning ( pop )

/**
PixMapSym holds the representation('pixels') of symbol 'symCode'.

Expects symbols that mostly fit the Bounding Box provided to the constructor.
That is the symbols are either already within the BBox,
or their size needs to be at most a few pixels more than the BBox (they'll get trimmed).

The height('rows'), width('cols') are also recorded.
The bitmap pitch is transformed into width, so 'pixels' is continuous.
Fields 'left' and 'top' indicate the position of the top-left corner within the drawing square.
*/
class PixMapSym : public IPixMapSym {
protected:
	/// glyph's mass center (coordinates are within a unit-square: 0..1 x 0..1)
	cv::Point2d mc;

	/// 256-shades of gray rectangle describing the character (top-down, left-right traversal)
	std::vector<unsigned char> pixels;

	cv::Mat colSums; ///< row with the sums of the pixels of each column of the symbol (each pixel in 0..1)
	cv::Mat rowSums; ///< row with the sums of the pixels of each row of the symbol (each pixel in 0..1)

	size_t symIdx = 0ULL;			///< symbol index within cmap
	double avgPixVal = 0.;			///< average of the pixel values divided by 255

	unsigned long symCode = 0UL;	///< symbol code

	unsigned char rows = 0U, cols = 0U;	// dimensions of 'pixels' rectangle from below 
	unsigned char left = 0U, top = 0U;	// position within the drawing square

	bool removable = false;	///< when set to true, the symbol will appear as marked (inversed) in the cmap viewer

	void knownSize(int rows_, int cols_);		///< provide symbol box size
	void knownPosition(int top_, int left_);	///< provide symbol box position

public:
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

	/// glyph's mass center (coordinates are within a unit-square: 0..1 x 0..1)
	const cv::Point2d& getMc() const override final;

	/// Row with the sums of the pixels of each column of the symbol (each pixel in 0..1)
	const cv::Mat& getColSums() const override final;

	/// Row with the sums of the pixels of each row of the symbol (each pixel in 0..1)
	const cv::Mat& getRowSums() const override final;

	size_t getSymIdx() const override final;			///< symbol index within cmap
	double getAvgPixVal() const override final;			///< average of the pixel values divided by 255
	unsigned long getSymCode() const override final;	///< the code of the symbol
	unsigned char getRows() const override final;		///< the height of the representation of this symbol
	unsigned char getCols() const override final;		///< the width of the representation of this symbol
	unsigned char getLeft() const override final;		///< horizontal position within the drawing square
	unsigned char getTop() const override final;		///< vertical position within the drawing square

	/// When set to true, the symbol will appear as marked (inversed) in the cmap viewer
	bool isRemovable() const override final;
	void setRemovable(bool removable_ = true) override final; ///< allows changing removable
	
	cv::Mat asNarrowMat() const override final;	///< a matrix with the symbol within its tight bounding box

	/// the symbol within a square of fontSz x fontSz, either as is, or inversed
	cv::Mat toMat(unsigned fontSz, bool inverse = false) const override;

	/// Conversion PixMapSym .. Mat of type double with range [0..1] instead of [0..255]
	cv::Mat toMatD01(unsigned fontSz) const override;

	/**
	Computing the mass center (mc) and average pixel value (divided by 255) of a given symbol.
	When the parameters colSums, rowSums are not nullptr, the corresponding sum is returned.

	It's static for easier Unit Testing.
	*/
	static void computeMcAndAvgPixVal(unsigned sz, double maxGlyphSum,
									  const std::vector<unsigned char> &data,
									  unsigned char rows, unsigned char cols,
									  unsigned char left, unsigned char top,
									  const cv::Mat &consec, const cv::Mat &revConsec,
									  cv::Point2d &mc, double &avgPixVal,
									  cv::Mat *colSums = nullptr, cv::Mat *rowSums = nullptr);

#ifdef UNIT_TESTING
	/// Constructs a PixMapSym in Unit Testing mode
	PixMapSym(const std::vector<unsigned char> &data, ///< values of the symbol's pixels
			  const cv::Mat &consec,	///< vector of consecutive values 0 .. sz-1
			  const cv::Mat &revConsec	///< vector of consecutive values sz-1 .. 0
			  );
#endif // UNIT_TESTING defined
};

#endif // H_PIXMAP_SYM
