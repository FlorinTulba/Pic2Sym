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

#ifndef H_PIXMAP_SYM_BASE
#define H_PIXMAP_SYM_BASE

#pragma warning ( push, 0 )

#include "std_memory.h"
#include <vector>

#include <opencv2/core/core.hpp>

#pragma warning ( pop )

/**
IPixMapSym handles the representation('pixels') of symbol 'symCode'.

Expects symbols that mostly fit the Bounding Box provided to the constructor.
That is the symbols are either already within the BBox,
or their size needs to be at most a few pixels more than the BBox (they'll get trimmed).

The height('rows'), width('cols') are also recorded.
The bitmap pitch is transformed into width, so 'pixels' is continuous.
Fields 'left' and 'top' indicate the position of the top-left corner within the drawing square.
*/
struct IPixMapSym /*abstract*/ {
	/// glyph's mass center (coordinates are within a unit-square: 0..1 x 0..1)
	virtual const cv::Point2d& getMc() const = 0;

	/// Row with the sums of the pixels of each column of the symbol (each pixel in 0..1)
	virtual const cv::Mat& getColSums() const = 0;

	/// Row with the sums of the pixels of each row of the symbol (each pixel in 0..1)
	virtual const cv::Mat& getRowSums() const = 0;

	virtual size_t getSymIdx() const = 0;			///< symbol index within cmap
	virtual double getAvgPixVal() const = 0;		///< average of the pixel values divided by 255
	virtual unsigned long getSymCode() const = 0;	///< the code of the symbol
	virtual unsigned char getRows() const = 0;		///< the height of the representation of this symbol
	virtual unsigned char getCols() const = 0;		///< the width of the representation of this symbol
	virtual unsigned char getLeft() const = 0;		///< horizontal position within the drawing square
	virtual unsigned char getTop() const = 0;		///< vertical position within the drawing square

	/// When set to true, the symbol will appear as marked (inversed) in the cmap viewer
	virtual bool isRemovable() const = 0;
	virtual void setRemovable(bool removable_ = true) = 0; ///< allows changing removable

	virtual cv::Mat asNarrowMat() const = 0;	///< a matrix with the symbol within its tight bounding box

	/// the symbol within a square of fontSz x fontSz, either as is, or inversed
	virtual cv::Mat toMat(unsigned fontSz, bool inverse = false) const = 0;

	/// Conversion PixMapSym .. Mat of type double with range [0..1] instead of [0..255]
	virtual cv::Mat toMatD01(unsigned fontSz) const = 0;

	virtual ~IPixMapSym() = 0 {}
};

/// Vector of IPixMapSym objects
typedef std::vector<const std::uniquePtr<const IPixMapSym>> VPixMapSym;

#endif // H_PIXMAP_SYM_BASE
