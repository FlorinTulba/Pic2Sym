/******************************************************************************
 The application Pic2Sym approximates images by a
 grid of colored symbols with colored backgrounds.

 Copyrights from the libraries used by the program:
 - (c) 2003 Boost (www.boost.org)
     License: doc/licenses/Boost.lic
     http://www.boost.org/LICENSE_1_0.txt
 - (c) 2015-2016 OpenCV (www.opencv.org)
     License: doc/licenses/OpenCV.lic
     http://opencv.org/license/
 - (c) 1996-2002, 2006 The FreeType Project (www.freetype.org)
     License: doc/licenses/FTL.txt
     http://git.savannah.gnu.org/cgit/freetype/freetype2.git/plain/docs/FTL.TXT
 - (c) 1997-2002 OpenMP Architecture Review Board (www.openmp.org)
   (c) Microsoft Corporation (implementation for OpenMP C/C++ v2.0 March 2002)
     See: https://msdn.microsoft.com/en-us/library/8y6825x5.aspx
 - (c) 1995-2017 zlib software (Jean-loup Gailly and Mark Adler - www.zlib.net)
     License: doc/licenses/zlib.lic
     http://www.zlib.net/zlib_license.html


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
 If not, see: http://www.gnu.org/licenses/agpl-3.0.txt .
 *****************************************************************************/

#ifndef H_PIXMAP_SYM_BASE
#define H_PIXMAP_SYM_BASE

#pragma warning(push, 0)

#include <memory>
#include <vector>

#include <opencv2/core/core.hpp>

#pragma warning(pop)

/**
IPixMapSym handles the representation('pixels') of symbol 'symCode'.

Expects symbols that mostly fit the Bounding Box provided to the constructor.
That is the symbols are either already within the BBox,
or their size needs to be at most a few pixels more than the BBox (they'll get
trimmed).

The height('rows'), width('cols') are also recorded.
The bitmap pitch is transformed into width, so 'pixels' is continuous.
Fields 'left' and 'top' indicate the position of the top-left corner within the
drawing square.
*/
class IPixMapSym /*abstract*/ {
 public:
  /// glyph's mass center (coordinates are within a unit-square: 0..1 x 0..1)
  virtual const cv::Point2d& getMc() const noexcept = 0;

  /// Row with the sums of the pixels of each column of the symbol (each pixel
  /// in 0..1)
  virtual const cv::Mat& getColSums() const noexcept = 0;

  /// Row with the sums of the pixels of each row of the symbol (each pixel in
  /// 0..1)
  virtual const cv::Mat& getRowSums() const noexcept = 0;

  virtual size_t getSymIdx() const noexcept = 0;  ///< symbol index within cmap

  /// Average of the pixel values divided by 255
  virtual double getAvgPixVal() const noexcept = 0;

  /// Code of the symbol
  virtual unsigned long getSymCode() const noexcept = 0;

  /// Height of the representation of this symbol
  virtual unsigned char getRows() const noexcept = 0;

  /// Width of the representation of this symbol
  virtual unsigned char getCols() const noexcept = 0;

  /// Horizontal position within the drawing square
  virtual unsigned char getLeft() const noexcept = 0;

  /// Vertical position within the drawing square
  virtual unsigned char getTop() const noexcept = 0;

  /// When set to true, the symbol will appear as marked (inversed) in the cmap
  /// viewer
  virtual bool isRemovable() const noexcept = 0;

  /// Allows changing removable
  virtual void setRemovable(bool removable_ = true) noexcept = 0;

  /// A matrix with the symbol within its tight bounding box
  virtual cv::Mat asNarrowMat() const noexcept = 0;

  /// the symbol within a square of fontSz x fontSz, either as is, or inversed
  virtual cv::Mat toMat(unsigned fontSz, bool inverse = false) const
      noexcept = 0;

  /// Conversion PixMapSym .. Mat of type double with range [0..1] instead of
  /// [0..255]
  virtual cv::Mat toMatD01(unsigned fontSz) const noexcept = 0;

  virtual ~IPixMapSym() noexcept {}

  // If slicing is observed and becomes a severe problem, use `= delete` for all
  IPixMapSym(const IPixMapSym&) noexcept = default;
  IPixMapSym(IPixMapSym&&) noexcept = default;
  IPixMapSym& operator=(const IPixMapSym&) noexcept = default;
  IPixMapSym& operator=(IPixMapSym&&) noexcept = default;

 protected:
  constexpr IPixMapSym() noexcept {}
};

/// Vector of IPixMapSym objects
typedef std::vector<std::unique_ptr<const IPixMapSym>> VPixMapSym;

#endif  // H_PIXMAP_SYM_BASE
