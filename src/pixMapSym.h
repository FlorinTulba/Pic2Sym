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

#ifndef H_PIXMAP_SYM
#define H_PIXMAP_SYM

#include "pixMapSymBase.h"

#pragma warning(push, 0)

#include <compare>

#include <ft2build.h>
#include FT_FREETYPE_H

#pragma warning(pop)

/**
PixMapSym holds the representation('pixels') of symbol 'symCode'.

Expects symbols that mostly fit the Bounding Box provided to the constructor.
That is the symbols are either already within the BBox,
or their size needs to be at most a few pixels more than the BBox (they'll get
trimmed).

The height('rows'), width('cols') are also recorded.
The bitmap pitch is transformed into width, so 'pixels' is continuous.
Fields 'left' and 'top' indicate the position of the top-left corner within the
drawing square.
*/
class PixMapSym : public IPixMapSym {
 public:
  /**
  Processes a FT_Bitmap object to store a faithful representation of the symCode
  drawn within a bounding box (BBox).

  The symbols must be:
  - either already within the BBox,
  - or their size needs to be at most a few pixels more than the BBox (they'll
  get trimmed).

  The other symbols need resizing before feeding the PixMapSym.

  After shifting the symbol inside, the regions still lying outside are trimmed.

  Any bitmap pitch is removed => storing pixels without gaps.
  */
  PixMapSym(
      unsigned long symCode,  ///< the symbol code
      size_t symIdx_,         ///< symbol index within cmap
      const FT_Bitmap& bm,    ///< the bitmap to process
      int leftBound,  ///< initial position of the symbol considered from the
                      ///< left
      int topBound,  ///< initial position of the symbol considered from the top
      int sz,        ///< font size
      double maxGlyphSum,        ///< max sum of a glyph's pixels
      const cv::Mat& consec,     ///< vector of consecutive values 0 .. sz-1
      const cv::Mat& revConsec,  ///< vector of consecutive values sz-1 .. 0
      const FT_BBox& bb          ///< the bounding box to fit
      ) noexcept;

  ~PixMapSym() noexcept = default;

  PixMapSym(const PixMapSym&) noexcept;
  PixMapSym(PixMapSym&&) noexcept;
  PixMapSym& operator=(PixMapSym&& other) noexcept;

  void operator=(const PixMapSym&) = delete;

#ifdef __cpp_lib_three_way_comparison
  /// Useful to detect duplicates
  std::strong_equality operator<=>(const PixMapSym& other) const noexcept;

#else   // __cpp_lib_three_way_comparison not defined
  /// Useful to detect duplicates
  bool operator==(const PixMapSym& other) const noexcept;

  /// Useful to detect duplicates
  bool operator!=(const PixMapSym& other) const noexcept {
    return !(*this == other);
  }
#endif  // __cpp_lib_three_way_comparison

  /// glyph's mass center (coordinates are within a unit-square: 0..1 x 0..1)
  const cv::Point2d& getMc() const noexcept final;

  /// Row with the sums of the pixels of each column of the symbol (each pixel
  /// in 0..1)
  const cv::Mat& getColSums() const noexcept final;

  /// Row with the sums of the pixels of each row of the symbol (each pixel in
  /// 0..1)
  const cv::Mat& getRowSums() const noexcept final;

  /// Symbol index within cmap
  size_t getSymIdx() const noexcept final;

  /// Average of the pixel values divided by 255
  double getAvgPixVal() const noexcept final;

  /// Code of the symbol
  unsigned long getSymCode() const noexcept final;

  /// Height of the representation of this symbol
  unsigned char getRows() const noexcept final;

  /// Width of the representation of this symbol
  unsigned char getCols() const noexcept final;

  /// Horizontal position within the drawing square
  unsigned char getLeft() const noexcept final;

  /// Vertical position within the drawing square
  unsigned char getTop() const noexcept final;

  /// When set to true, the symbol will appear as marked (inversed) in the cmap
  /// viewer
  bool isRemovable() const noexcept final;

  /// Allows changing removable
  void setRemovable(bool removable_ = true) noexcept final;

  /// A matrix with the symbol within its tight bounding box
  cv::Mat asNarrowMat() const noexcept final;

  /// the symbol within a square of fontSz x fontSz, either as is, or inversed
  cv::Mat toMat(unsigned fontSz, bool inverse = false) const noexcept override;

  /// Conversion PixMapSym .. Mat of type double with range [0..1] instead of
  /// [0..255]
  cv::Mat toMatD01(unsigned fontSz) const noexcept override;

  /**
  Computing the mass center (mc) and average pixel value (divided by 255) of a
  given symbol. When the parameters colSums, rowSums are not nullptr, the
  corresponding sum is returned.

  It's static for easier Unit Testing.
  */
  static void computeMcAndAvgPixVal(unsigned sz,
                                    double maxGlyphSum,
                                    const std::vector<unsigned char>& data,
                                    unsigned char rows,
                                    unsigned char cols,
                                    unsigned char left,
                                    unsigned char top,
                                    const cv::Mat& consec,
                                    const cv::Mat& revConsec,
                                    cv::Point2d& mc,
                                    double& avgPixVal,
                                    cv::Mat* colSums = nullptr,
                                    cv::Mat* rowSums = nullptr) noexcept;

#ifdef UNIT_TESTING
  /// Constructs a PixMapSym in Unit Testing mode
  PixMapSym(
      const std::vector<unsigned char>&
          data,                 ///< values of the symbol's pixels
      const cv::Mat& consec,    ///< vector of consecutive values 0 .. sz-1
      const cv::Mat& revConsec  ///< vector of consecutive values sz-1 .. 0
      ) noexcept;
#endif  // UNIT_TESTING defined

 protected:
  /// Provide symbol box size
  void knownSize(int rows_, int cols_) noexcept;

  /// Provide symbol box position
  void knownPosition(int top_, int left_) noexcept;

 private:
  /// glyph's mass center (coordinates are within a unit-square: 0..1 x 0..1)
  cv::Point2d mc;

  /// 256-shades of gray rectangle describing the character (top-down,
  /// left-right traversal)
  std::vector<unsigned char> pixels;

  /// Row with the sums of the pixels of each column of the symbol (each pixel
  /// in 0..1)
  cv::Mat colSums;

  /// Row with the sums of the pixels of each row of the symbol (each pixel in
  /// 0..1)
  cv::Mat rowSums;

  size_t symIdx = 0ULL;   ///< symbol index within cmap
  double avgPixVal = 0.;  ///< average of the pixel values divided by 255

  unsigned long symCode = 0UL;  ///< symbol code

  unsigned char rows = 0U,
                cols = 0U;  // dimensions of 'pixels' rectangle from below
  unsigned char left = 0U, top = 0U;  // position within the drawing square

  /// When set to true, the symbol will appear as marked (inversed) in the cmap
  /// viewer
  bool removable = false;
};

#endif  // H_PIXMAP_SYM
