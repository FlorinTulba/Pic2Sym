/******************************************************************************
 The application Pic2Sym approximates images by a
 grid of colored symbols with colored backgrounds.

 Copyrights from the libraries used by the program:
 - (c) 2003-2021 Boost (www.boost.org)
     License: doc/licenses/Boost.lic
     http://www.boost.org/LICENSE_1_0.txt
 - (c) 2015-2021 OpenCV (www.opencv.org)
     License: doc/licenses/OpenCV.lic
     http://opencv.org/license/
 - (c) 1996-2021 The FreeType Project (www.freetype.org)
     License: doc/licenses/FTL.txt
     http://git.savannah.gnu.org/cgit/freetype/freetype2.git/plain/docs/FTL.TXT
 - (c) 1997-2021 OpenMP Architecture Review Board (www.openmp.org)
   (c) Microsoft Corporation (implementation for OpenMP C/C++ v2.0 March 2002)
     See: https://msdn.microsoft.com/en-us/library/8y6825x5.aspx
 - (c) 1995-2021 zlib software (Jean-loup Gailly and Mark Adler - www.zlib.net)
     License: doc/licenses/zlib.lic
     http://www.zlib.net/zlib_license.html
 - (c) 2015-2021 Microsoft Guidelines Support Library - github.com/microsoft/GSL
     License: doc/licenses/MicrosoftGSL.lic
     https://raw.githubusercontent.com/microsoft/GSL/main/LICENSE


 (c) 2016-2021 Florin Tulba <florintulba@yahoo.com>

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

#ifndef H_SYM_DATA_BASE
#define H_SYM_DATA_BASE

#include "misc.h"

#pragma warning(push, 0)

#include <array>
#include <memory>
#include <opencv2/core/core.hpp>
#include <stdexcept>
#include <vector>

#include <gsl/gsl>

#pragma warning(pop)

namespace pic2sym::syms {

/// Interface for symbol data
class ISymData /*abstract*/ {
 public:
  /// Indices of each matrix type within a MatArray object
  enum struct MaskType {
    Fg,    ///< mask isolating the foreground of the glyph
    Bg,    ///< mask isolating the background of the glyph
    Edge,  ///< mask isolating the edge of the glyph (transition region fg-bg)

    /// Symbol shifted (in brightness) to have black background (0..1)
    GroundedSym,

    BlurredGrSym,   ///< blurred version of the grounded symbol (0..1)
    VarianceGrSym,  ///< variance of the grounded symbol

    /// KEEP THIS LAST and DON'T USE IT AS INDEX in MatArray objects!
    MATRICES_COUNT
  };

  // For each symbol from cmap, there'll be several additional helpful matrices
  // to store along with the one for the given glyph. The enum from above should
  // be used for selection.
  using MatArray = std::array<cv::Mat, (size_t)MaskType::MATRICES_COUNT>;

  /// Retrieve specific mask
  const cv::Mat& getMask(MaskType maskIdx) const noexcept {
    Expects(maskIdx < MaskType::MATRICES_COUNT);

    return getMasks()[(size_t)maskIdx];
  }

  /// mass center of the symbol given original fg & bg (coordinates are within a
  /// unit-square: 0..1 x 0..1)
  virtual const cv::Point2d& getMc() const noexcept = 0;

  /// negative of the symbol (0..255 byte for normal symbols; double for tiny)
  virtual const cv::Mat& getNegSym() const noexcept = 0;

  /// The pixel values (double) are shifted so that the average pixel value
  /// (miu) is 0
  virtual const cv::Mat& getSymMiu0() const noexcept = 0;

  /// norm L2 of (symbol - average pixel value)
  virtual double getNormSymMiu0() const noexcept = 0;

  /// various masks
  virtual const MatArray& getMasks() const noexcept = 0;

  /// symbol index within cmap
  virtual size_t getSymIdx() const noexcept = 0;

#ifdef UNIT_TESTING
  /// the value of darkest pixel, range 0..1
  virtual double getMinVal() const noexcept = 0;
#endif  // UNIT_TESTING defined

  /// difference between brightest and darkest pixels, each in 0..1
  virtual double getDiffMinMax() const noexcept = 0;

  /// average pixel value, each pixel in 0..1
  virtual double getAvgPixVal() const noexcept = 0;

  /// the code of the symbol
  virtual unsigned long getCode() const noexcept = 0;

  /**
  Enabled symbol filters might mark this symbol as removable,
  but PreserveRemovableSymbolsForExamination from configuration might allow it
  to remain in the active symbol set used during image transformation.

  However, when removable == true && PreserveRemovableSymbolsForExamination,
  the symbol will appear as marked (inversed) in the cmap viewer

  This field doesn't need to be serialized, as filtering options might be
  different for distinct run sessions.
  */
  virtual bool isRemovable() const noexcept = 0;

  virtual ~ISymData() noexcept = 0 {}
};

/// VSymData - vector with most information about each symbol
using VSymData = std::vector<std::unique_ptr<const ISymData>>;

}  // namespace pic2sym::syms

#endif  // H_SYM_DATA_BASE
