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

#ifndef H_PATCH
#define H_PATCH

#include "patchBase.h"

namespace pic2sym::input {

/**
Holds useful patch information.

It decides whether this patch needs approximation or not - uniform patches
don't produce interesting approximations.
*/
class Patch : public IPatch {
 public:
  /**
  Initializer

  @param orig_ patch to be approximated
  @param blurred_ blurred version of the patch, either considering real borders,
  or replicated ones
  @param isColor_ type of image - color => true; grayscale => false
  */
  Patch(const cv::Mat& orig_, const cv::Mat& blurred_, bool isColor_) noexcept;

  /// The patch to approximate
  const cv::Mat& getOrig() const noexcept final;

  /// The blurred version of the orig
  const cv::Mat& getBlurred() const noexcept final;

  /// Is the patch color or grayscale?
  bool isColored() const noexcept final;

  /// Patches that appear uniform use 'blurred' as approximation
  bool nonUniform() const noexcept final;

  /**
  Specifies which matrix to use during the approximation process (gray, of type
  double)
  @throw logic_error if called for uniform patches

  Exception to be only reported, not handled
  */
  const cv::Mat& matrixToApprox() const noexcept(!UT) override;

  /// @return a clone of itself
  std::unique_ptr<const IPatch> clone() const noexcept override;

#ifdef UNIT_TESTING
  /// Constructor delegating its job to the one with 3 parameters
  explicit Patch(const cv::Mat& orig_) noexcept;

  /// Specifies which new matrix to use during the approximation process (gray,
  /// of type double); @return itself
  Patch& setMatrixToApprox(const cv::Mat& m) noexcept;

  /// Forces needsApproximation value on true
  void forceApproximation() noexcept;
#endif  // UNIT_TESTING defined

 private:
  /// Gray version of the patch to process (its data type is double)
  cv::Mat grayD;
  cv::Mat orig;     ///< the patch to approximate
  cv::Mat blurred;  ///< the blurred version of the orig

  bool isColor;  ///< is the patch color or grayscale?

  /// Patches that appear uniform use 'blurred' as approximation
  bool needsApproximation{true};
};

}  // namespace pic2sym::input

#endif  // H_PATCH
