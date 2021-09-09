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

#ifndef H_EXT_BOX_BLUR
#define H_EXT_BOX_BLUR

#include "blur.h"

namespace pic2sym::blur {

/*
Extended Box blurring with border repetition.

Based on the considerations found in:
http://www.mia.uni-saarland.de/Publications/gwosdek-ssvm11.pdf

The standard deviation and the iterations count are configurable.

This blur still has O(1) / pixel time-performance, like the box blur and was
introduced due to its quality: results are extremely similar to Gaussian
filtering after only 2 iterations.
*/
class ExtBoxBlur : public BlurEngine {
 public:
  /// Configure the filter through the desired standard deviation and the
  /// iterations count
  ExtBoxBlur(double desiredSigma, unsigned iterations_ = 1U) noexcept(!UT);

  /// Reconfigure the filter through a new desired standard deviation and a new
  /// iterations count
  ExtBoxBlur& setSigma(double desiredSigma,
                       unsigned iterations_ = 1U) noexcept(!UT);

  /// Reconfigure iterations count
  ExtBoxBlur& setIterations(unsigned iterations_) noexcept(!UT);

 protected:
  /// Actual implementation for the current configuration. toBlur is checked;
  /// blurred is initialized
  void doProcess(const cv::Mat& toBlur,
                 cv::Mat& blurred,
                 bool forTinySym) const noexcept override;

 private:
  /// Handle class
  class Impl;

  static Impl& nonTinySyms() noexcept;  ///< handler for non-tiny symbols
  static Impl& tinySyms() noexcept;     ///< handler for tiny symbols

  /// @return a fully configured instance
  static const ExtBoxBlur& configuredInstance() noexcept(!UT);

  /// Registers the configured instance plus its name
  static ConfInstRegistrator cir;
};

}  // namespace pic2sym::blur

#endif  // H_EXT_BOX_BLUR
