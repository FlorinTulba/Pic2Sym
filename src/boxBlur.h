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

#ifndef H_BOX_BLUR
#define H_BOX_BLUR

#include "blur.h"

namespace pic2sym::blur {

/*
Box blurring with optional repetitions and border repetition.
Allows configuring also a desired standard deviation.

Based on the considerations found in:
http://www.web.uwa.edu.au/__data/assets/file/0008/826172/filterdesign.pdf

For the basic standalone box filtering, blur from OpenCV was used.

This blur type was included to provide the quickest available sequential blur
(this happens when iterations_ is 1). Its performance doesn't depend on the
boxWidth / sigma and is only proportional to iterations_. For sigma = 1.5
(typical value used during this application) and iterations_ = 1, box blur
normally needs just 1/3 of the time of GaussianBlur with the same sigma.

However, box blur quality depends on iterations_. For values larger than 2 it
gets more similar to Gaussian's quality.
*/
class BoxBlur : public BlurEngine {
 public:
  /// Configure the filter through the mask width and the iterations count
  BoxBlur(unsigned boxWidth_ = 1U, unsigned iterations_ = 1U) noexcept(!UT);

  /// Reconfigure the filter through a new desired standard deviation and a new
  /// iterations count
  BoxBlur& setSigma(double desiredSigma,
                    unsigned iterations_ = 1U) noexcept(!UT);

  /// Reconfigure mask width
  BoxBlur& setWidth(unsigned boxWidth_) noexcept(!UT);

  /// Reconfigure iterations count
  BoxBlur& setIterations(unsigned iterations_) noexcept(!UT);

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
  static const BoxBlur& configuredInstance() noexcept(!UT);

  /// Registers the configured instance plus its name
  static ConfInstRegistrator cir;
};

}  // namespace pic2sym::blur

#endif  // H_BOX_BLUR
