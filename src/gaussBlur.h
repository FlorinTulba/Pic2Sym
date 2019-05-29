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

#ifndef H_GAUSS_BLUR
#define H_GAUSS_BLUR

#include "blur.h"

/*
Just forwarding to GaussianBlur from OpenCV.
Using only replicated borders.

This is the reference quality blur, but it's also the most time-expensive
filter.
*/
class GaussBlur : public BlurEngine {
 public:
  /**
  Configure the filter through the desired standard deviation and the mask
  width
  @throw invalid_argument only in UnitTesting if desiredSigma<=0 or kernelWidth_
  isn't 0 or odd

  Exception to be only reported, not handled
  */
  GaussBlur(double desiredSigma, unsigned kernelWidth_ = 0U) noexcept(!UT);

  /**
  Reconfigure the filter through a new desired standard deviation and a new
  mask width
  @throw invalid_argument if desiredSigma<=0 or kernelWidth_ isn't 0 or odd

  Exception to be only reported, not handled. It propagates only in UnitTesting
  */
  GaussBlur& configure(double desiredSigma,
                       unsigned kernelWidth_ = 0U) noexcept(!UT);

 protected:
  /// Relevant Gaussian blur parameters
  struct BlurParams {
    double sigma;  ///< desired standard deviation

    /// Mask width (must be an odd value, or 0 - to let it be set from sigma)
    unsigned kernelWidth;
  };

  BlurParams nonTinySymsParams;  ///< parameters for non-tiny symbols
  BlurParams tinySymsParams;     ///< parameters for tiny symbols

  /// Forwarding to GaussianBlur. toBlur is checked; blurred is initialized
  void doProcess(const cv::Mat& toBlur, cv::Mat& blurred, bool forTinySym) const
      noexcept override;

 private:
  /// @return a fully configured instance
  static const GaussBlur& configuredInstance() noexcept(!UT);

  /// Registers the configured instance plus its name
  static ConfInstRegistrator cir;
};

#endif  // H_GAUSS_BLUR
