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

#include "precompiled.h"
// This keeps precompiled.h first; Otherwise header sorting might move it

#include "gaussBlur.h"

#include "warnings.h"

#pragma warning(push, 0)

#include <opencv2/imgproc/imgproc.hpp>

#include <gsl/gsl>

#pragma warning(pop)

using namespace std;
using namespace cv;

namespace pic2sym {

extern const int StructuralSimilarity_RecommendedWindowSide;
extern const double StructuralSimilarity_SIGMA;

namespace blur {

GaussBlur::GaussBlur(double desiredSigma,
                     unsigned kernelWidth_ /* = 0U*/) noexcept(!UT) {
  configure(desiredSigma, kernelWidth_);
}

#pragma warning(disable : WARN_THROWS_ALTHOUGH_NOEXCEPT)
GaussBlur& GaussBlur::configure(double desiredSigma,
                                unsigned kernelWidth_ /* = 0U*/) noexcept(!UT) {
  EXPECTS_OR_REPORT_AND_THROW_CONST_MSG(
      desiredSigma > 0., invalid_argument,
      HERE.function_name() + " desiredSigma should be > 0!"s);

  EXPECTS_OR_REPORT_AND_THROW_CONST_MSG(
      !kernelWidth_ || ((kernelWidth_ & 1U) == 1U), invalid_argument,
      HERE.function_name() + " kernelWidth_ should be an odd value or 0!"s);

  nonTinySymsParams = {.sigma = desiredSigma, .kernelWidth = kernelWidth_};

  // Tiny symbols should use a sigma = desiredSigma/2. and kernel whose width is
  // next odd value >= kernelWidth_/2.
  tinySymsParams = {.sigma = desiredSigma * .5,
                    .kernelWidth = (kernelWidth_ >> 1) | 1U};

  return *this;
}
#pragma warning(default : WARN_THROWS_ALTHOUGH_NOEXCEPT)

void GaussBlur::doProcess(const cv::Mat& toBlur,
                          cv::Mat& blurred,
                          bool forTinySym) const noexcept {
  if (forTinySym)
    GaussianBlur(
        toBlur, blurred,
        Size{(int)tinySymsParams.kernelWidth, (int)tinySymsParams.kernelWidth},
        tinySymsParams.sigma, tinySymsParams.sigma, BORDER_REPLICATE);
  else
    GaussianBlur(toBlur, blurred,
                 Size{(int)nonTinySymsParams.kernelWidth,
                      (int)nonTinySymsParams.kernelWidth},
                 nonTinySymsParams.sigma, nonTinySymsParams.sigma,
                 BORDER_REPLICATE);
}

const GaussBlur& GaussBlur::configuredInstance() noexcept(!UT) {
  // Gaussian blur with desired standard deviation and window width
  static GaussBlur result{StructuralSimilarity_SIGMA,
                          (unsigned)StructuralSimilarity_RecommendedWindowSide};

  return result;
}

}  // namespace blur
}  // namespace pic2sym