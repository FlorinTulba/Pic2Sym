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

#include "precompiled.h"

#include "gaussBlur.h"
#include "warnings.h"

#pragma warning(push, 0)

#include <opencv2/imgproc/imgproc.hpp>

#pragma warning(pop)

using namespace std;
using namespace cv;

GaussBlur::GaussBlur(double desiredSigma,
                     unsigned kernelWidth_ /* = 0U*/) noexcept(!UT) {
  configure(desiredSigma, kernelWidth_);
}

#pragma warning(disable : WARN_THROWS_ALTHOUGH_NOEXCEPT)
GaussBlur& GaussBlur::configure(double desiredSigma,
                                unsigned kernelWidth_ /* = 0U*/) noexcept(!UT) {
  if (desiredSigma <= 0.)
    THROW_WITH_CONST_MSG(__FUNCTION__ " desiredSigma should be > 0!",
                         invalid_argument);
  if ((kernelWidth_ != 0U) && ((kernelWidth_ & 1U) != 1U))
    THROW_WITH_CONST_MSG(
        __FUNCTION__ " kernelWidth_ should be an odd value or 0!",
        invalid_argument);

  nonTinySymsParams.sigma = desiredSigma;
  nonTinySymsParams.kernelWidth = kernelWidth_;

  // Tiny symbols should use a sigma = desiredSigma/2. and kernel whose width is
  // next odd value >= kernelWidth_/2.
  tinySymsParams.sigma = desiredSigma * .5;
  tinySymsParams.kernelWidth = (kernelWidth_ >> 1) | 1U;

  return *this;
}
#pragma warning(default : WARN_THROWS_ALTHOUGH_NOEXCEPT)

void GaussBlur::doProcess(const cv::Mat& toBlur,
                          cv::Mat& blurred,
                          bool forTinySym) const noexcept {
  if (forTinySym)
    GaussianBlur(
        toBlur, blurred,
        Size((int)tinySymsParams.kernelWidth, (int)tinySymsParams.kernelWidth),
        tinySymsParams.sigma, tinySymsParams.sigma, BORDER_REPLICATE);
  else
    GaussianBlur(toBlur, blurred,
                 Size((int)nonTinySymsParams.kernelWidth,
                      (int)nonTinySymsParams.kernelWidth),
                 nonTinySymsParams.sigma, nonTinySymsParams.sigma,
                 BORDER_REPLICATE);
}

const GaussBlur& GaussBlur::configuredInstance() noexcept(!UT) {
  // Gaussian blur with desired standard deviation and window width
  extern const int StructuralSimilarity_RecommendedWindowSide;
  extern const double StructuralSimilarity_SIGMA;

  static GaussBlur result(StructuralSimilarity_SIGMA,
                          (unsigned)StructuralSimilarity_RecommendedWindowSide);

  return result;
}
