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

#ifndef H_BLUR
#define H_BLUR

#include "blurBase.h"
#include "misc.h"

#pragma warning(push, 0)

#include <string>
#include <unordered_map>

#pragma warning(pop)

namespace pic2sym::blur {

/**
Base class for various versions of blurring that can be configured within the
application.

One possible matching aspect to be used during image approximation is
Structural Similarity (https://ece.uwaterloo.ca/~z70wang/research/ssim ).
It relies heavily on Gaussian blurring, whose implementation is already
optimized in OpenCV for a sequential run.

This class addresses the issue that GaussianBlur function is the most
time-consuming operation during image approximation.

For the typical standard deviation of 1.5, GaussianBlur from OpenCV still
remains the fastest when compared to other tested sequential innovative
algorithms:
- Young & van Vliet (implementation from CImg library - http://cimg.eu/)
- Deriche (implementation from CImg library - http://cimg.eu/)
- Stacked Integral Image (implementation from
http://dev.ipol.im/~getreuer/code/doc/gaussian_20131215_doc/group__sii__gaussian.html)

All those competitor algorithms are less accurate than Extended Box Blur
configured with just 2 repetitions. When applied only once, sequential Box-based
blur techniques can be up to 3 times faster than GaussianBlur from OpenCV.
However, basic Box blurring with no repetitions has poor quality,
while Extended Box blurring incurs additional time costs for an improved
quality.

An implementation of the Extended Box blurring exists at:
http://dev.ipol.im/~getreuer/code/doc/gaussian_20131215_doc/group__ebox__gaussian.html
This project contains its own implementation of this blur technique
(ExtBoxBlur).

The project includes following blur algorithms:
- GaussBlur - the reference blur, delegating to sequential GaussianBlur from
OpenCV
- BoxBlur - for its versatility: quickest for no repetitions and slower, but
increasingly accurate for more repetitions (Every repetition delegates to blur
from OpenCV)
- ExtBoxBlur - for its accuracy, even for only a few repetitions. The sequential
algorithm is highly parallelizable

All derived classes are expected to provide a static method that provides an
instance of them already configured for blurring serving structural similarity
matching aspect:

  static const Derived& configuredInstance();

Besides, the derived classes will also declare a static field:

  static ConfInstRegistrator cir;

that will be initialized in varConfig.cpp unit like this:

  BlurEngine::ConfInstRegistrator
Derived::cir("<blurTypeName_from_varConfig.txt>",
Derived::configuredInstance());
*/
class BlurEngine /*abstract*/ : public IBlurEngine {
 public:
  /**
  Provides a specific, completely configured blur engine.
  @throw invalid_argument for an unrecognized blurType

  Exception to be only reported, not handled
  */
  static const IBlurEngine& byName(const std::string& blurType) noexcept(!UT);

  /**
  Template method checking toBlur, initializing blurred and calling doProcess

  @param toBlur is a single channel matrix with values of type double
  @param blurred the result of the blur (not initialized when method gets
  called)
  @param forTinySym demands generating a Gaussian blur with smaller window and
  standard deviation for tiny symbols

  @throw invalid_argument for empty toBlur or if it contains not double values

  Exception to be only reported, not handled
  */
  void process(const cv::Mat& toBlur, cv::Mat& blurred, bool forTinySym) const
      noexcept(!UT) override;

 protected:
  /// Mapping type between blurTypes and corresponding configured blur instances
  using ConfiguredInstances = std::unordered_map<const std::string,
                                                 const IBlurEngine*,
                                                 std::hash<std::string>>;

  BlurEngine() noexcept : IBlurEngine() {}

  // Slicing prevention
  BlurEngine(const BlurEngine&) = delete;
  BlurEngine(BlurEngine&&) = delete;
  void operator=(const BlurEngine&) = delete;
  void operator=(BlurEngine&&) = delete;

  /**
  Derived classes register themselves like:
  configuredInstances().insert(blurType, configuredInst) Instead of such a call,
  they simply declare a static field of type ConfInstRegistrator (see below) who
  performs the mentioned operation within its constructor.
  */
  static ConfiguredInstances& configuredInstances() noexcept;

  /**
  Actual implementation of the blur algorithm

  @param toBlur is a single channel matrix with values of type double
  @param blurred the result of the blur (already initialized when method gets
  called)
  @param forTinySym demands generating a Gaussian blur with smaller window and
  standard deviation for tiny symbols
  */
  virtual void doProcess(const cv::Mat& toBlur,
                         cv::Mat& blurred,
                         bool forTinySym) const noexcept = 0;

  /**
  Derived classes from BlurEngine need to declare a static ConfInstRegistrator
  cir to self-register within BlurEngine::configuredInstances()
  */
  class ConfInstRegistrator {
   public:
    /// Provides the blur name and the instance to be registered within
    /// BlurEngine::configuredInstances()
    ConfInstRegistrator(const std::string& blurType,
                        const IBlurEngine& configuredInstance) noexcept;
  };
};

}  // namespace pic2sym::blur

#endif  // H_BLUR
