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

#include "boxBlur.h"

#include "warnings.h"

#pragma warning(push, 0)

#include <opencv2/imgproc/imgproc.hpp>

#include <gsl/gsl>

#pragma warning(pop)

using namespace std;
using namespace cv;

namespace pic2sym {

extern const double StructuralSimilarity_SIGMA;

namespace blur {

// Handle class
class BoxBlur::Impl {
  friend class BoxBlur;

  // Inline static initialization complains about undefined BoxBlur::Impl type
  static BoxBlur::Impl _tinySyms;
  static BoxBlur::Impl _nonTinySyms;

  /// Default anchor point for a given kernel (its center)
  static inline const Point midPoint{-1, -1};

#pragma warning(disable : WARN_THROWS_ALTHOUGH_NOEXCEPT)
  /**
  Reconfigure the filter through a new desired standard deviation and a new
  iterations count See
  http://www.web.uwa.edu.au/__data/assets/file/0008/826172/filterdesign.pdf
  for details

  @throw invalid_argument for desiredSigma<=0 or iterations_==0

  Exception only to be reported, not handled.
  */
  Impl& setSigma(double desiredSigma, unsigned iterations_ = 1U) noexcept(!UT) {
    EXPECTS_OR_REPORT_AND_THROW_CONST_MSG(
        0U < iterations_, invalid_argument,
        HERE.function_name() + " needs iterations_ >= 1!"s);
    EXPECTS_OR_REPORT_AND_THROW_CONST_MSG(
        desiredSigma > 0., invalid_argument,
        HERE.function_name() + " needs desiredSigma > 0!"s);

    iterations = iterations_;
    const double common{12. * desiredSigma * desiredSigma};
    const double idealBoxWidth{sqrt(1. + common / iterations_)};
    wl = (((unsigned((idealBoxWidth - 1.) / 2.)) << 1) | 1U);
    wu = wl + 2U;  // next odd value

    countWl = (unsigned)max(
        0, int(round((common - iterations_ * (3. + wl * (wl + 4.))) /
                     (-4. * (wl + 1.)))));
    countWu = iterations_ - countWl;

    return *this;
  }
#pragma warning(default : WARN_THROWS_ALTHOUGH_NOEXCEPT)

#pragma warning(disable : WARN_THROWS_ALTHOUGH_NOEXCEPT)
  /**
  Reconfigure mask width (wl) for performing all iterations and destroys the
  wu mask.

  @throw invalid_argument for even boxWidth_

  Exception only to be reported, not handled.
  */
  Impl& setWidth(unsigned boxWidth_) noexcept(!UT) {
    EXPECTS_OR_REPORT_AND_THROW_CONST_MSG(
        (boxWidth_ & 1U) == 1U, invalid_argument,
        HERE.function_name() + " boxWidth_ should be an odd value!"s);

    wl = boxWidth_;
    countWu = 0U;  // cancels any iterations for filters with width wu
    countWl = iterations;

    return *this;
  }
#pragma warning(default : WARN_THROWS_ALTHOUGH_NOEXCEPT)

#pragma warning(disable : WARN_THROWS_ALTHOUGH_NOEXCEPT)
  /**
  Reconfigure iterations count for wl and destroys the wu mask

  @throw invalid_argument for iterations_==0

  Exception only to be reported, not handled.
  */
  Impl& setIterations(unsigned iterations_) noexcept(!UT) {
    EXPECTS_OR_REPORT_AND_THROW_CONST_MSG(
        0U < iterations_, invalid_argument,
        HERE.function_name() + " needs iterations_ >= 1!"s);

    iterations = countWl = iterations_;
    countWu = 0U;  // cancels any iterations for filters with width wu

    return *this;
  }
#pragma warning(default : WARN_THROWS_ALTHOUGH_NOEXCEPT)

  /// Actual implementation for the current configuration. toBlur is checked;
  /// blurred is initialized See
  /// http://www.web.uwa.edu.au/__data/assets/file/0008/826172/filterdesign.pdf
  /// for details
  void apply(const cv::Mat& toBlur, cv::Mat& blurred) noexcept {
    if (!countWl && !countWu) {
      toBlur.copyTo(blurred);
      return;
    }

    bool applied{false};

    // The smaller mask (wl) can be a single-point mask (wl == 1), which can be
    // skipped, as it doesn't affect the result at all
    if (wl > 1U && countWl > 0U) {
      const Size boxL{(int)wl, (int)wl};

      // 1st time with mask wl
      cv::blur(toBlur, blurred, boxL, midPoint, BORDER_REPLICATE);

      // rest of the times with mask wu
      for (unsigned i{1U}; i < countWl; ++i)
        cv::blur(blurred, blurred, boxL, midPoint, BORDER_REPLICATE);

      applied = true;
    }

    if (countWu > 0U) {
      const Size boxU{(int)wu, (int)wu};

      // 1st time with mask wu
      if (!applied)
        cv::blur(toBlur, blurred, boxU, midPoint, BORDER_REPLICATE);
      else
        cv::blur(blurred, blurred, boxU, midPoint, BORDER_REPLICATE);

      // rest of the times with mask wu
      for (unsigned i{1U}; i < countWu; ++i)
        cv::blur(blurred, blurred, boxU, midPoint, BORDER_REPLICATE);
    }
  }

  /// First odd width of the box mask less than the ideal width
  unsigned wl{};

  /// First odd width of the box mask greater than the ideal width
  unsigned wu{};

  /// The number of times to iterate the filter of width wl
  unsigned countWl{};

  /// The number of times to iterate the filter of width wu
  unsigned countWu{};

  /// Desired number of iterations (countWl + countWu)
  unsigned iterations{};
};

BoxBlur::Impl BoxBlur::Impl::_tinySyms;
BoxBlur::Impl BoxBlur::Impl::_nonTinySyms;

BoxBlur::Impl& BoxBlur::nonTinySyms() noexcept {
  return BoxBlur::Impl::_nonTinySyms;
}

BoxBlur::Impl& BoxBlur::tinySyms() noexcept {
  return BoxBlur::Impl::_tinySyms;
}

BoxBlur::BoxBlur(unsigned boxWidth_ /* = 1U*/,
                 unsigned iterations_ /* = 1U*/) noexcept(!UT) {
  setWidth(boxWidth_).setIterations(iterations_);
}

BoxBlur& BoxBlur::setSigma(double desiredSigma,
                           unsigned iterations_ /* = 1U*/) noexcept(!UT) {
  nonTinySyms().setSigma(desiredSigma, iterations_);

  // Tiny symbols should use a sigma = desiredSigma/2.
  tinySyms().setSigma(desiredSigma * .5, iterations_);

  return *this;
}

BoxBlur& BoxBlur::setWidth(unsigned boxWidth_) noexcept(!UT) {
  nonTinySyms().setWidth(boxWidth_);

  // Tiny symbols should use a box whose width is next odd value >= boxWidth_/2.
  tinySyms().setWidth((boxWidth_ >> 1) | 1U);

  return *this;
}

BoxBlur& BoxBlur::setIterations(unsigned iterations_) noexcept(!UT) {
  nonTinySyms().setIterations(iterations_);
  tinySyms().setIterations(iterations_);

  return *this;
}

void BoxBlur::doProcess(const cv::Mat& toBlur,
                        cv::Mat& blurred,
                        bool forTinySym) const noexcept {
  if (forTinySym)
    tinySyms().apply(toBlur, blurred);
  else
    nonTinySyms().apply(toBlur, blurred);
}

const BoxBlur& BoxBlur::configuredInstance() noexcept(!UT) {
  static BoxBlur result;
  static bool initialized{false};

  if (!initialized) {
    // Box blur with single iteration and desired standard deviation
    result.setSigma(StructuralSimilarity_SIGMA);
    initialized = true;
  }
  return result;
}

}  // namespace blur
}  // namespace pic2sym
