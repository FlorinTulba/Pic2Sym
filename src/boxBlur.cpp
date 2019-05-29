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

#include "boxBlur.h"
#include "warnings.h"

#pragma warning(push, 0)

#include <opencv2/imgproc/imgproc.hpp>

#pragma warning(pop)

using namespace std;
using namespace cv;

// Handle class
class BoxBlur::Impl {
  friend class BoxBlur;

  static BoxBlur::Impl _tinySyms, _nonTinySyms;

  /// Default anchor point for a given kernel (its center)
  static const Point midPoint;

  /// First odd width of the box mask less than the ideal width
  unsigned wl = 0U;

  /// First odd width of the box mask greater than the ideal width
  unsigned wu = 0U;

  /// The number of times to iterate the filter of width wl
  unsigned countWl = 0U;

  /// The number of times to iterate the filter of width wu
  unsigned countWu = 0U;

  /// Desired number of iterations (countWl + countWu)
  unsigned iterations = 0U;

  constexpr Impl() noexcept {}

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
    if (0U == iterations_)
      THROW_WITH_CONST_MSG(__FUNCTION__ " needs iterations_ >= 1!",
                           invalid_argument);
    if (desiredSigma <= 0.)
      THROW_WITH_CONST_MSG(__FUNCTION__ " needs desiredSigma > 0!",
                           invalid_argument);

    iterations = iterations_;
    const double common = 12. * desiredSigma * desiredSigma,
                 idealBoxWidth = sqrt(1. + common / iterations_);
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
    if ((boxWidth_ & 1U) != 1U)
      THROW_WITH_CONST_MSG(__FUNCTION__ " boxWidth_ should be an odd value!",
                           invalid_argument);

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
    if (0U == iterations_)
      THROW_WITH_CONST_MSG(__FUNCTION__ " needs iterations_ >= 1!",
                           invalid_argument);

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
    if (countWl == 0U && countWu == 0U) {
      toBlur.copyTo(blurred);
      return;
    }

    bool applied = false;

    // The smaller mask (wl) can be a single-point mask (wl == 1), which can be
    // skipped, as it doesn't affect the result at all
    if (wl > 1U && countWl > 0U) {
      const Size boxL((int)wl, (int)wl);

      // 1st time with mask wl
      blur(toBlur, blurred, boxL, midPoint, BORDER_REPLICATE);

      // rest of the times with mask wu
      for (unsigned i = 1U; i < countWl; ++i)
        blur(blurred, blurred, boxL, midPoint, BORDER_REPLICATE);

      applied = true;
    }

    if (countWu > 0U) {
      const Size boxU((int)wu, (int)wu);

      // 1st time with mask wu
      if (!applied)
        blur(toBlur, blurred, boxU, midPoint, BORDER_REPLICATE);
      else
        blur(blurred, blurred, boxU, midPoint, BORDER_REPLICATE);

      // rest of the times with mask wu
      for (unsigned i = 1U; i < countWu; ++i)
        blur(blurred, blurred, boxU, midPoint, BORDER_REPLICATE);
    }
  }
};

BoxBlur::Impl BoxBlur::Impl::_nonTinySyms;
BoxBlur::Impl BoxBlur::Impl::_tinySyms;

const Point BoxBlur::Impl::midPoint(-1, -1);

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
  extern const double StructuralSimilarity_SIGMA;
  static BoxBlur result;
  static bool initialized = false;

  if (!initialized) {
    // Box blur with single iteration and desired standard deviation
    result.setSigma(StructuralSimilarity_SIGMA);
    initialized = true;
  }
  return result;
}
