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

#include "extBoxBlur.h"
#include "warnings.h"

#pragma warning(push, 0)

#include <numeric>

#include <opencv2/imgproc/imgproc.hpp>

#pragma warning(pop)

using namespace std;
using namespace cv;

// Handle class
class ExtBoxBlur::Impl {
  friend class ExtBoxBlur;

  static ExtBoxBlur::Impl _nonTinySyms, _tinySyms;
  double sigma = 0.;    ///< desired standard deviation
  unsigned times = 0U;  ///< iterations count

  // Following fields all are derived from sigma and times. See
  // extendedBoxKernel
  unsigned kernelWidth = 0U;  ///< width of the mask (extended box)

  /// Ceiling of the box (majority of extended box's values)
  double boxHeight = 0.;

  double w = 0.;   ///< values on the extension edges of the mask
  double w1 = 0.;  ///< boxHeight - w

  constexpr Impl() noexcept {}

#pragma warning(disable : WARN_THROWS_ALTHOUGH_NOEXCEPT)
  /**
  Reconfigures the kernel
  See http://www.mia.uni-saarland.de/Publications/gwosdek-ssvm11.pdf for
  details

  @throw invalid_argument if times_==0 or sigma_<=0

  Exception to be only reported, not handled
  */
  void extendedBoxKernel(double sigma_, unsigned times_) noexcept(!UT) {
    if (0U == times_)
      THROW_WITH_CONST_MSG(__FUNCTION__ " needs times_ >= 1!",
                           invalid_argument);
    if (sigma_ <= 0.)
      THROW_WITH_CONST_MSG(__FUNCTION__ " needs sigma_ > 0!", invalid_argument);

    sigma = sigma_;
    times = times_;

    const double sigmaSq = sigma_ * sigma_, sigmaSqOverD = sigmaSq / times_,
                 idealBoxSize = sqrt(1. + 12. * sigmaSqOverD);
    const int l = int((idealBoxSize - 1.) / 2.), L = (l << 1) | 1;
    const double lp1 = l + 1.,
                 alpha = L * (l * lp1 - 3. * sigmaSqOverD) /
                         (6. * (sigmaSqOverD - lp1 * lp1)),
                 lambda = L + alpha + alpha;

    kernelWidth = unsigned(L + 2);
    boxHeight = 1. / lambda;
    w = alpha / lambda;
    w1 = boxHeight - w;
  }
#pragma warning(default : WARN_THROWS_ALTHOUGH_NOEXCEPT)

  /// Reconfigure the filter through a new desired standard deviation and a new
  /// iterations count
  Impl& setSigma(double desiredSigma, unsigned iterations_ = 1U) noexcept(!UT) {
    extendedBoxKernel(desiredSigma, iterations_);

    return *this;
  }

  /// Reconfigure iterations count for wl and destroys the wu mask
  Impl& setIterations(unsigned iterations_) noexcept(!UT) {
    extendedBoxKernel(sigma, iterations_);

    return *this;
  }

  /// Actual implementation for the current configuration. toBlur is checked;
  /// blurred is initialized See
  /// http://www.mia.uni-saarland.de/Publications/gwosdek-ssvm11.pdf for details
  void apply(const cv::Mat& toBlur, cv::Mat& blurred) noexcept {
    const int origWidth = toBlur.cols, origHeight = toBlur.rows,
              kernelRadius = (int)kernelWidth >> 1,
              kernelRadiusP1 = kernelRadius + 1;

    // Apply the number of times requested on the transposed original
    // Temp needs to be processed; blurred holds the outcome
    Mat temp;
    transpose(toBlur, temp);
    blurred = Mat(origWidth, origHeight, CV_64FC1);
    int dataRows = temp.rows, dataCols = temp.cols, dataRowsM1 = dataRows - 1,
        dataColsM1 = dataCols - 1;

    // Lambda to be used for both traversal directions
    const auto applyKernelTimesHorizontally = [&]() noexcept {
      for (unsigned iteration = 0U; iteration < times; ++iteration) {
        // Unfortunately, the CPU's are already busy in parallel parts calling
        // this blur method, so OMP won't make this faster, given current
        // context: #pragma omp parallel #pragma omp for schedule(static, 1)
        // nowait
        for (int row = 0; row < dataRows; ++row) {
          // Computations for 1st pixel on current row
          double* resultIt = blurred.ptr<double>(row);
          const double* dataItBegin = temp.ptr<double>(row);
          const double* frontEdge = dataItBegin + kernelRadiusP1;
          const double firstPixel = *dataItBegin,
                       lastPixel = dataItBegin[(size_t)dataColsM1];
          double prevFrontEdgePixel = dataItBegin[(size_t)kernelRadius],
                 frontEdgePixel = 0.,
                 prevSum = *resultIt =
                     w * (firstPixel + prevFrontEdgePixel) +
                     boxHeight * (firstPixel * kernelRadius +
                                  accumulate(dataItBegin + 1,
                                             dataItBegin + kernelRadius, 0.));

          // Next few pixels use all first pixel from temp as back edge
          // correction
          int col = 1;
          for (; col <= kernelRadius;
               prevFrontEdgePixel = frontEdgePixel, ++frontEdge, ++col) {
            frontEdgePixel = *frontEdge;
            prevSum += w * (frontEdgePixel - firstPixel) +
                       w1 * (prevFrontEdgePixel - firstPixel);
            *++resultIt = prevSum;
          }

          // Most pixels have both front & back edges for correction
          const double* backEdge = dataItBegin + 1;
          double backEdgePixel = 0., prevBackEdgePixel = firstPixel;
          for (; col < dataCols - kernelRadius;
               prevFrontEdgePixel = frontEdgePixel, ++frontEdge,
               prevBackEdgePixel = backEdgePixel, ++backEdge, ++col) {
            frontEdgePixel = *frontEdge;
            backEdgePixel = *backEdge;
            prevSum += w * (frontEdgePixel - prevBackEdgePixel) +
                       w1 * (prevFrontEdgePixel - backEdgePixel);
            *++resultIt = prevSum;
          }

          // Last pixels use last col from temp as front edge correction
          for (; col < dataCols;
               prevBackEdgePixel = backEdgePixel, ++backEdge, ++col) {
            backEdgePixel = *backEdge;
            prevSum += w * (lastPixel - prevBackEdgePixel) +
                       w1 * (lastPixel - backEdgePixel);
            *++resultIt = prevSum;
          }
        }

        swap(temp, blurred);
      }
    };

    applyKernelTimesHorizontally();

    // Apply the number of times requested on the temporary blurred transposed
    // back Temp needs to be processed; blurred holds the outcome
    transpose(temp, temp);
    if (blurred.isContinuous())
      blurred = blurred.reshape(1, origHeight);
    else
      blurred = Mat(toBlur.size(), toBlur.type());
    dataRows = temp.rows;
    dataCols = temp.cols;
    dataRowsM1 = dataRows - 1;
    dataColsM1 = dataCols - 1;
    applyKernelTimesHorizontally();

    blurred = temp;
  }
};

ExtBoxBlur::Impl ExtBoxBlur::Impl::_nonTinySyms;
ExtBoxBlur::Impl ExtBoxBlur::Impl::_tinySyms;

ExtBoxBlur::Impl& ExtBoxBlur::nonTinySyms() noexcept {
  return ExtBoxBlur::Impl::_nonTinySyms;
}

ExtBoxBlur::Impl& ExtBoxBlur::tinySyms() noexcept {
  return ExtBoxBlur::Impl::_tinySyms;
}

ExtBoxBlur::ExtBoxBlur(double desiredSigma,
                       unsigned iterations_ /* = 1U*/) noexcept(!UT) {
  setSigma(desiredSigma, iterations_);
}

ExtBoxBlur& ExtBoxBlur::setSigma(double desiredSigma,
                                 unsigned iterations_ /* = 1U*/) noexcept(!UT) {
  nonTinySyms().setSigma(desiredSigma, iterations_);

  // Tiny symbols should use a sigma = desiredSigma/2.
  tinySyms().setSigma(desiredSigma * .5, iterations_);

  return *this;
}

ExtBoxBlur& ExtBoxBlur::setIterations(unsigned iterations_) noexcept(!UT) {
  nonTinySyms().setIterations(iterations_);
  tinySyms().setIterations(iterations_);

  return *this;
}

void ExtBoxBlur::doProcess(const cv::Mat& toBlur,
                           cv::Mat& blurred,
                           bool forTinySym) const noexcept {
  if (forTinySym)
    tinySyms().apply(toBlur, blurred);
  else
    nonTinySyms().apply(toBlur, blurred);
}

const ExtBoxBlur& ExtBoxBlur::configuredInstance() noexcept(!UT) {
  // Extended Box blur with no iterations and desired standard deviation
  extern const double StructuralSimilarity_SIGMA;
  static ExtBoxBlur result(StructuralSimilarity_SIGMA);
  return result;
}
