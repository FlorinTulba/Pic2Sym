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

#include "sievesSymsFilter.h"

using namespace std;
using namespace cv;

namespace pic2sym {

SYM_FILTER_DEFINE_IS_ENABLED(SievesSymsFilter)

namespace syms::inline filter {

namespace {
/**
Compares the pixel value sums for each quadrant of the symbol with 2 thresholds
around an average obtained considering all 4 quadrants together.
If any of the sums is outside the allowed range, the symbol appears not evenly
distributed, so it cannot be a sieve.
*/
bool isEvenlyDistributed(const IPixMapSym& pms,
                         const SymFilterCache& sfc,
                         const Mat& brightGlyph,
                         bool toInvert,
                         unsigned sz,
                         unsigned halfSz,
                         unsigned lastSzBit) noexcept {
  // threshold ratio between 1/4 of the symbol sum and the sum of each quadrant
  static constexpr double SumQuarterThreshold{1.5364};

  double sumBrightGlyph{
      (toInvert ? (1. - pms.getAvgPixVal()) : pms.getAvgPixVal()) *
      sfc.getAreaD()};

  // Ignore central lines when sz is odd
  if (lastSzBit) {
    sumBrightGlyph -= *sum(brightGlyph.row((int)halfSz)).val +
                      *sum(brightGlyph.col((int)halfSz)).val;

    // add back the central pixel, as it was subtracted twice
    sumBrightGlyph += brightGlyph.at<double>((int)halfSz, (int)halfSz);
  }

  // central average value
  const double sumQuarterBrightGlyph{sumBrightGlyph / 4.};

  // min and max limits of the sum within each quadrant
  const double minSumQuarterBrightGlyph{sumQuarterBrightGlyph /
                                        SumQuarterThreshold};
  const double maxSumQuarterBrightGlyph{sumQuarterBrightGlyph *
                                        SumQuarterThreshold};

  // The 4 quadrants and their sums (ignoring mid rows & columns for odd sz)
  const Range firstHalf{0, (int)halfSz};
  const Range secondHalf{int(halfSz + lastSzBit), (int)sz};
  const Mat q1{brightGlyph, firstHalf, firstHalf};
  const Mat q2{brightGlyph, firstHalf, secondHalf};
  const Mat q3{brightGlyph, secondHalf, secondHalf};
  const Mat q4{brightGlyph, secondHalf, firstHalf};
  const double q1Sum{*sum(sum(q1)).val};
  const double q2Sum{*sum(sum(q2)).val};
  const double q3Sum{*sum(sum(q3)).val};
  const double q4Sum{*sum(sum(q4)).val};
  const Mat qSums = (Mat_<double>(1, 4) << q1Sum, q2Sum, q3Sum, q4Sum);

  // min and max from qSums
  double minQsum, maxQsum;
  minMaxIdx(qSums, &minQsum, &maxQsum);

  // The 4 quadrants must contain approx sumQuarterBrightGlyph, as sieves are
  // quite evenly distributed
  return (minQsum > minSumQuarterBrightGlyph &&
          maxQsum < maxSumQuarterBrightGlyph);
}

/**
INSPECT_FFT_MAGNITUDE_SPECTRUM can be used in Debug mode to view the magnitude
spectrum from a 2D FFT transform in natural order. A breakpoint should be set on
a line after the shifting of the spectrum was performed and the spectrum can be
inspected as a matrix.
*/
//#define INSPECT_FFT_MAGNITUDE_SPECTRUM
#if defined(_DEBUG) && defined(INSPECT_FFT_MAGNITUDE_SPECTRUM)
/**
Rearranging the quadrants of the magnitude spectrum from the 3412 order to their
natural position. The method is for debugging, to get a clearer perspective on
the magnitude spectrum.
*/
Mat fftShift(const Mat& rawMagnSpectrum,
             unsigned sz,
             unsigned halfSz,
             unsigned lastSzBit) noexcept {
  // 0. parameter prevents using initializer_list ctor of Mat
  Mat result{(int)sz, (int)sz, CV_64FC1, 0.};
  const unsigned mid1{halfSz + lastSzBit};
  const unsigned mid2{sz - mid1};
  const Range range1{0, (int)mid1};
  const Range range2{(int)mid1, (int)sz};  // ranges within the raw spectrum
  const Range range1_{0, (int)mid2};

  // ranges within the rearranged spectrum
  const Range range2_{(int)mid2, (int)sz};

  // quadrants of the raw spectrum
  const Mat q1raw{rawMagnSpectrum, range1, range1};
  const Mat q3raw{rawMagnSpectrum, range2, range2};
  const Mat q4raw{rawMagnSpectrum, range2, range1};
  const Mat q2raw{rawMagnSpectrum, range1, range2};

  // quadrants of the rearranged spectrum
  Mat q1{result, range1_, range1_};
  Mat q2{result, range1_, range2_};
  Mat q3{result, range2_, range2_};
  Mat q4{result, range2_, range1_};

  q1raw.copyTo(q3);
  q3raw.copyTo(q1);
  q4raw.copyTo(q2);
  q2raw.copyTo(q4);

  return result;
}
#endif  // _DEBUG, INSPECT_FFT_MAGNITUDE_SPECTRUM

/// Computes the magnitude of the DFT spectrum in raw quadrants order 3412.
Mat magnitudeSpectrum(const Mat& brightGlyph, unsigned sz) noexcept {
  // Creating the complex input for DFT - adding zeros for the imaginary parts
  const Mat cplxGlyphPlanes[] = {brightGlyph,
                                 Mat::zeros((int)sz, (int)sz, CV_64FC1)};
  Mat cplxGlyph, dftGlyph, result, dftGlyphPlanes[2ULL];
  merge(cplxGlyphPlanes, 2ULL, cplxGlyph);

  /*
  Computing DFT for the exact data - no windowing, nor 0-padding,
  since the symbols are quite small and the overhead of windowing / padding
  might just:
  - incur a higher time-penalty
  - reduce the dominant magnitude
  - provide several higher-frequency FT modes

  Current approach has already good accuracy.

  However, there are many glyphs whose borders don't mirror each other, so
  generally borders can be ignored when detecting sieves.

  Therefore, for even better accuracy, DFT could be applied to a square composed
  of tiles containing the analyzed symbol with its borders gradually converging
  to gray.
  */
  dft(cplxGlyph, dftGlyph, DFT_COMPLEX_OUTPUT);
  // normal output, not CCS(complex-conjugate-symmetrical)

  // Computing magnitude of the spectrum from real and imaginary planes of the
  // DFT
  split(dftGlyph, dftGlyphPlanes);
  magnitude(dftGlyphPlanes[0ULL], dftGlyphPlanes[1ULL], result);

  return result;
}

/**
Analyzes the magnitude spectrum rawMagnSpectrum to extract its peaks for
quadrants 1&3 and 2&4. The peaks are represented by the pairs (hFreq13, vFreq13)
and (hFreq24, vFreq24) and must be above a certain threshold to qualify as
peaks.

*/
void extractDominantFtModes(const Mat& rawMagnSpectrum,
                            unsigned sz,
                            unsigned halfSz,
                            unsigned lastSzBit,
                            unsigned& hFreq13,
                            unsigned& vFreq13,
                            unsigned& hFreq24,
                            unsigned& vFreq24) noexcept {
#if defined(_DEBUG) && defined(INSPECT_FFT_MAGNITUDE_SPECTRUM)
  // Useful while Debugging, to visualize the spectrum quadrants in natural 1234
  // order
  Mat shiftedMagnSpectrum{fftShift(rawMagnSpectrum, sz, halfSz, lastSzBit)};

  // DC value = 0 to maximize the contrast for rest
  shiftedMagnSpectrum.at<double>((int)halfSz, (int)halfSz) = 0;
#endif  // _DEBUG, INSPECT_FFT_MAGNITUDE_SPECTRUM

  /* Building q13 and q24 as representatives for quadrants 1&3 and 2&4 */

  // Used ranges
  const int mid{int(halfSz + lastSzBit)};
  const Range range1{0, mid};
  const Range range2{mid, (int)sz};
  const Range range3{1, mid};
  const Range range4{0, mid - 1};

  // quadrant 3 as representative for 1&3
  const Mat q13{rawMagnSpectrum, range2, range2};

  // quadrant 4 without DC column from its left
  const Mat q4raw{rawMagnSpectrum, range2, range3};

  // representative for quadrants 2&4; 0. parameter avoids initializer_list ctor
  Mat q24{(int)halfSz, (int)halfSz, CV_64FC1, 0.};

  // where to copy quadrant 4 within q24
  Mat q4Dest{q24, Range::all(), range4};
  q4raw.copyTo(q4Dest);

  // For even sz, there is additional information for quadrant 2&4
  // to be extracted from 1st column of quadrant 2
  if (!lastSzBit) {
    const Range range5{mid, mid + 1};
    const Range range6{(int)halfSz - 1, (int)halfSz};
    Mat q2rawRest{rawMagnSpectrum, range3, range5};
    const Mat q2restDest{q24, range3, range6};
    flip(q2rawRest, q2restDest, 0);  // copy flipped horizontally
  }

  /* Extracting the peaks */

  // Sieves require FT modes magnitudes above MagnitudePercentThreshold of their
  // range
  static constexpr double MagnitudePercentThreshold{.16};
  Mat maskDC{(int)sz, (int)sz, CV_8UC1, 255.};
  maskDC.at<unsigned char>(0, 0) = 0U;  // mask to ignore DC
  double minV, maxV;                    // limits
  minMaxIdx(rawMagnSpectrum, &minV, &maxV, nullptr, nullptr,
            maskDC);  // min & max ignoring DC
  // Consider only FT modes whose magnitude is larger than the threshold below
  const double thresholdMagn{minV + (maxV - minV) * MagnitudePercentThreshold};

  // coordinates of the max magnitude FT mode within quadrant
  array<int, 2ULL> maxIdx;

  // Peak for quadrant 1&3
  minMaxIdx(q13, nullptr, &maxV, nullptr, maxIdx.data());
  if (maxV < thresholdMagn) {
    hFreq13 = vFreq13 = 0U;  // flag value
  } else {
    // maxIdx coordinates need to be reflected from halfSz,
    // since the frequencies should counted from the bottom-right corner of the
    // q13
    vFreq13 = halfSz - (unsigned)maxIdx[0ULL];
    hFreq13 = halfSz - (unsigned)maxIdx[1ULL];
  }

  // Peak for quadrant 2&4
  minMaxIdx(q24, nullptr, &maxV, nullptr, maxIdx.data());
  if (maxV < thresholdMagn) {
    hFreq24 = vFreq24 = 0U;  // flag value
  } else {
    // maxIdx[0] (vertical coordinate) needs to be reflected from halfSz,
    // since the frequencies should counted from the bottom-left corner of the
    // q24
    vFreq24 = halfSz - (unsigned)maxIdx[0ULL];

    // maxIdx[1] (horizontal coordinate) needs to be incremented, to comply with
    // the range: 1 .. halfSz    instead of the obtained     0 .. halfSz-1
    hFreq24 = (unsigned)maxIdx[1ULL] + 1U;
  }

  // For a sieve of even size and dominant FT mode of maximum frequencies in
  // quadrant 1&3, quadrant 2&4 won't be able to store a similar dominant FT
  // mode
  if (!lastSzBit && !hFreq24 && !vFreq24 && hFreq13 + vFreq13 == sz) {
    hFreq24 = vFreq24 = halfSz;
  }
}
}  // anonymous namespace

SievesSymsFilter::SievesSymsFilter(
    unique_ptr<ISymFilter> nextFilter_ /* = nullptr*/) noexcept
    : TSymFilter{4U, "sieve-like symbols", move(nextFilter_)} {}

bool SievesSymsFilter::isDisposable(const IPixMapSym& pms,
                                    const SymFilterCache& sfc) noexcept {
  if (!isEnabled())
    return false;

  const unsigned sz{sfc.getSzU()};    // symbol size
  const unsigned halfSz{sz >> 1U};    // floor(sz/2.)
  const unsigned lastSzBit{sz & 1U};  // 1 for odd sz, 0 for even sz

  // Using inverted glyph if original is too dark
  const bool toInvert{pms.getAvgPixVal() < .5};
  Mat brightGlyph{pms.toMatD01(sz)};
  if (toInvert)
    brightGlyph = 1. - brightGlyph;

  // Glyphs that aren't distributed evenly cannot be sieves => skip them
  if (!isEvenlyDistributed(pms, sfc, brightGlyph, toInvert, sz, halfSz,
                           lastSzBit))
    return false;

  const Mat magnSpectrum{magnitudeSpectrum(brightGlyph, sz)};

  // Dominant FT modes within quadrants 1&3 and 2&4
  unsigned hFreq13, vFreq13, hFreq24, vFreq24;
  extractDominantFtModes(magnSpectrum, sz, halfSz, lastSzBit, hFreq13, vFreq13,
                         hFreq24, vFreq24);

  // All dominant FT modes must be strictly positive (0 means DC value)
  if (!min(hFreq13, vFreq13) || !min(hFreq24, vFreq24))
    return false;

  // Dominant FT modes from quadrants 1&3 and 2&4 must be at most 1 unit apart
  // (L1 norm)
  if (abs(hFreq13 - hFreq24) + abs(vFreq13 - vFreq24) > 1U)
    return false;  // the symbol isn't symmetric enough

  // Minimum number of holes within the sieve (product of count per rows and
  // count per columns)
  static constexpr unsigned MinHolesCount{9U};

  // Vertical and horizontal frequencies need to be >1 and the count of holes of
  // the sieves must be >=9
  const unsigned hFreqMax{max(hFreq13, hFreq24)};
  const unsigned vFreqMax{max(vFreq13, vFreq24)};
  if (hFreqMax < 2U || vFreqMax < 2U || hFreqMax * vFreqMax < MinHolesCount)
    return false;

  return true;
}

}  // namespace syms::inline filter
}  // namespace pic2sym
