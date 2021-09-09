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

#include "structuralSimilarity.h"

#include "blurBase.h"
#include "cachedData.h"
#include "match.h"
#include "matchParams.h"

using namespace std;
using namespace cv;

namespace pic2sym {

#ifdef _DEBUG
extern const string StructuralSimilarity_BlurType;
#endif  // _DEBUG
extern const double StructuralSimilarity_C1;
extern const double StructuralSimilarity_C2;

using syms::ISymData;
using transform::CachedData;

namespace match {

StructuralSimilarity::StructuralSimilarity(
    const p2s::cfg::IMatchSettings& ms) noexcept
    : MatchAspect{ms.get_kSsim()} {}

/**
Poor structural similarity produces ssim close to -1.
Good structural similarity sets ssim towards 1.
The returned value is in 0..1 range,
    small for small ssim-s or large k (>1)
    larger for good ssim-s or 0 < k <= 1
*/
double StructuralSimilarity::score(const IMatchParams& mp,
                                   const CachedData&) const noexcept {
  return pow((1. + mp.getSsim().value()) / 2., *k);
}

/*
Match aspect implementing the method described in
https://ece.uwaterloo.ca/~z70wang/research/ssim .

Downsampling was not used, as the results normally get inspected by
enlarging the regions of interest.
*/
void StructuralSimilarity::fillRequiredMatchParams(
    const Mat& patch,
    const ISymData& symData,
    const CachedData& cachedData,
    IMatchParamsRW& mp) const noexcept {
  mp.computeSsim(patch, symData, cachedData);
}

double StructuralSimilarity::relativeComplexity() const noexcept {
  return 1'000.;  // extremely complex compared to the rest of the aspects
}

void MatchParams::computeBlurredPatch(const Mat& patch,
                                      const CachedData& cachedData) noexcept {
  if (blurredPatch)
    return;

  Mat blurredPatch_;
  StructuralSimilarity::supportBlur.process(patch, blurredPatch_,
                                            cachedData.forTinySyms);
  blurredPatch = blurredPatch_;
}

void MatchParams::computeBlurredPatchSq(const Mat& patch,
                                        const CachedData& cachedData) noexcept {
  if (blurredPatchSq)
    return;

  computeBlurredPatch(patch, cachedData);
  blurredPatchSq = blurredPatch.value().mul(blurredPatch.value());
}

void MatchParams::computeVariancePatch(const Mat& patch,
                                       const CachedData& cachedData) noexcept {
  if (variancePatch)
    return;

  computeBlurredPatchSq(patch, cachedData);
  computePatchSq(patch);

  Mat variancePatch_;
  StructuralSimilarity::supportBlur.process(patchSq.value(), variancePatch_,
                                            cachedData.forTinySyms);
  variancePatch_ -= blurredPatchSq.value();
  variancePatch = variancePatch_;
}

void MatchParams::computeSsim(const Mat& patch,
                              const ISymData& symData,
                              const CachedData& cachedData) noexcept {
  if (ssim)
    return;

#ifdef _DEBUG
  static const bool usingGaussianBlur{StructuralSimilarity_BlurType ==
                                      "gaussian"};
#endif  // _DEBUG

  Mat covariance, ssimMap;

  computeVariancePatch(patch, cachedData);
  computePatchApprox(patch, symData);
  const Mat& approxPatch = patchApprox.value();

  // Saving 2 calls to BlurEngine each time current symbol is compared to a
  // patch: Blur and Variance of the approximated patch are computed based on
  // the blur and variance of the grounded version of the original symbol
  const double diffRatio{contrast.value() / symData.getDiffMinMax()};
  const Mat blurredPatchApprox{
      bg.value() +
      diffRatio * symData.getMask(ISymData::MaskType::BlurredGrSym)};
  const Mat blurredPatchApproxSq{blurredPatchApprox.mul(blurredPatchApprox)};
  const Mat variancePatchApprox{
      diffRatio * diffRatio *
      symData.getMask(ISymData::MaskType::VarianceGrSym)};

#ifdef _DEBUG  // checking the simplifications mentioned above
  // Since the other blur algorithms have lower quality, it is difficult to set
  // an error threshold that is also valid for them That's why the
  // simplifications are checked only for the Gaussian blur
  if (usingGaussianBlur) {
    double minVal, maxVal;
    Mat blurredPatchApprox_, variancePatchApprox_;  // computed by brute-force

    StructuralSimilarity::supportBlur.process(approxPatch, blurredPatchApprox_,
                                              cachedData.forTinySyms);
    minMaxIdx(blurredPatchApprox - blurredPatchApprox_, &minVal,
              &maxVal);  // math vs. brute-force
    assert(abs(minVal) < Eps);
    assert(abs(maxVal) < Eps);

    StructuralSimilarity::supportBlur.process(approxPatch.mul(approxPatch),
                                              variancePatchApprox_,
                                              cachedData.forTinySyms);
    variancePatchApprox_ -= blurredPatchApproxSq;
    minMaxIdx(variancePatchApprox - variancePatchApprox_, &minVal,
              &maxVal);  // math vs. brute-force
    assert(abs(minVal) < Eps);
    assert(abs(maxVal) < Eps);
  }
#endif  // checking the simplifications mentioned above

  const Mat productMats{patch.mul(approxPatch)};
  const Mat productBlurredMats{blurredPatch.value().mul(blurredPatchApprox)};
  StructuralSimilarity::supportBlur.process(productMats, covariance,
                                            cachedData.forTinySyms);
  covariance -= productBlurredMats;

  const Mat numerator{(2. * productBlurredMats + StructuralSimilarity_C1)
                          .mul(2. * covariance + StructuralSimilarity_C2)};
  const Mat denominator{
      (blurredPatchSq.value() + blurredPatchApproxSq + StructuralSimilarity_C1)
          .mul(variancePatch.value() + variancePatchApprox +
               StructuralSimilarity_C2)};

  divide(numerator, denominator, ssimMap);
  ssim = *mean(ssimMap).val;
#ifdef _DEBUG  // checking that ssim is in -1..1 range
  // Since the other blur algorithms have lower quality, it is difficult to set
  // an error threshold that is also valid for them That's why the range check
  // is performed only for the Gaussian blur
  if (usingGaussianBlur)
    assert(abs(*ssim) < EpsPlus1);
#endif  // _DEBUG
}

}  // namespace match
}  // namespace pic2sym
