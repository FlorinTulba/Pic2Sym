/****************************************************************************************
 The application Pic2Sym approximates images by a
 grid of colored symbols with colored backgrounds.

 This file belongs to the Pic2Sym project.

 Copyrights from the libraries used by the program:
 - (c) 2015 Boost (www.boost.org)
   License: <http://www.boost.org/LICENSE_1_0.txt>
            or doc/licenses/Boost.lic
 - (c) 2015 The FreeType Project (www.freetype.org)
   License: <http://git.savannah.gnu.org/cgit/freetype/freetype2.git/plain/docs/FTL.TXT>
	        or doc/licenses/FTL.txt
 - (c) 2015 OpenCV (www.opencv.org)
   License: <http://opencv.org/license.html>
            or doc/licenses/OpenCV.lic
 - (c) 1997-2002 OpenMP Architecture Review Board (www.openmp.org)
   (c) Microsoft Corporation (Visual C++ implementation for OpenMP C/C++ Version 2.0 March 2002)
   See: <https://msdn.microsoft.com/en-us/library/8y6825x5(v=vs.140).aspx>
 
 (c) 2016 Florin Tulba <florintulba@yahoo.com>

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
 If not, see <http://www.gnu.org/licenses/agpl-3.0.txt>.
 ****************************************************************************************/

#include "match.h"
#include "structuralSimilarity.h"
#include "blur.h"
#include "matchParams.h"
#include "misc.h"

#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

REGISTERED_MATCH_ASPECT(StructuralSimilarity);

StructuralSimilarity::StructuralSimilarity(const CachedData &cachedData_, const MatchSettings &cfg) :
	MatchAspect(cachedData_, cfg.get_kSsim()) {}

/**
Poor structural similarity produces ssim close to -1.
Good structural similarity sets ssim towards 1.
The returned value is in 0..1 range,
		small for small ssim-s or large k (>1)
		larger for good ssim-s or 0 < k <= 1
*/
double StructuralSimilarity::score(const MatchParams &mp) const {
	return pow((1. + mp.ssim.value()) / 2., k);
}

/*
Match aspect implementing the method described in https://ece.uwaterloo.ca/~z70wang/research/ssim .

Downsampling was not used, as the results normally get inspected by
enlarging the regions of interest.
*/
void StructuralSimilarity::fillRequiredMatchParams(const Mat &patch,
												   const SymData &symData,
												   MatchParams &mp) const {
	mp.computeSsim(patch, symData);
}

double StructuralSimilarity::relativeComplexity() const {
	return 1000.; // extremely complex compared to the rest of the aspects
}

void MatchParams::computeBlurredPatch(const Mat &patch) {
	if(blurredPatch)
		return;

	Mat blurredPatch_;
	StructuralSimilarity::supportBlur.process(patch, blurredPatch_);
	blurredPatch = blurredPatch_;
}

void MatchParams::computeBlurredPatchSq(const Mat &patch) {
	if(blurredPatchSq)
		return;

	computeBlurredPatch(patch);
	blurredPatchSq = blurredPatch.value().mul(blurredPatch.value());
}

void MatchParams::computeVariancePatch(const Mat &patch) {
	if(variancePatch)
		return;

	computeBlurredPatchSq(patch);

	Mat variancePatch_;
	StructuralSimilarity::supportBlur.process(patch.mul(patch), variancePatch_);
	variancePatch_ -= blurredPatchSq.value();
	variancePatch = variancePatch_;
}

void MatchParams::computeSsim(const Mat &patch, const SymData &symData) {
	if(ssim)
		return;

	Mat covariance, ssimMap;

	computeVariancePatch(patch);
	computePatchApprox(patch, symData);
	const Mat &approxPatch = patchApprox.value();

	// Saving 2 calls to BlurEngine each time current symbol is compared to a patch:
	// Blur and Variance of the approximated patch are computed based on the blur and variance
	// of the grounded version of the original symbol
	const double diffRatio = contrast.value() / symData.diffMinMax;
	const Mat blurredPatchApprox = bg.value() + diffRatio *
										symData.symAndMasks[SymData::BLURRED_GR_SYM_IDX],
				blurredPatchApproxSq = blurredPatchApprox.mul(blurredPatchApprox),
				variancePatchApprox = diffRatio * diffRatio *
										symData.symAndMasks[SymData::VARIANCE_GR_SYM_IDX];

#ifdef _DEBUG // checking the simplifications mentioned above
	double minVal, maxVal;
	Mat blurredPatchApprox_, variancePatchApprox_; // computed by brute-force

	StructuralSimilarity::supportBlur.process(approxPatch, blurredPatchApprox_);
	minMaxIdx(blurredPatchApprox - blurredPatchApprox_, &minVal, &maxVal); // math vs. brute-force
	assert(abs(minVal) < EPS);
	assert(abs(maxVal) < EPS);

	StructuralSimilarity::supportBlur.process(approxPatch.mul(approxPatch), variancePatchApprox_);
	variancePatchApprox_ -= blurredPatchApproxSq;
	minMaxIdx(variancePatchApprox - variancePatchApprox_, &minVal, &maxVal); // math vs. brute-force
	assert(abs(minVal) < EPS);
	assert(abs(maxVal) < EPS);
#endif // checking the simplifications mentioned above

	const Mat productMats = patch.mul(approxPatch),
		productBlurredMats = blurredPatch.value().mul(blurredPatchApprox);
	StructuralSimilarity::supportBlur.process(productMats, covariance);
	covariance -= productBlurredMats;

	extern const double StructuralSimilarity_C1, StructuralSimilarity_C2;
	const Mat numerator = (2.*productBlurredMats + StructuralSimilarity_C1).
							mul(2.*covariance + StructuralSimilarity_C2),
			denominator = (blurredPatchSq.value() + blurredPatchApproxSq + StructuralSimilarity_C1).
							mul(variancePatch.value() + variancePatchApprox + StructuralSimilarity_C2);

	divide(numerator, denominator, ssimMap);
	ssim = *mean(ssimMap).val;
	assert(abs(*ssim) < 1.+EPS);
}
