/****************************************************************************************
 The application Pic2Sym approximates images by a
 grid of colored symbols with colored backgrounds.

 This file was created on 2016-4-9
 and belongs to the Pic2Sym project.

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
#include "matchParams.h"
#include "misc.h"

#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

/*
Match aspect implementing the method described in https://ece.uwaterloo.ca/~z70wang/research/ssim .

Downsampling was not used, as the results normally get inspected by
enlarging the regions of interest.
*/
double StructuralSimilarity::assessMatch(const Mat &patch,
										 const SymData &symData,
										 MatchParams &mp) const {

	mp.computeSsim(patch, symData);

	// Poor structural similarity produces ssim close to -1.
	// Good structural similarity sets ssim towards 1.
	// The returned value is in 0..1 range,
	//		small for small ssim-s or large k (>1)
	//		larger for good ssim-s or 0 < k <= 1
	return pow((1. + mp.ssim.value()) / 2., k);
}

void MatchParams::computeBlurredPatch(const Mat &patch) {
	if(blurredPatch)
		return;

	Mat blurredPatch_;
	GaussianBlur(patch, blurredPatch_,
					 StructuralSimilarity::WIN_SIZE, StructuralSimilarity::SIGMA, 0.,
					 BORDER_REPLICATE);
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
	GaussianBlur(patch.mul(patch), variancePatch_,
					 StructuralSimilarity::WIN_SIZE, StructuralSimilarity::SIGMA, 0.,
					 BORDER_REPLICATE);
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

	// Saving 2 calls to GaussianBlur each time current symbol is compared to a patch:
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

	GaussianBlur(approxPatch, blurredPatchApprox_,
					 StructuralSimilarity::WIN_SIZE, StructuralSimilarity::SIGMA, 0.,
					 BORDER_REPLICATE);
	minMaxIdx(blurredPatchApprox - blurredPatchApprox_, &minVal, &maxVal); // math vs. brute-force
	assert(abs(minVal) < EPS);
	assert(abs(maxVal) < EPS);

	GaussianBlur(approxPatch.mul(approxPatch), variancePatchApprox_,
					 StructuralSimilarity::WIN_SIZE, StructuralSimilarity::SIGMA, 0.,
					 BORDER_REPLICATE);
	variancePatchApprox_ -= blurredPatchApproxSq;
	minMaxIdx(variancePatchApprox - variancePatchApprox_, &minVal, &maxVal); // math vs. brute-force
	assert(abs(minVal) < EPS);
	assert(abs(maxVal) < EPS);
#endif // checking the simplifications mentioned above

	const Mat productMats = patch.mul(approxPatch),
		productBlurredMats = blurredPatch.value().mul(blurredPatchApprox);
	GaussianBlur(productMats, covariance,
					 StructuralSimilarity::WIN_SIZE, StructuralSimilarity::SIGMA, 0.,
					 BORDER_REPLICATE);
	covariance -= productBlurredMats;

	const Mat numerator = (2.*productBlurredMats + StructuralSimilarity::C1).
		mul(2.*covariance + StructuralSimilarity::C2),
		denominator = (blurredPatchSq.value() + blurredPatchApproxSq + StructuralSimilarity::C1).
		mul(variancePatch.value() + variancePatchApprox + StructuralSimilarity::C2);

	divide(numerator, denominator, ssimMap);
	ssim = *mean(ssimMap).val;
	assert(abs(*ssim) < 1.+EPS);
}
