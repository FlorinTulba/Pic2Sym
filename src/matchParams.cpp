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

#include "matchParams.h"
#include "matchSettings.h"
#include "symData.h"
#include "cachedData.h"
#include "patch.h"
#include "misc.h"

using namespace std;
using namespace boost;
using namespace cv;

void MatchParams::reset(bool skipPatchInvariantParts/* = true*/) {
	mcPatchApprox = none;
	patchApprox = none;
	ssim = fg = bg = contrast = sdevFg = sdevBg = sdevEdge = symDensity = mcsOffset = none;

	if(!skipPatchInvariantParts) {
		mcPatch = none;
		blurredPatch = blurredPatchSq = variancePatch = none;
	}
}

void MatchParams::computeMean(const Mat &patch, const Mat &mask, optional<double> &miu) {
	if(miu)
		return;

	miu = *mean(patch, mask).val;
	assert(*miu > -EPS && *miu < 255.+EPS);
}

void MatchParams::computeFg(const Mat &patch, const SymData &symData) {
	computeMean(patch, symData.symAndMasks[SymData::FG_MASK_IDX], fg);
}

void MatchParams::computeBg(const Mat &patch, const SymData &symData) {
	computeMean(patch, symData.symAndMasks[SymData::BG_MASK_IDX], bg);
}

void MatchParams::computeContrast(const Mat &patch, const SymData &symData) {
	if(contrast)
		return;

	computeFg(patch, symData);
	computeBg(patch, symData);

	contrast = fg.value() - bg.value();
	assert(abs(contrast.value()) < 255.5);
}

void MatchParams::computeSdev(const Mat &patch, const Mat &mask,
							  optional<double> &miu, optional<double> &sdev) {
	if(sdev)
		return;

	if(miu) {
		sdev = norm(patch - miu.value(), NORM_L2, mask) / sqrt(countNonZero(mask));
	} else {
		Scalar miu_, sdev_;
		meanStdDev(patch, miu_, sdev_, mask);
		miu = *miu_.val;
		sdev = *sdev_.val;
	}
	assert(*sdev < CachedData::sdevMaxFgBg+EPS);
}

void MatchParams::computeSdevFg(const Mat &patch, const SymData &symData) {
	computeSdev(patch, symData.symAndMasks[SymData::FG_MASK_IDX], fg, sdevFg);
}

void MatchParams::computeSdevBg(const Mat &patch, const SymData &symData) {
	computeSdev(patch, symData.symAndMasks[SymData::BG_MASK_IDX], bg, sdevBg);
}

void MatchParams::computePatchApprox(const Mat &patch, const SymData &symData) {
	if(patchApprox)
		return;

	computeContrast(patch, symData);

	if(contrast.value() == 0.) {
		patchApprox = Mat(patch.rows, patch.cols, CV_64FC1, Scalar(bg.value()));
		return;
	}

	patchApprox = bg.value() +
		symData.symAndMasks[SymData::GROUNDED_SYM_IDX] * (contrast.value() / symData.diffMinMax);
}

void MatchParams::computeSdevEdge(const Mat &patch, const SymData &symData) {
	if(sdevEdge)
		return;

	const auto &edgeMask = symData.symAndMasks[SymData::EDGE_MASK_IDX];
	const int cnz = countNonZero(edgeMask);
	if(cnz == 0) {
		sdevEdge = 0.;
		return;
	}

	computePatchApprox(patch, symData);

	sdevEdge = norm(patch, patchApprox.value(), NORM_L2, edgeMask) / sqrt(cnz);
	assert(*sdevEdge < CachedData::sdevMaxEdge+EPS);
}

void MatchParams::computeSymDensity(const SymData &symData, const CachedData &cachedData) {
	if(symDensity)
		return;

	symDensity = symData.pixelSum / cachedData.sz2;
	assert(*symDensity < 1.+EPS);
}

void MatchParams::computeMcPatch(const Mat &patch, const CachedData &cachedData) {
	if(mcPatch)
		return;

	const double patchSum = *sum(patch).val;
	Mat temp, temp1;
	reduce(patch, temp, 0, CV_REDUCE_SUM);	// sum all rows
	reduce(patch, temp1, 1, CV_REDUCE_SUM);	// sum all columns

	mcPatch = Point2d(temp.dot(cachedData.consec), temp1.t().dot(cachedData.consec))
		/ patchSum;
	assert(mcPatch->x > -EPS && mcPatch->x < cachedData.sz_1+EPS);
	assert(mcPatch->y > -EPS && mcPatch->y < cachedData.sz_1+EPS);
}

void MatchParams::computeMcPatchApprox(const Mat &patch, const SymData &symData,
									   const CachedData &cachedData) {
	if(mcPatchApprox)
		return;

	computeContrast(patch, symData);
	computeSymDensity(symData, cachedData);

	// Obtaining glyph's mass center
	const double k = symDensity.value() * contrast.value(),
		delta = .5 * bg.value() * cachedData.sz_1,
		denominator = k + bg.value();
	if(denominator == 0.)
		mcPatchApprox = cachedData.patchCenter;
	else
		mcPatchApprox = (k * symData.mc + Point2d(delta, delta)) / denominator;
	assert(mcPatchApprox->x > -EPS && mcPatchApprox->x < cachedData.sz_1+EPS);
	assert(mcPatchApprox->y > -EPS && mcPatchApprox->y < cachedData.sz_1+EPS);
}

void MatchParams::computeMcsOffset(const Mat &patch, const SymData &symData,
								   const CachedData &cachedData) {
	if(mcsOffset)
		return;

	computeMcPatch(patch, cachedData);
	computeMcPatchApprox(patch, symData, cachedData);

	mcsOffset = norm(mcPatch.value() - mcPatchApprox.value());
	assert(mcsOffset < cachedData.sz_1*sqrt(2) + EPS);
}

BestMatch& BestMatch::update(double score_, unsigned long symCode_,
							 unsigned symIdx_, const SymData &sd,
							 const MatchParams &mp) {
	score = score_;
	symCode = symCode_;
	symIdx = symIdx_;
	pSymData = &sd;
	bestVariant.params = mp;
	return *this;
}

BestMatch& BestMatch::updatePatchApprox(const MatchSettings &ms) {
	if(nullptr == pSymData) {
		bestVariant.approx = patch.blurred;
		return *this;
	}

	const auto &dataOfBest = *pSymData;
	const auto &matricesForBest = dataOfBest.symAndMasks;
	const Mat &groundedBest = matricesForBest[SymData::GROUNDED_SYM_IDX];

	Mat patchResult;

	if(patch.isColor) {
		const Mat &fgMask = matricesForBest[SymData::FG_MASK_IDX],
			&bgMask = matricesForBest[SymData::BG_MASK_IDX];

		vector<Mat> channels;
		split(patch.orig, channels);

		double miuFg, miuBg, newDiff, diffFgBg = 0.;
		for(auto &ch : channels) {
			ch.convertTo(ch, CV_64FC1); // processing double values

			miuFg = *mean(ch, fgMask).val;
			miuBg = *mean(ch, bgMask).val;
			newDiff = miuFg - miuBg;

			groundedBest.convertTo(ch, CV_8UC1, newDiff / dataOfBest.diffMinMax, miuBg);

			diffFgBg += abs(newDiff);
		}

		if(diffFgBg < 3.*ms.getBlankThreshold())
			patchResult = Mat(patch.sz, patch.sz, CV_8UC3, mean(patch.orig));
		else
			merge(channels, patchResult);

	} else { // grayscale result
		auto &params = bestVariant.params;
		params.computeContrast(patch.orig, *pSymData);

		if(abs(*params.contrast) < ms.getBlankThreshold())
			patchResult = Mat(patch.sz, patch.sz, CV_8UC1, Scalar(*mean(patch.orig).val));
		else
			groundedBest.convertTo(patchResult, CV_8UC1,
			*params.contrast / dataOfBest.diffMinMax,
			*params.bg);
	}

	bestVariant.approx = patchResult;

	// For non-hybrid result mode we're done
	if(!ms.isHybridResult())
		return *this;

	// Hybrid Result Mode - Combine the approximation with the blurred patch:
	// the less satisfactory the approximation is,
	// the more the weight of the blurred patch should be
	Scalar miu, sdevApproximation, sdevBlurredPatch;
	meanStdDev(patch.orig-bestVariant.approx, miu, sdevApproximation);
	meanStdDev(patch.orig-patch.blurred, miu, sdevBlurredPatch);
	double totalSdevBlurredPatch = *sdevBlurredPatch.val,
		totalSdevApproximation = *sdevApproximation.val;
	if(patch.isColor) {
		totalSdevBlurredPatch += sdevBlurredPatch.val[1] + sdevBlurredPatch.val[2];
		totalSdevApproximation += sdevApproximation.val[1] + sdevApproximation.val[2];
	}
	const double sdevSum = totalSdevBlurredPatch + totalSdevApproximation;
	const double weight = (sdevSum > 0.) ? (totalSdevApproximation / sdevSum) : 0.;
	Mat combination;
	addWeighted(patch.blurred, weight, bestVariant.approx, 1.-weight, 0., combination);
	bestVariant.approx = combination;

	return *this;
}

#if defined _DEBUG || defined UNIT_TESTING

BestMatch& BestMatch::setUnicode(bool unicode_) {
	unicode = unicode_;
	return *this;
}

#endif // _DEBUG