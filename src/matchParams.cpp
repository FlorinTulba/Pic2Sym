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
#include "symData.h"
#include "cachedData.h"
#include "misc.h"

using namespace std;
using namespace boost;

#if defined _DEBUG || defined UNIT_TESTING

BestMatch::BestMatch(bool isUnicode/* = true*/) : unicode(isUnicode) {}

BestMatch& BestMatch::operator=(const BestMatch &other) {
	if(this != &other) {
		score = other.score;
		symIdx = other.symIdx;
		symCode = other.symCode;
		const_cast<bool&>(unicode) = other.unicode;
		params = other.params;
	}
	return *this;
}

#endif // _DEBUG

void MatchParams::reset(bool skipPatchInvariantParts/* = true*/) {
	mcPatchApprox = none;
	patchApprox = none;
	ssim = fg = bg = contrast = sdevFg = sdevBg = sdevEdge = symDensity = mcsOffset = none;

	if(!skipPatchInvariantParts) {
		mcPatch = none;
		blurredPatch = blurredPatchSq = variancePatch = none;
	}
}

void MatchParams::computeMean(const cv::Mat &patch, const cv::Mat &mask, optional<double> &miu) {
	if(miu)
		return;

	miu = *cv::mean(patch, mask).val;
	assert(*miu > -EPS && *miu < 255.+EPS);
}

void MatchParams::computeFg(const cv::Mat &patch, const SymData &symData) {
	computeMean(patch, symData.symAndMasks[SymData::FG_MASK_IDX], fg);
}

void MatchParams::computeBg(const cv::Mat &patch, const SymData &symData) {
	computeMean(patch, symData.symAndMasks[SymData::BG_MASK_IDX], bg);
}

void MatchParams::computeContrast(const cv::Mat &patch, const SymData &symData) {
	if(contrast)
		return;

	computeFg(patch, symData);
	computeBg(patch, symData);

	contrast = fg.value() - bg.value();
	assert(abs(contrast.value()) < 255.5);
}

void MatchParams::computeSdev(const cv::Mat &patch, const cv::Mat &mask,
							  optional<double> &miu, optional<double> &sdev) {
	if(sdev)
		return;

	if(miu) {
		sdev = cv::norm(patch - miu.value(), cv::NORM_L2, mask) / sqrt(countNonZero(mask));
	} else {
		cv::Scalar miu_, sdev_;
		meanStdDev(patch, miu_, sdev_, mask);
		miu = *miu_.val;
		sdev = *sdev_.val;
	}
	assert(*sdev < CachedData::sdevMaxFgBg+EPS);
}

void MatchParams::computeSdevFg(const cv::Mat &patch, const SymData &symData) {
	computeSdev(patch, symData.symAndMasks[SymData::FG_MASK_IDX], fg, sdevFg);
}

void MatchParams::computeSdevBg(const cv::Mat &patch, const SymData &symData) {
	computeSdev(patch, symData.symAndMasks[SymData::BG_MASK_IDX], bg, sdevBg);
}

void MatchParams::computePatchApprox(const cv::Mat &patch, const SymData &symData) {
	if(patchApprox)
		return;

	computeContrast(patch, symData);

	if(contrast.value() == 0.) {
		patchApprox = cv::Mat(patch.rows, patch.cols, CV_64FC1, cv::Scalar(bg.value()));
		return;
	}

	patchApprox = bg.value() +
		symData.symAndMasks[SymData::GROUNDED_SYM_IDX] * (contrast.value() / symData.diffMinMax);
}

void MatchParams::computeSdevEdge(const cv::Mat &patch, const SymData &symData) {
	if(sdevEdge)
		return;

	const auto &edgeMask = symData.symAndMasks[SymData::EDGE_MASK_IDX];
	const int cnz = countNonZero(edgeMask);
	if(cnz == 0) {
		sdevEdge = 0.;
		return;
	}

	computePatchApprox(patch, symData);

	sdevEdge = cv::norm(patch, patchApprox.value(), cv::NORM_L2, edgeMask) / sqrt(cnz);
	assert(*sdevEdge < CachedData::sdevMaxEdge+EPS);
}

void MatchParams::computeSymDensity(const SymData &symData, const CachedData &cachedData) {
	if(symDensity)
		return;

	symDensity = symData.pixelSum / cachedData.sz2;
	assert(*symDensity < 1.+EPS);
}

void MatchParams::computeMcPatch(const cv::Mat &patch, const CachedData &cachedData) {
	if(mcPatch)
		return;

	const double patchSum = *sum(patch).val;
	cv::Mat temp, temp1;
	reduce(patch, temp, 0, CV_REDUCE_SUM);	// sum all rows
	reduce(patch, temp1, 1, CV_REDUCE_SUM);	// sum all columns

	mcPatch = cv::Point2d(temp.dot(cachedData.consec), temp1.t().dot(cachedData.consec))
		/ patchSum;
	assert(mcPatch->x > -EPS && mcPatch->x < cachedData.sz_1+EPS);
	assert(mcPatch->y > -EPS && mcPatch->y < cachedData.sz_1+EPS);
}

void MatchParams::computeMcPatchApprox(const cv::Mat &patch, const SymData &symData,
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
		mcPatchApprox = (k * symData.mc + cv::Point2d(delta, delta)) / denominator;
	assert(mcPatchApprox->x > -EPS && mcPatchApprox->x < cachedData.sz_1+EPS);
	assert(mcPatchApprox->y > -EPS && mcPatchApprox->y < cachedData.sz_1+EPS);
}

void MatchParams::computeMcsOffset(const cv::Mat &patch, const SymData &symData,
								   const CachedData &cachedData) {
	if(mcsOffset)
		return;

	computeMcPatch(patch, cachedData);
	computeMcPatchApprox(patch, symData, cachedData);

	mcsOffset = cv::norm(mcPatch.value() - mcPatchApprox.value());
	assert(mcsOffset < cachedData.sz_1*sqrt(2) + EPS);
}

void BestMatch::update(double score_, unsigned symIdx_, unsigned long symCode_,
					   const MatchParams &params_) {
	score = score_;
	symIdx = symIdx_;
	symCode = symCode_;
	params = params_;
}
