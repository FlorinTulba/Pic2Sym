/************************************************************************************************
 The application Pic2Sym approximates images by a
 grid of colored symbols with colored backgrounds.

 Copyrights from the libraries used by the program:
 - (c) 2016 Boost (www.boost.org)
		License: <http://www.boost.org/LICENSE_1_0.txt>
			or doc/licenses/Boost.lic
 - (c) 2015 OpenCV (www.opencv.org)
		License: <http://opencv.org/license.html>
            or doc/licenses/OpenCV.lic
 - (c) 2015 The FreeType Project (www.freetype.org)
		License: <http://git.savannah.gnu.org/cgit/freetype/freetype2.git/plain/docs/FTL.TXT>
	        or doc/licenses/FTL.txt
 - (c) 1997-2002 OpenMP Architecture Review Board (www.openmp.org)
   (c) Microsoft Corporation (Visual C++ implementation for OpenMP C/C++ Version 2.0 March 2002)
		See: <https://msdn.microsoft.com/en-us/library/8y6825x5(v=vs.140).aspx>
 - (c) 1995-2013 zlib software (Jean-loup Gailly and Mark Adler - see: www.zlib.net)
		License: <http://www.zlib.net/zlib_license.html>
            or doc/licenses/zlib.lic
 
 (c) 2016, 2017 Florin Tulba <florintulba@yahoo.com>

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
 ***********************************************************************************************/

#include "bestMatch.h"
#include "matchParams.h"
#include "patchBase.h"
#include "matchSettings.h"
#include "symDataBase.h"

using namespace std;
using namespace boost;
using namespace cv;

BestMatch::BestMatch(const IPatch &patch_) :
		patch(patch_.clone()),
		params(patch_.nonUniform() ? make_unique<MatchParams>() : nullptr) {
	assert(patch);
}

const IPatch& BestMatch::getPatch() const { return *patch; }
const Mat& BestMatch::getApprox() const { return approx; }

const optional<const IMatchParams&> BestMatch::getParams() const {
	if(params)
		return *params;
	return none;
}

unique_ptr<IMatchParamsRW>& BestMatch::refParams() { return params; }
const optional<unsigned>& BestMatch::getSymIdx() const { return symIdx; }

const optional<unsigned>& BestMatch::getLastPromisingNontrivialCluster() const {
	return lastPromisingNontrivialCluster;
}

BestMatch& BestMatch::setLastPromisingNontrivialCluster(unsigned clustIdx) {
	lastPromisingNontrivialCluster = clustIdx;
	return *this;
}

const optional<unsigned long>& BestMatch::getSymCode() const { return symCode; }
double BestMatch::getScore() const { return score; }

BestMatch& BestMatch::setScore(double score_) { score = score_; return *this; }

BestMatch& BestMatch::reset() {
	score = 0.;
	symCode = none;
	symIdx = lastPromisingNontrivialCluster = none;
	pSymData = nullptr;
	approx = Mat();
	if(params)
		params->reset(); // keeps patch-invariant parameters
	return *this;
}

BestMatch& BestMatch::update(double score_, unsigned long symCode_,
							 unsigned symIdx_, const ISymData &sd) {
	score = score_;
	symCode = symCode_;
	symIdx = symIdx_;
	pSymData = &sd;
	return *this;
}

BestMatch& BestMatch::updatePatchApprox(const MatchSettings &ms) {
	if(nullptr == pSymData) {
		approx = patch->getBlurred();
		return *this;
	}

	const auto &dataOfBest = *pSymData;
	const auto &matricesForBest = dataOfBest.getMasks();
	const Mat &groundedBest = matricesForBest[ISymData::GROUNDED_SYM_IDX];
	const auto patchSz = patch->getOrig().rows;

	Mat patchResult;
	if(patch->isColored()) {
		const Mat &fgMask = matricesForBest[ISymData::FG_MASK_IDX],
			&bgMask = matricesForBest[ISymData::BG_MASK_IDX];

		vector<Mat> channels;
		split(patch->getOrig(), channels);

		double diffFgBg = 0.;
		const size_t channelsCount = channels.size();
		for(auto &ch : channels) {
			ch.convertTo(ch, CV_64FC1); // processing double values

			double miuFg, miuBg, newDiff;
			miuFg = *mean(ch, fgMask).val;
			miuBg = *mean(ch, bgMask).val;
			newDiff = miuFg - miuBg;

			groundedBest.convertTo(ch, CV_8UC1, newDiff / dataOfBest.getDiffMinMax(), miuBg);

			diffFgBg += abs(newDiff);
		}

		if(diffFgBg < channelsCount * ms.getBlankThreshold())
			patchResult = Mat(patchSz, patchSz, CV_8UC3, mean(patch->getOrig()));
		else
			merge(channels, patchResult);

	} else { // grayscale result
		auto &params = refParams();
		assert(params);
		params->computeContrast(patch->getOrig(), *pSymData);

		if(abs(*params->getContrast()) < ms.getBlankThreshold())
			patchResult = Mat(patchSz, patchSz, CV_8UC1, Scalar(*mean(patch->getOrig()).val));
		else
			groundedBest.convertTo(patchResult, CV_8UC1,
								*params->getContrast() / dataOfBest.getDiffMinMax(),
								*params->getBg());
	}

	approx = patchResult;

	// For non-hybrid result mode we're done
	if(!ms.isHybridResult())
		return *this;

	// Hybrid Result Mode - Combine the approximation with the blurred patch:
	// the less satisfactory the approximation is,
	// the more the weight of the blurred patch should be
	Scalar miu, sdevApproximation, sdevBlurredPatch;
	meanStdDev(patch->getOrig()-approx, miu, sdevApproximation);
	meanStdDev(patch->getOrig()-patch->getBlurred(), miu, sdevBlurredPatch);

	double totalSdevBlurredPatch = *sdevBlurredPatch.val,
		totalSdevApproximation = *sdevApproximation.val;
	if(patch->isColored()) {
		totalSdevBlurredPatch += sdevBlurredPatch.val[1] + sdevBlurredPatch.val[2];
		totalSdevApproximation += sdevApproximation.val[1] + sdevApproximation.val[2];
	}
	const double sdevSum = totalSdevBlurredPatch + totalSdevApproximation;
	const double weight = (sdevSum > 0.) ? (totalSdevApproximation / sdevSum) : 0.;
	Mat combination;
	addWeighted(patch->getBlurred(), weight, approx, 1.-weight, 0., combination);
	approx = combination;

	return *this;
}

#if defined _DEBUG || defined UNIT_TESTING

bool BestMatch::isUnicode() const {
	return unicode;
}

BestMatch& BestMatch::setUnicode(bool unicode_) {
	unicode = unicode_;
	return *this;
}

#endif // _DEBUG || UNIT_TESTING
