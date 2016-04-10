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

#include "matchEngine.h"
#include "matchParams.h"
#include "settings.h"
#include "misc.h"

#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

namespace {
	/// Conversion PixMapSym -> cv::Mat of type double with range [0..1] instead of [0..255]
	cv::Mat toMat(const PixMapSym &pms, unsigned fontSz) {
		cv::Mat result((int)fontSz, (int)fontSz, CV_8UC1, cv::Scalar(0U));

		int firstRow = (int)fontSz-(int)pms.top-1;
		cv::Mat region(result,
					   cv::Range(firstRow, firstRow+(int)pms.rows),
					   cv::Range((int)pms.left, (int)(pms.left+pms.cols)));

		const cv::Mat pmsData((int)pms.rows, (int)pms.cols, CV_8UC1, (void*)pms.pixels.data());
		pmsData.copyTo(region);

		static const double INV_255 = 1./255;
		result.convertTo(result, CV_64FC1, INV_255); // convert to double

		return result;
	}
}

MatchEngine::MatchEngine(const Settings &cfg_, FontEngine &fe_) :
		cfg(cfg_), fe(fe_), strSimMatch(cachedData, cfg_.matchSettings()),
		fgMatch(cachedData, cfg_.matchSettings()), bgMatch(cachedData, cfg_.matchSettings()),
		edgeMatch(cachedData, cfg_.matchSettings()), conMatch(cachedData, cfg_.matchSettings()),
		grMatch(cachedData, cfg_.matchSettings()), dirMatch(cachedData, cfg_.matchSettings()),
		lsMatch(cachedData, cfg_.matchSettings()) {}

extern const double MatchEngine_updateSymbols_STILL_BG;

void MatchEngine::updateSymbols() {
	const string idForSymsToUse = getIdForSymsToUse(); // throws for invalid cmap/size
	if(symsIdReady.compare(idForSymsToUse) == 0)
		return; // already up to date

	// constants for foreground / background thresholds
	// 1/255 = 0.00392, so 0.004 tolerates pixels with 1 brightness unit less / more than ideal
	// STILL_BG was set to 0, as there are font families with extremely similar glyphs.
	// When Unit Testing shouldn't identify exactly each glyph, STILL_BG might be > 0.
	// But testing on 'BPmonoBold.ttf' does tolerate such larger values (0.025, for instance).
	static const double STILL_BG = MatchEngine_updateSymbols_STILL_BG,	// darkest shades
					STILL_FG = 1. - STILL_BG;							// brightest shades
	symsSet.clear();
	symsSet.reserve(fe.symsSet().size());

	double minVal, maxVal;
	const unsigned sz = cfg.symSettings().getFontSz();
	for(const auto &pms : fe.symsSet()) {
		cv::Mat negGlyph, edgeMask, blurOfGroundedGlyph, varianceOfGroundedGlyph;
		const cv::Mat glyph = toMat(pms, sz);
		glyph.convertTo(negGlyph, CV_8UC1, -255., 255.);

		// for very small fonts, minVal might be > 0 and maxVal might be < 255
		minMaxIdx(glyph, &minVal, &maxVal);
		const cv::Mat groundedGlyph = (minVal==0. ? glyph : (glyph - minVal)), // min val on 0
			fgMask = (glyph >= (minVal + STILL_FG * (maxVal-minVal))),
			bgMask = (glyph <= (minVal + STILL_BG * (maxVal-minVal)));
		inRange(glyph, minVal+EPS, maxVal-EPS, edgeMask);

		// Storing a blurred version of the grounded glyph for structural similarity match aspect
		GaussianBlur(groundedGlyph, blurOfGroundedGlyph,
					 StructuralSimilarity::WIN_SIZE, StructuralSimilarity::SIGMA, 0.,
					 cv::BORDER_REPLICATE);

		// Storing also the variance of the grounded glyph for structural similarity match aspect
		GaussianBlur(groundedGlyph.mul(groundedGlyph), varianceOfGroundedGlyph,
					 StructuralSimilarity::WIN_SIZE, StructuralSimilarity::SIGMA, 0.,
					 cv::BORDER_REPLICATE);
		varianceOfGroundedGlyph -= blurOfGroundedGlyph.mul(blurOfGroundedGlyph);

		symsSet.emplace_back(pms.symCode,
							 minVal, maxVal-minVal,
							 pms.glyphSum, pms.mc,
							 SymData::MatArray { {
									 fgMask,					// FG_MASK_IDX 
									 bgMask,					// BG_MASK_IDX
									 edgeMask,					// EDGE_MASK_IDX
									 negGlyph,					// NEG_SYM_IDX
									 groundedGlyph,				// GROUNDED_SYM_IDX
									 blurOfGroundedGlyph,		// BLURRED_GR_SYM_IDX
									 varianceOfGroundedGlyph	// VARIANCE_GR_SYM_IDX
								 } });
	}

	symsIdReady = idForSymsToUse; // ready to use the new cmap&size
}

MatchEngine::VSymDataCItPair MatchEngine::getSymsRange(unsigned from, unsigned count) const {
	const unsigned sz = (unsigned)symsSet.size();
	const VSymDataCIt itEnd = symsSet.cend();
	if(from >= sz)
		return make_pair(itEnd, itEnd);

	const VSymDataCIt itStart = next(symsSet.cbegin(), from);
	if(from + count >= sz)
		return make_pair(itStart, itEnd);

	return make_pair(itStart, next(itStart, count));
}

unsigned MatchEngine::getSymsCount() const {
	return (unsigned)symsSet.size();
}

void MatchEngine::getReady() {
	updateSymbols();

	cachedData.update(cfg.symSettings().getFontSz(), fe);

	aspects.clear();
	for(auto pAspect : MatchAspect::getAvailAspects())
		if(pAspect->enabled())
			aspects.push_back(pAspect);
}

cv::Mat MatchEngine::approxPatch(const cv::Mat &patch_, BestMatch &best) {
	// All blurring techniques I've tried seem not worthy => using original
	cv::Mat blurredPatch = patch_;
	const unsigned sz = patch_.rows;
	const bool isColor = (blurredPatch.channels() > 1);
	cv::Mat patchColor, patch, patchResult;
	if(isColor) {
		patchColor = blurredPatch;
		cv::cvtColor(patchColor, patch, cv::COLOR_RGB2GRAY);
	} else patch = blurredPatch;
	patch.convertTo(patch, CV_64FC1);

	findBestMatch(patch, best);

	const auto &dataOfBest = symsSet[best.symIdx];
	const auto &matricesForBest = dataOfBest.symAndMasks;
	const cv::Mat &groundedBest = matricesForBest[SymData::GROUNDED_SYM_IDX];

	if(isColor) {
		const cv::Mat &fgMask = matricesForBest[SymData::FG_MASK_IDX],
			&bgMask = matricesForBest[SymData::BG_MASK_IDX];

		vector<cv::Mat> channels;
		cv::split(patchColor, channels);

		double miuFg, miuBg, newDiff, diffFgBg = 0.;
		for(auto &ch : channels) {
			ch.convertTo(ch, CV_64FC1); // processing double values

			miuFg = *cv::mean(ch, fgMask).val;
			miuBg = *cv::mean(ch, bgMask).val;
			newDiff = miuFg - miuBg;

			groundedBest.convertTo(ch, CV_8UC1, newDiff / dataOfBest.diffMinMax, miuBg);

			diffFgBg += abs(newDiff);
		}

		if(diffFgBg < 3.*cfg.matchSettings().getBlankThreshold())
			patchResult = cv::Mat(sz, sz, CV_8UC3, cv::mean(patchColor));
		else
			cv::merge(channels, patchResult);

	} else { // grayscale result
		auto &params = best.params;
		params.computeContrast(patch, symsSet[best.symIdx]);

		if(abs(*params.contrast) < cfg.matchSettings().getBlankThreshold())
			patchResult = cv::Mat(sz, sz, CV_8UC1, cv::Scalar(*cv::mean(patch).val));
		else
			groundedBest.convertTo(patchResult, CV_8UC1,
			*params.contrast / dataOfBest.diffMinMax,
			*params.bg);
	}
	return patchResult;
}

double MatchEngine::assessMatch(const cv::Mat &patch,
								const SymData &symData,
								MatchParams &mp) const {
	double score = 1.;
	for(auto pAspect : aspects)
		score *= pAspect->assessMatch(patch, symData, mp);
	return score;
}

void MatchEngine::findBestMatch(const cv::Mat &patch, BestMatch &best) {
	MatchParams mp;
	unsigned idx = 0U;
	for(const auto &symData : symsSet) {
		double score = assessMatch(patch, symData, mp);

		if(score > best.score)
			best.update(score, idx, symData.code, mp);

		mp.reset();
		++idx;
	}
}

#ifdef _DEBUG
bool MatchEngine::usesUnicode() const {
	return fe.getEncoding().compare("UNICODE") == 0;
}
#endif // _DEBUG
