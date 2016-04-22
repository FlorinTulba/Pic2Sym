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
#include "matchAspectsFactory.h"
#include "matchParams.h"
#include "patch.h"
#include "settings.h"
#include "misc.h"

#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

/// Conversion PixMapSym -> Mat of type double with range [0..1] instead of [0..255]
static Mat toMat(const PixMapSym &pms, unsigned fontSz) {
	Mat result((int)fontSz, (int)fontSz, CV_8UC1, Scalar(0U));

	int firstRow = (int)fontSz-(int)pms.top-1;
	Mat region(result,
					Range(firstRow, firstRow+(int)pms.rows),
					Range((int)pms.left, (int)(pms.left+pms.cols)));

	const Mat pmsData((int)pms.rows, (int)pms.cols, CV_8UC1, (void*)pms.pixels.data());
	pmsData.copyTo(region);

	static const double INV_255 = 1./255;
	result.convertTo(result, CV_64FC1, INV_255); // convert to double

	return result;
}

MatchEngine::MatchEngine(const Settings &cfg_, FontEngine &fe_) : cfg(cfg_), fe(fe_) {
	for(const auto &aspectName: MatchAspect::aspectNames())
		availAspects.push_back(
			MatchAspectsFactory::create(aspectName, cachedData, cfg_.matchSettings()));
}

void MatchEngine::updateSymbols() {
	const string idForSymsToUse = getIdForSymsToUse(); // throws for invalid cmap/size
	if(symsIdReady.compare(idForSymsToUse) == 0)
		return; // already up to date

	extern const Size BlurWinSize;
	extern const double BlurStandardDeviation;
	extern const bool PrepareMoreGlyphsAtOnce, ParallelizeGlyphMasks;

	// constants for foreground / background thresholds
	// 1/255 = 0.00392, so 0.004 tolerates pixels with 1 brightness unit less / more than ideal
	// STILL_BG was set to 0, as there are font families with extremely similar glyphs.
	// When Unit Testing shouldn't identify exactly each glyph, STILL_BG might be > 0.
	// But testing on 'BPmonoBold.ttf' does tolerate such larger values (0.025, for instance).
	extern const double MatchEngine_updateSymbols_STILL_BG;					// darkest shades
	static const double STILL_FG = 1. - MatchEngine_updateSymbols_STILL_BG;	// brightest shades
	symsSet.clear();
	const auto &rawSyms = fe.symsSet();
	const int symsCount = (int)rawSyms.size();
	symsSet.reserve(symsCount);

	double minVal, maxVal;
	const unsigned sz = cfg.symSettings().getFontSz();
#pragma omp parallel for schedule(dynamic) if(PrepareMoreGlyphsAtOnce)
	for(int i = 0; i<symsCount; ++i) {
		const auto &pms = rawSyms[i];
		Mat negGlyph, fgMask, bgMask, edgeMask, blurOfGroundedGlyph, varianceOfGroundedGlyph;
		const Mat glyph = toMat(pms, sz);
		glyph.convertTo(negGlyph, CV_8UC1, -255., 255.);

		// for very small fonts, minVal might be > 0 and maxVal might be < 255
		minMaxIdx(glyph, &minVal, &maxVal);
		const Mat groundedGlyph = (minVal==0. ? glyph : (glyph - minVal)); // min val on 0
#pragma omp parallel sections if(ParallelizeGlyphMasks) // Nested parallel regions are serialized by default
		{
#pragma omp section
			{
				fgMask = (glyph >= (minVal + STILL_FG * (maxVal-minVal)));
				bgMask = (glyph <= (minVal + MatchEngine_updateSymbols_STILL_BG * (maxVal-minVal)));

				// Storing a blurred version of the grounded glyph for structural similarity match aspect
				GaussianBlur(groundedGlyph, blurOfGroundedGlyph,
							 BlurWinSize, BlurStandardDeviation, 0.,
							 BORDER_REPLICATE);
			}
#pragma omp section
			{
				// edgeMask selects all pixels that are not minVal, nor maxVal
				inRange(glyph, minVal+EPS, maxVal-EPS, edgeMask);

				// Storing also the variance of the grounded glyph for structural similarity match aspect
				// Actual varianceOfGroundedGlyph is obtained in the subtraction after the blur
				GaussianBlur(groundedGlyph.mul(groundedGlyph), varianceOfGroundedGlyph,
							 BlurWinSize, BlurStandardDeviation, 0.,
							 BORDER_REPLICATE);
			}
		}
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

	enabledAspects.clear();
	for(auto pAspect : availAspects)
		if(pAspect->enabled())
			enabledAspects.push_back(&*pAspect);
}

BestMatch MatchEngine::approxPatch(const Patch &patch) const {
	BestMatch best(patch);
	if(!patch.needsApproximation)
		return BestMatch(patch).updatePatchApprox(cfg.matchSettings());

	MatchParams mp;
	unsigned idx = 0U;
	for(const auto &symData : symsSet) {
		double score = assessMatch(patch.matrixToApprox(), symData, mp);

		if(score > best.score)
			best.update(score, symData.code, idx, symData, mp);

		mp.reset();
		++idx;
	}
	return best.updatePatchApprox(cfg.matchSettings());
}

double MatchEngine::assessMatch(const Mat &patch,
								const SymData &symData,
								MatchParams &mp) const {
	double score = 1.;
	for(auto pAspect : enabledAspects)
		score *= pAspect->assessMatch(patch, symData, mp);
	return score;
}

#ifdef _DEBUG

bool MatchEngine::usesUnicode() const {
	return fe.getEncoding().compare("UNICODE") == 0;
}

#else // _DEBUG not defined

bool MatchEngine::usesUnicode() const { return true; }

#endif // _DEBUG
