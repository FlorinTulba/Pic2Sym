/**********************************************************
 Project:     Pic2Sym
 File:        match.cpp

 Author:      Florin Tulba
 Created on:  2016-2-1
 
 Copyright (c) 2016 Florin Tulba
 **********************************************************/

#include "match.h"
#include "controller.h"

#include <numeric>

#include <boost/optional/optional_io.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace boost;

namespace {
	// Conversion PixMapSym -> cv::Mat of type double with range [0..1] instead of [0..255]
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

const double CachedData::sdevMax = 127.5;

#if defined _DEBUG || defined UNIT_TESTING

#	define comma L",\t"

const wstring MatchParams::HEADER(L"#mcGlyphX" comma L"#mcGlyphY" comma
								  L"#mcPatchX" comma L"#mcPatchY" comma
								  L"#fg" comma L"#bg" comma
								  L"#sdevFg" comma L"#sdevEdge" comma L"#sdevBg" comma
								  L"#rho");

wostream& operator<<(wostream &os, const MatchParams &mp) {
	if(mp.mcGlyph)
		os<<mp.mcGlyph->x<<comma<<mp.mcGlyph->y<<comma;
	else
		os<<none<<comma<<none<<comma;

	if(mp.mcPatch)
		os<<mp.mcPatch->x<<comma<<mp.mcPatch->y<<comma;
	else
		os<<none<<comma<<none<<comma;

	os<<mp.fg<<comma<<mp.bg<<comma
		<<mp.sdevFg<<comma<<mp.sdevEdge<<comma<<mp.sdevBg<<comma
		<<mp.glyphWeight;
	return os;
}

const wstring BestMatch::HEADER(wstring(L"#GlyphCode" comma L"#ChosenScore" comma) +
								MatchParams::HEADER);

wostream& operator<<(wostream &os, const BestMatch &bm) {
	unsigned long symCode = bm.symCode;
	if(bm.unicode) {
		if(symCode == (unsigned long)',')
			os<<L"COMMA";
		else if(symCode == (unsigned long)'(')
			os<<L"OPEN_PAR";
		else if(symCode == (unsigned long)')')
			os<<L"CLOSE_PAR";
		else if(os<<(wchar_t)symCode)
			os<<'('<<symCode<<')';
		else {
			os.clear();
			os<<symCode;
		}
	} else
		os<<symCode;

	os<<comma<<bm.score<<comma<<bm.params;
	return os;
}

#	undef comma

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

SymData::SymData(unsigned long code_, double minVal_, double diffMinMax_, double pixelSum_,
				 const cv::Point2d &mc_, const MatArray &symAndMasks_) :
		code(code_), minVal(minVal_), diffMinMax(diffMinMax_),
		pixelSum(pixelSum_), mc(mc_), symAndMasks(symAndMasks_) {
}

void MatchParams::reset(bool skipMcPatch/* = true*/) {
	mcGlyph = none;
	glyphWeight = fg = bg = sdevFg = sdevBg = sdevEdge = none;

	if(!skipMcPatch)
		mcPatch = none;
}

void MatchParams::computeMean(const cv::Mat &patch, const cv::Mat &mask, optional<double> &miu) {
	if(miu)
		return;
	miu = *cv::mean(patch, mask).val;
}

void MatchParams::computeFg(const cv::Mat &patch, const SymData &symData) {
	computeMean(patch, symData.symAndMasks[SymData::FG_MASK_IDX], fg);
}

void MatchParams::computeBg(const cv::Mat &patch, const SymData &symData) {
	computeMean(patch, symData.symAndMasks[SymData::BG_MASK_IDX], bg);
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
}

void MatchParams::computeSdevFg(const cv::Mat &patch, const SymData &symData) {
	computeSdev(patch, symData.symAndMasks[SymData::FG_MASK_IDX], fg, sdevFg);
}

void MatchParams::computeSdevBg(const cv::Mat &patch, const SymData &symData) {
	computeSdev(patch, symData.symAndMasks[SymData::BG_MASK_IDX], bg, sdevBg);
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

	computeFg(patch, symData);
	computeBg(patch, symData);

	const double diffFgBg = fg.value() - bg.value();
	if(diffFgBg == 0.) {
		sdevEdge = cv::norm(patch - bg.value(), cv::NORM_L2, edgeMask) / sqrt(cnz);
		return;
	}

	const cv::Mat approximationOfPatch = bg.value() +
		symData.symAndMasks[SymData::GROUNDED_GLYPH_IDX] * (diffFgBg / symData.diffMinMax);

	sdevEdge = cv::norm(patch, approximationOfPatch, cv::NORM_L2, edgeMask) / sqrt(cnz);
}

void MatchParams::computeRhoApproxSym(const SymData &symData, const CachedData &cachedData) {
	if(glyphWeight)
		return;

	glyphWeight = symData.pixelSum / cachedData.sz2;
}

void MatchParams::computeMcPatch(const cv::Mat &patch, const CachedData &cachedData) {
	if(mcPatch)
		return;

	const double patchSum = *sum(patch).val;
	cv::Mat temp, temp1;
	reduce(patch, temp, 0, CV_REDUCE_SUM);	// sum all rows
	reduce(patch, temp1, 1, CV_REDUCE_SUM);	// sum all columns

	mcPatch = cv::Point2d(temp.dot(cachedData.consec), temp1.t().dot(cachedData.consec)) / patchSum;
}

void MatchParams::computeMcApproxSym(const cv::Mat &patch, const SymData &symData,
									 const CachedData &cachedData) {
	if(mcGlyph)
		return;

	computeFg(patch, symData);
	computeBg(patch, symData);
	computeRhoApproxSym(symData, cachedData);

	// Obtaining glyph's mass center
	const double k = glyphWeight.value() * (fg.value() - bg.value()),
				delta = .5 * bg.value() * cachedData.sz_1,
				denominator = k + bg.value();
	if(denominator == 0.)
		mcGlyph = cachedData.patchCenter;
	else
		mcGlyph = (k * symData.mc + cv::Point2d(delta, delta)) / denominator;
}

void BestMatch::update(double score_, unsigned symIdx_, unsigned long symCode_,
					  const MatchParams &params_) {
	score = score_;
	symIdx = symIdx_;
	symCode = symCode_;
	params = params_;
}

double FgMatch::assessMatch(const cv::Mat &patch,
							const SymData &symData,
							MatchParams &mp) const {
	mp.computeSdevFg(patch, symData);

	// Returned value discourages large std. devs.
	// For sdev =     0 (min) => returns 1 no matter k
	// For sdev = 127.5 (max) => returns 0 no matter k (For k=0, this matching aspect is disabled)
	// For other sdev-s       =>
	//		returns closer to 1 for k in (0..1)
	//		returns sdev for k=1
	//		returns closer to 0 for k>1 (Large k => higher penalty for large sdev-s)
	return pow(1. - mp.sdevFg.value() / CachedData::sdevMax, k);
}

double BgMatch::assessMatch(const cv::Mat &patch,
							const SymData &symData,
							MatchParams &mp) const {
	mp.computeSdevBg(patch, symData);

	// Returned value discourages large std. devs.
	// For sdev =     0 (min) => returns 1 no matter k
	// For sdev = 127.5 (max) => returns 0 no matter k (For k=0, this matching aspect is disabled)
	// For other sdev-s       =>
	//		returns closer to 1 for k in (0..1)
	//		returns sdev for k=1
	//		returns closer to 0 for k>1 (Large k => higher penalty for large sdev-s)
	return pow(1. - mp.sdevBg.value()/CachedData::sdevMax, k);
}

double EdgeMatch::assessMatch(const cv::Mat &patch,
							const SymData &symData,
							MatchParams &mp) const {
	mp.computeSdevEdge(patch, symData);

	// Returned value discourages large std. devs.
	// For sdev =     0 (min) => returns 1 no matter k
	// For sdev = 127.5 (max) => returns 0 no matter k (For k=0, this matching aspect is disabled)
	// For other sdev-s       =>
	//		returns closer to 1 for k in (0..1)
	//		returns sdev for k=1
	//		returns closer to 0 for k>1 (Large k => higher penalty for large sdev-s)
	return pow(1. - mp.sdevEdge.value()/CachedData::sdevMax, k);
}

double BetterContrast::assessMatch(const cv::Mat &patch,
							const SymData &symData,
							MatchParams &mp) const {
	mp.computeFg(patch, symData);
	mp.computeBg(patch, symData);
	
	// range 0 .. 255, best when large
	const double contrast = abs(mp.fg.value() - mp.bg.value());

	// Encourages larger contrasts:
	// 0 for no contrast; 1 for max contrast (255)
	return pow(contrast / 255., k);
}

double GravitationalSmoothness::assessMatch(const cv::Mat &patch,
							const SymData &symData,
							MatchParams &mp) const {
	mp.computeMcPatch(patch, cachedData);
	mp.computeMcApproxSym(patch, symData, cachedData);

	// best glyph location is when mc-s are near to each other
	// range 0 .. 1.42*(fontSz-1), best when 0
	const double mcsOffset = cv::norm(mp.mcPatch.value() - mp.mcGlyph.value());

	// Discourages mcsOffset larger than preferredMaxMcDist:
	//		returns 1 for mcsOffset == preferredMaxMcDist, no matter k
	//		returns 0 for mcsOffset == 1.42*(fontSz-1), no matter k (k > 0)
	//		returns in (0..1) for mcsOffset in (preferredMaxMcDist .. 1.42*(fontSz-1) )
	//		returns > 1 for mcsOffset < preferredMaxMcDist
	// Larger k induces larger penalty for large mcsOffset and
	// also larger reward for small mcsOffset
	return pow(1. + (cachedData.preferredMaxMcDist - mcsOffset) / cachedData.complPrefMaxMcDist, k);
}

double DirectionalSmoothness::assessMatch(const cv::Mat &patch,
							const SymData &symData,
							MatchParams &mp) const {
	static const double SQRT2 = sqrt(2), TWOmSQRT2 = 2. - SQRT2;
	static const cv::Point2d ORIGIN; // (0, 0)

	mp.computeMcPatch(patch, cachedData);
	mp.computeMcApproxSym(patch, symData, cachedData);

	// best glyph location is when mc-s are near to each other
	// range 0 .. 1.42*(fontSz-1), best when 0
	const double mcsOffset = cv::norm(mp.mcPatch.value() - mp.mcGlyph.value());

	const cv::Point2d relMcPatch = mp.mcPatch.value() - cachedData.patchCenter;
	const cv::Point2d relMcGlyph = mp.mcGlyph.value() - cachedData.patchCenter;

	// best gradient orientation when angle between mc-s is 0 => cos = 1	
	double cosAngleMCs = 0.; // -1..1 range, best when 1
	if(relMcGlyph != ORIGIN && relMcPatch != ORIGIN) // avoid DivBy0
		cosAngleMCs = relMcGlyph.dot(relMcPatch) / (cv::norm(relMcGlyph) * cv::norm(relMcPatch));

	// Penalizes large angle between mc-s, but no so much when they are close to each other.
	// The mc-s are consider close when the distance between them is < preferredMaxMcDist
	//		(1. + cosAngleMCs) * (2-sqrt(2)) is <=1 for |angleMCs| >= 45  and  >1 otherwise
	// So, large k generally penalizes large angles and encourages small ones,
	// but fades gradually for nearer mc-s or fades completely when the mc-s overlap.
	return pow((1. + cosAngleMCs) * TWOmSQRT2,
			   k * min(mcsOffset, cachedData.preferredMaxMcDist) / cachedData.preferredMaxMcDist);
}

double LargerSym::assessMatch(const cv::Mat&,
							const SymData &symData,
							MatchParams &mp) const {
	mp.computeRhoApproxSym(symData, cachedData);

	// Encourages approximations with symbols filling at least x% of their box.
	// The threshold x is provided by smallGlyphsCoverage.
	// Returns < 1 for glyphs under threshold;   >= 1 otherwise
	return pow(mp.glyphWeight.value() + 1. - cachedData.smallGlyphsCoverage, k);
}

void CachedData::useNewSymSize(unsigned sz_) {
	sz = sz_;
	sz_1 = sz - 1U;
	sz2 = (double)sz * sz;

	preferredMaxMcDist = sz / 8.;
	complPrefMaxMcDist = sz_1 * sqrt(2) - preferredMaxMcDist;
	patchCenter = cv::Point2d(sz_1, sz_1) / 2;

	consec = cv::Mat(1, sz, CV_64FC1);
	iota(consec.begin<double>(), consec.end<double>(), 0.);
}

void CachedData::update(unsigned sz_, const FontEngine &fe_) {
	useNewSymSize(sz_);

	smallGlyphsCoverage = fe_.smallGlyphsCoverage();
}


MatchEngine::MatchEngine(const Settings &cfg_, FontEngine &fe_) :
	cfg(cfg_), fe(fe_),
	fgMatch(cachedData, cfg_.matchSettings()), bgMatch(cachedData, cfg_.matchSettings()),
	edgeMatch(cachedData, cfg_.matchSettings()), conMatch(cachedData, cfg_.matchSettings()),
	grMatch(cachedData, cfg_.matchSettings()), dirMatch(cachedData, cfg_.matchSettings()),
	lsMatch(cachedData, cfg_.matchSettings()) {
}

string MatchEngine::getIdForSymsToUse() {
	const unsigned sz = cfg.symSettings().getFontSz();
	if(!Settings::isFontSizeOk(sz)) {
		cerr<<"Invalid font size to use: "<<sz<<endl;
		throw logic_error("Invalid font size for getIdForSymsToUse");
	}

	ostringstream oss;
	oss<<fe.getFamily()<<'_'<<fe.getStyle()<<'_'<<fe.getEncoding()<<'_'<<sz;
	// this also throws logic_error if no family/style

	return oss.str();
}

void MatchEngine::updateSymbols() {
	const string idForSymsToUse = getIdForSymsToUse(); // throws for invalid cmap/size
	if(symsIdReady.compare(idForSymsToUse) == 0)
		return; // already up to date

	// constants for foreground / background thresholds
	// 1/255 = 0.00392, so 0.004 tolerates pixels with 1 brightness unit less / more than ideal
	// STILL_BG was set to 0, as there are font families with extremely similar glyphs.
	// When Unit Testing shouldn't identify exactly each glyph, STILL_BG might be > 0.
	// But testing on 'BPmonoBold.ttf' does tolerate such larger values (0.025, for instance).
	static const double STILL_BG = 0.,			// darkest shades
					STILL_FG = 1. - STILL_BG,	// brightest shades
					EPS = 1e-6;
	symsSet.clear();
	symsSet.reserve(fe.symsSet().size());

	double minVal, maxVal;
	const unsigned sz = cfg.symSettings().getFontSz();
	const int szGlyph[] = { 2, sz, sz },
		szMasks[] = { 4, sz, sz };
	for(const auto &pms : fe.symsSet()) {
		cv::Mat negGlyph, edgeMask;
		const cv::Mat glyph = toMat(pms, sz);
		glyph.convertTo(negGlyph, CV_8UC1, -255., 255.);

		// for very small fonts, minVal might be > 0 and maxVal might be < 255
		minMaxIdx(glyph, &minVal, &maxVal);
		const cv::Mat groundedGlyph = (minVal==0. ? glyph : (glyph - minVal)), // min val on 0
				fgMask = (glyph >= (minVal + STILL_FG * (maxVal-minVal))),
				bgMask = (glyph <= (minVal + STILL_BG * (maxVal-minVal)));
		inRange(glyph, minVal+EPS, maxVal-EPS, edgeMask);

		symsSet.emplace_back(pms.symCode,
							 minVal, maxVal-minVal,
							 pms.glyphSum, pms.mc,
							 SymData::MatArray { {
								fgMask,			// FG_MASK_IDX 
								bgMask,			// BG_MASK_IDX
								edgeMask,		// EDGE_MASK_IDX
								negGlyph,		// NEG_GLYPH_IDX
								groundedGlyph	// GROUNDED_GLYPH_IDX
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

void MatchEngine::getReady() {
	updateSymbols();

	cachedData.update(cfg.symSettings().getFontSz(), fe);

	aspects.clear();
	for(auto pAspect : getAvailAspects())
		if(pAspect->enabled())
			aspects.push_back(pAspect);
}

cv::Mat MatchEngine::approxPatch(const cv::Mat &patch_, BestMatch &best) {
	// All blurring techniques I've tried seem not worthy => using original
	cv::Mat blurredPatch = patch_;
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
	const cv::Mat &groundedBest = matricesForBest[SymData::GROUNDED_GLYPH_IDX];

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
			patchResult = cv::mean(patchColor);
		else
			cv::merge(channels, patchResult);

	} else { // grayscale result
		auto &params = best.params;
		if(!params.fg)
			params.computeFg(patch, symsSet[best.symIdx]);
		if(!params.bg)
			params.computeBg(patch, symsSet[best.symIdx]);
		double diff = *params.fg - *params.bg;

		if(abs(diff) < cfg.matchSettings().getBlankThreshold())
			patchResult = cv::mean(patch);
		else
			groundedBest.convertTo(patchResult, CV_8UC1, diff / dataOfBest.diffMinMax, *params.bg);
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

#ifndef UNIT_TESTING // UnitTesting project has a different implementation for this method
const vector<MatchAspect*>& MatchEngine::getAvailAspects() {
	static const vector<MatchAspect*> availAspects {
		&fgMatch, &bgMatch, &edgeMatch, &conMatch, &grMatch, &dirMatch, &lsMatch
	};
	return availAspects;
}
#endif // UNIT_TESTING

#ifdef _DEBUG
bool MatchEngine::usesUnicode() const {
	return fe.getEncoding().compare("UNICODE") == 0;
}
#endif // _DEBUG
