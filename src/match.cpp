/**********************************************************
 Project:     Pic2Sym
 File:        match.cpp

 Author:      Florin Tulba
 Created on:  2016-2-1
 
 Copyright (c) 2016 Florin Tulba
 **********************************************************/

#include "match.h"

#include <numeric>

#include <boost/optional/optional_io.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;
using namespace boost;

namespace {
	// Conversion PixMapSym -> Mat of type double with range [0..1] instead of [0..255]
	Mat toMat(const PixMapSym &pms, unsigned fontSz) {
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

	pair<double, double> averageFgBg(const Mat &patch, const Mat &fgMask, const Mat &bgMask) {
		const Scalar miuFg = mean(patch, fgMask),
			miuBg = mean(patch, bgMask);
		return make_pair(*miuFg.val, *miuBg.val);
	}

}

const double CachedData::sdevMax = 127.5;

#ifdef _DEBUG

#define comma L",\t"

const wstring MatchParams::HEADER(L"#mcGlyphX" comma L"#mcGlyphY" comma
								  L"#mcPatchX" comma L"#mcPatchY" comma
								  L"#fg" comma L"#bg" comma
								  L"#sdevFg" comma L"#sdevEdge" comma L"#sdevBg" comma
								  L"#rho");
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

#undef comma

BestMatch::BestMatch(bool isUnicode/* = true*/) : unicode(isUnicode) {}

BestMatch& BestMatch::operator=(const BestMatch &other) {
	if(this != &other) {
		score = other.score;
		symIdx = other.symIdx;
		symCode = other.symCode;
		*const_cast<bool*>(&unicode) = other.unicode;
		params = other.params;
	}
	return *this;
}
#endif // _DEBUG

SymData::SymData(const unsigned long code_, const double pixelSum_,
				 const Point2d &mc_, const MatArray &symAndMasks_) :
		code(code_), pixelSum(pixelSum_), mc(mc_), symAndMasks(symAndMasks_) {
}

void MatchParams::resetSymData() {
	mcGlyph = none;
	glyphWeight = fg = bg = sdevFg = sdevBg = sdevEdge = none;
}

void MatchParams::computeFg(const Mat &patch, const SymData &symData) {
	if(fg)
		return;
	fg = *mean(patch, symData.symAndMasks[SymData::FG_MASK_IDX]).val;
}

void MatchParams::computeBg(const Mat &patch, const SymData &symData) {
	if(bg)
		return;
	bg = *mean(patch, symData.symAndMasks[SymData::BG_MASK_IDX]).val;
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
}

void MatchParams::computeSdevFg(const Mat &patch, const SymData &symData) {
	computeSdev(patch, symData.symAndMasks[SymData::FG_MASK_IDX], fg, sdevFg);
}

void MatchParams::computeSdevBg(const Mat &patch, const SymData &symData) {
	computeSdev(patch, symData.symAndMasks[SymData::BG_MASK_IDX], bg, sdevBg);
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

	computeFg(patch, symData);
	computeBg(patch, symData);

	const double diffFgBg = fg.value() - bg.value();
	if(diffFgBg == 0.) {
		sdevEdge = norm(patch - bg.value(), NORM_L2, edgeMask) / sqrt(cnz);
		return;
	}

	const Mat approximationOfPatch =
		symData.symAndMasks[SymData::GLYPH_IDX] * diffFgBg + bg.value();

	sdevEdge = norm(patch, approximationOfPatch, NORM_L2, edgeMask) / sqrt(cnz);
}

void MatchParams::computeRhoApproxSym(const SymData &symData, const CachedData &cachedData) {
	if(glyphWeight)
		return;

	glyphWeight = symData.pixelSum / cachedData.sz2;
}

void MatchParams::computeMcPatch(const Mat &patch, const CachedData &cachedData) {
	if(mcPatch)
		return;

	const double patchSum = *sum(patch).val;
	Mat temp, temp1;
	reduce(patch, temp, 0, CV_REDUCE_SUM);	// sum all rows
	reduce(patch, temp1, 1, CV_REDUCE_SUM);	// sum all columns

	mcPatch = Point2d(temp.dot(cachedData.consec), temp1.t().dot(cachedData.consec)) / patchSum;
}

void MatchParams::computeMcApproxSym(const Mat &patch, const SymData &symData,
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
		mcGlyph = (k * symData.mc + Point2d(delta, delta)) / denominator;
}

void BestMatch::update(double score_, unsigned symIdx_, unsigned long symCode_,
					  const MatchParams &params_) {
	score = score_;
	symIdx = symIdx_;
	symCode = symCode_;
	params = params_;
}

double FgMatch::assessMatch(const Mat &patch,
							const SymData &symData,
							MatchParams &mp) const {
	mp.computeSdevFg(patch, symData);
	return pow(1. - mp.sdevFg.value()/CachedData::sdevMax, k);
}

double BgMatch::assessMatch(const Mat &patch,
							const SymData &symData,
							MatchParams &mp) const {
	mp.computeSdevBg(patch, symData);
	return pow(1. - mp.sdevBg.value()/CachedData::sdevMax, k);
}

double EdgeMatch::assessMatch(const Mat &patch,
							const SymData &symData,
							MatchParams &mp) const {
	mp.computeSdevEdge(patch, symData);
	return pow(1. - mp.sdevEdge.value()/CachedData::sdevMax, k);
}

double BetterContrast::assessMatch(const Mat &patch,
							const SymData &symData,
							MatchParams &mp) const {
	static const double MIN_CONTRAST_BRIGHT = 2., // less contrast needed for bright tones
		MIN_CONTRAST_DARK = 5.; // more contrast needed for dark tones
	static const double CONTRAST_RATIO = (MIN_CONTRAST_DARK - MIN_CONTRAST_BRIGHT) / (2.*255);
	
	mp.computeFg(patch, symData);
	mp.computeBg(patch, symData);
	
	const double minimalContrast = // minimal contrast for the average brightness
		MIN_CONTRAST_BRIGHT + CONTRAST_RATIO * (mp.fg.value() + mp.bg.value());
	// range 0 .. 255, best when large
	const double contrast = abs(mp.fg.value() - mp.bg.value());
	// Encourage contrasts larger than minimalContrast:
	// <1 for low contrast;  1 for minimalContrast;  >1 otherwise
	return pow(contrast / minimalContrast, k);
}

double GravitationalSmoothness::assessMatch(const Mat &patch,
							const SymData &symData,
							MatchParams &mp) const {
	mp.computeMcPatch(patch, cachedData);
	mp.computeMcApproxSym(patch, symData, cachedData);

	// best glyph location is when mc-s are near to each other
	// range 0 .. mcDistMax = 1.42*fontSz_1, best when 0
	const double mcsOffset = norm(mp.mcPatch.value() - mp.mcGlyph.value());
	// <=1 for mcsOffset >= preferredMaxMcDist;  >1 otherwise
	
	return pow(1. + (cachedData.preferredMaxMcDist - mcsOffset)/cachedData.mcDistMax, k);
}

double DirectionalSmoothness::assessMatch(const Mat &patch,
							const SymData &symData,
							MatchParams &mp) const {
	static const double SQRT2 = sqrt(2), TWO_SQRT2 = 2. - SQRT2;
	static const Point2d ORIGIN; // (0, 0)

	mp.computeMcPatch(patch, cachedData);
	mp.computeMcApproxSym(patch, symData, cachedData);

	// best glyph location is when mc-s are near to each other
	// range 0 .. mcDistMax = 1.42*fontSz_1, best when 0
	const double mcsOffset = norm(mp.mcPatch.value() - mp.mcGlyph.value());
	// <=1 for mcsOffset >= preferredMaxMcDist;  >1 otherwise

	const Point2d relMcPatch = mp.mcPatch.value() - cachedData.patchCenter;
	const Point2d relMcGlyph = mp.mcGlyph.value() - cachedData.patchCenter;

	// best gradient orientation when angle between mc-s is 0 => cos = 1
	// Maintaining the cosine of the angle is ok, as it stays near 1 for small angles.
	// -1..1 range, best when 1
	double cosAngleMCs = 0.;
	if(relMcGlyph != ORIGIN && relMcPatch != ORIGIN) // avoid DivBy0
		cosAngleMCs = relMcGlyph.dot(relMcPatch) / (norm(relMcGlyph) * norm(relMcPatch));

	// <=1 for |angleMCs| >= 45;  >1 otherwise
	// lessen the importance for small mcsOffset-s (< preferredMaxMcDist)
	return pow((1. + cosAngleMCs) * TWO_SQRT2,
			   k * min(mcsOffset, cachedData.preferredMaxMcDist) / cachedData.preferredMaxMcDist);
}

double LargerSym::assessMatch(const Mat &patch,
							const SymData &symData,
							MatchParams &mp) const {
	mp.computeRhoApproxSym(symData, cachedData);

	// <=1 for glyphs considered small;   >1 otherwise
	return pow(mp.glyphWeight.value() + 1. - cachedData.smallGlyphsCoverage, k);
}

void CachedData::update(unsigned sz_, const FontEngine &fe_) {
	sz = sz_;
	sz_1 = sz - 1U;
	sz2 = (double)sz*sz;

	preferredMaxMcDist = 3.*sz/8;
	mcDistMax = sz_1*sqrt(2);
	patchCenter = Point2d(sz_1, sz_1) / 2;

	consec = Mat(1, sz, CV_64FC1);
	iota(consec.begin<double>(), consec.end<double>(), 0.);

	smallGlyphsCoverage = fe_.smallGlyphsCoverage();
}


MatchEngine::MatchEngine(Controller &ctrler_, const Config &cfg_, FontEngine &fe_) :
	cfg(cfg_), fe(fe_),
	fgMatch(cachedData, cfg_), bgMatch(cachedData, cfg_), edgeMatch(cachedData, cfg_),
	conMatch(cachedData, cfg_), grMatch(cachedData, cfg_), dirMatch(cachedData, cfg_),
	lsMatch(cachedData, cfg_) {
}

#ifdef _DEBUG
bool MatchEngine::usesUnicode() const {
	return fe.getEncoding().compare("UNICODE") == 0;
}
#endif // _DEBUG

string MatchEngine::getIdForSymsToUse() {
	const unsigned sz = cfg.getFontSz();
	if(!Config::isFontSizeOk(sz)) {
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

	static const double STILL_BG = .025,			// darkest shades
		STILL_FG = 1. - STILL_BG, // brightest shades
		EPS = 1e-6;
	symsSet.clear();
	symsSet.reserve(fe.symsSet().size());

	double minVal, maxVal;
	const unsigned sz = cfg.getFontSz();
	const int szGlyph[] = { 2, sz, sz },
		szMasks[] = { 4, sz, sz };
	for(const auto &pms : fe.symsSet()) {
		Mat negGlyph, edgeMask;
		const Mat glyph = toMat(pms, sz);
		glyph.convertTo(negGlyph, CV_8UC1, -255., 255.);

		// for very small fonts, minVal might be > 0 and maxVal might be < 255
		minMaxIdx(glyph, &minVal, &maxVal);
		const Mat fgMask = (glyph > (minVal + STILL_FG * (maxVal-minVal))),
				bgMask = (glyph < (minVal + STILL_BG * (maxVal-minVal)));
		inRange(glyph, minVal+EPS, maxVal-EPS, edgeMask);

		symsSet.emplace_back(pms.symCode, pms.glyphSum, pms.mc, SymData::MatArray { {
								glyph,			// GLYPH_IDX
								fgMask,			// FG_MASK_IDX 
								bgMask,			// BG_MASK_IDX
								edgeMask,		// EDGE_MASK_IDX
								negGlyph		// NEG_GLYPH_IDX
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
	static const vector<MatchAspect*> availAspects {
		&fgMatch, &bgMatch, &edgeMatch, &conMatch, &grMatch, &dirMatch, &lsMatch
	};
	
	updateSymbols();

	cachedData.update(cfg.getFontSz(), fe);

	aspects.clear();
	for(auto pAspect : availAspects)
		if(pAspect->enabled())
			aspects.push_back(pAspect);
}

Mat MatchEngine::approxPatch(const Mat &patch_, BestMatch &best) {
	const bool isColor = (patch_.channels() > 1);
	Mat patchColor, patch, patchResult;
	if(isColor) {
		patchColor = patch_;
		cvtColor(patchColor, patch, COLOR_RGB2GRAY);
	} else patch = patch_;
	patch.convertTo(patch, CV_64FC1);

	findBestMatch(patch, best);

	const auto &matricesForBest = symsSet[best.symIdx].symAndMasks;
	const Mat &bestGlyph = matricesForBest[SymData::GLYPH_IDX];

	if(isColor) {
		const Mat &fgMask = matricesForBest[SymData::FG_MASK_IDX],
				&bgMask = matricesForBest[SymData::BG_MASK_IDX];

		vector<Mat> channels;
		split(patchColor, channels);

		double miuFg, miuBg, newDiff, diffFgBg = 0.;
		for(auto &ch : channels) {
			ch.convertTo(ch, CV_64FC1); // processing double values

			tie(miuFg, miuBg) = averageFgBg(ch, fgMask, bgMask);
			newDiff = miuFg - miuBg;

			bestGlyph.convertTo(ch, CV_8UC1, newDiff, miuBg);

			diffFgBg += abs(newDiff);
		}

		if(diffFgBg < 3.*cfg.getBlankThreshold())
			patchResult = mean(patchColor);
		else
			merge(channels, patchResult);

	} else { // grayscale result
		if(abs(*best.params.fg - *best.params.bg) < cfg.getBlankThreshold())
			patchResult = mean(patch);
		else
			bestGlyph.convertTo(patchResult, CV_8UC1,
							*best.params.fg - *best.params.bg,
							*best.params.bg);
	}
	return patchResult;
}

void MatchEngine::findBestMatch(const Mat &patch, BestMatch &best) {
	MatchParams mp;
	unsigned idx = 0U;
	for(const auto &symData : symsSet) {
		double score = 1.;
		for(auto pAspect : aspects)
			score *= pAspect->assessMatch(patch, symData, mp);

		if(score > best.score)
			best.update(score, idx, symData.code, mp);

		mp.resetSymData();
		++idx;
	}
}
