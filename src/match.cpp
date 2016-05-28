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

#include "controller.h"

#include <numeric>

#include <boost/optional/optional_io.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace boost;

namespace {
	const double EPS = 1e-6;

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

const double CachedData::sdevMaxFgBg = 127.5;
const double CachedData::sdevMaxEdge = 255.;

#if defined _DEBUG || defined UNIT_TESTING

#	define comma L",\t"

const wstring MatchParams::HEADER(L"#ssim" comma
								  L"#sdFg" comma L"#sdEdge" comma L"#sdBg" comma
								  L"#fg" comma L"#bg" comma
								  L"#mcPaX" comma L"#mcPaY" comma
								  L"#mcPX" comma L"#mcPY" comma
								  L"#density");

wostream& operator<<(wostream &os, const MatchParams &mp) {
	os<<mp.ssim<<comma
		<<mp.sdevFg<<comma<<mp.sdevEdge<<comma<<mp.sdevBg<<comma
		<<mp.fg<<comma<<mp.bg<<comma;

	if(mp.mcPatchApprox)
		os<<mp.mcPatchApprox->x<<comma<<mp.mcPatchApprox->y<<comma;
	else
		os<<none<<comma<<none<<comma;

	if(mp.mcPatch)
		os<<mp.mcPatch->x<<comma<<mp.mcPatch->y<<comma;
	else
		os<<none<<comma<<none<<comma;

	os<<mp.symDensity;

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

void MatchParams::computeBlurredPatch(const cv::Mat &patch) {
	if(blurredPatch)
		return;

	cv::Mat blurredPatch_;
	cv::GaussianBlur(patch, blurredPatch_,
					 StructuralSimilarity::WIN_SIZE, StructuralSimilarity::SIGMA, 0.,
					 cv::BORDER_REPLICATE);
	blurredPatch = blurredPatch_;
}

void MatchParams::computeBlurredPatchSq(const cv::Mat &patch) {
	if(blurredPatchSq)
		return;

	computeBlurredPatch(patch);
	blurredPatchSq = blurredPatch.value().mul(blurredPatch.value());
}

void MatchParams::computeVariancePatch(const cv::Mat &patch) {
	if(variancePatch)
		return;

	computeBlurredPatchSq(patch);

	cv::Mat variancePatch_;
	cv::GaussianBlur(patch.mul(patch), variancePatch_,
					 StructuralSimilarity::WIN_SIZE, StructuralSimilarity::SIGMA, 0.,
					 cv::BORDER_REPLICATE);
	variancePatch_ -= blurredPatchSq.value();
	variancePatch = variancePatch_;
}

void MatchParams::computeSsim(const cv::Mat &patch, const SymData &symData) {
	if(ssim)
		return;

	cv::Mat covariance, ssimMap;

	computeVariancePatch(patch);
	computePatchApprox(patch, symData);
	const cv::Mat &approxPatch = patchApprox.value();

	// Saving 2 calls to GaussianBlur each time current symbol is compared to a patch:
	// Blur and Variance of the approximated patch are computed based on the blur and variance
	// of the grounded version of the original symbol
	const double diffRatio = contrast.value() / symData.diffMinMax;
	const cv::Mat blurredPatchApprox = bg.value() + diffRatio *
										symData.symAndMasks[SymData::BLURRED_GR_SYM_IDX],
				blurredPatchApproxSq = blurredPatchApprox.mul(blurredPatchApprox),
				variancePatchApprox = diffRatio * diffRatio *
										symData.symAndMasks[SymData::VARIANCE_GR_SYM_IDX];

#ifdef _DEBUG // checking the simplifications mentioned above
	double minVal, maxVal;
	cv::Mat blurredPatchApprox_, variancePatchApprox_; // computed by brute-force

	cv::GaussianBlur(approxPatch, blurredPatchApprox_,
					 StructuralSimilarity::WIN_SIZE, StructuralSimilarity::SIGMA, 0.,
					 cv::BORDER_REPLICATE);
	minMaxIdx(blurredPatchApprox - blurredPatchApprox_, &minVal, &maxVal); // math vs. brute-force
	assert(abs(minVal) < EPS);
	assert(abs(maxVal) < EPS);

	cv::GaussianBlur(approxPatch.mul(approxPatch), variancePatchApprox_,
					 StructuralSimilarity::WIN_SIZE, StructuralSimilarity::SIGMA, 0.,
					 cv::BORDER_REPLICATE);
	variancePatchApprox_ -= blurredPatchApproxSq;
	minMaxIdx(variancePatchApprox - variancePatchApprox_, &minVal, &maxVal); // math vs. brute-force
	assert(abs(minVal) < EPS);
	assert(abs(maxVal) < EPS);
#endif // checking the simplifications mentioned above

	const cv::Mat productMats = patch.mul(approxPatch),
				productBlurredMats = blurredPatch.value().mul(blurredPatchApprox);
	cv::GaussianBlur(productMats, covariance,
					 StructuralSimilarity::WIN_SIZE, StructuralSimilarity::SIGMA, 0.,
					 cv::BORDER_REPLICATE);
	covariance -= productBlurredMats;

	const cv::Mat numerator = (2.*productBlurredMats + StructuralSimilarity::C1).
					mul(2.*covariance + StructuralSimilarity::C2),
		denominator = (blurredPatchSq.value() + blurredPatchApproxSq + StructuralSimilarity::C1).
					mul(variancePatch.value() + variancePatchApprox + StructuralSimilarity::C2);

	cv::divide(numerator, denominator, ssimMap);
	ssim = *cv::mean(ssimMap).val;
	assert(abs(*ssim) < 1.+EPS);
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

const cv::Size StructuralSimilarity::WIN_SIZE(11, 11);
const double StructuralSimilarity::SIGMA = 1.5;
const double StructuralSimilarity::C1 = .01*.01*255.*255.; // (.01*255)^2
const double StructuralSimilarity::C2 = .03*.03*255.*255.; // (.03*255)^2

/*
Match aspect implementing the method described in https://ece.uwaterloo.ca/~z70wang/research/ssim .

Downsampling was not used, as the results normally get inspected by
enlarging the regions of interest.
*/
double StructuralSimilarity::assessMatch(const cv::Mat &patch,
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
	return pow(1. - mp.sdevFg.value() / CachedData::sdevMaxFgBg, k);
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
	return pow(1. - mp.sdevBg.value()/CachedData::sdevMaxFgBg, k);
}

double EdgeMatch::assessMatch(const cv::Mat &patch,
							  const SymData &symData,
							  MatchParams &mp) const {
	mp.computeSdevEdge(patch, symData);

	// Returned value discourages large std. devs.
	// For sdev =     0 (min) => returns 1 no matter k
	// For sdev =	255 (max) => returns 0 no matter k (For k=0, this matching aspect is disabled)
	// For other sdev-s       =>
	//		returns closer to 1 for k in (0..1)
	//		returns sdev for k=1
	//		returns closer to 0 for k>1 (Large k => higher penalty for large sdev-s)
	return pow(1. - mp.sdevEdge.value()/CachedData::sdevMaxEdge, k);
}

double BetterContrast::assessMatch(const cv::Mat &patch,
								   const SymData &symData,
								   MatchParams &mp) const {
	mp.computeContrast(patch, symData);
	
	// Encourages larger contrasts:
	// 0 for no contrast; 1 for max contrast (255)
	return pow( abs(mp.contrast.value()) / 255., k);
}

double GravitationalSmoothness::assessMatch(const cv::Mat &patch,
											const SymData &symData,
											MatchParams &mp) const {
	mp.computeMcsOffset(patch, symData, cachedData);

	// Discourages mcsOffset larger than preferredMaxMcDist:
	//		returns 1 for mcsOffset == preferredMaxMcDist, no matter k
	//		returns 0 for mcsOffset == 1.42*(fontSz-1), no matter k (k > 0)
	//		returns in (0..1) for mcsOffset in (preferredMaxMcDist .. 1.42*(fontSz-1) )
	//		returns > 1 for mcsOffset < preferredMaxMcDist
	// Larger k induces larger penalty for large mcsOffset and
	// also larger reward for small mcsOffset
	return pow(1. + (cachedData.preferredMaxMcDist - mp.mcsOffset.value()) /
			   cachedData.complPrefMaxMcDist, k);
}

double DirectionalSmoothness::assessMatch(const cv::Mat &patch,
										  const SymData &symData,
										  MatchParams &mp) const {
	static const double SQRT2 = sqrt(2), TWOmSQRT2 = 2. - SQRT2;
	static const cv::Point2d ORIGIN; // (0, 0)

	mp.computeMcsOffset(patch, symData, cachedData);

	const cv::Point2d relMcPatch = mp.mcPatch.value() - cachedData.patchCenter;
	const cv::Point2d relMcGlyph = mp.mcPatchApprox.value() - cachedData.patchCenter;

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
			   k * min(mp.mcsOffset.value(), cachedData.preferredMaxMcDist) /
			   cachedData.preferredMaxMcDist);
}

double LargerSym::assessMatch(const cv::Mat&,
							  const SymData &symData,
							  MatchParams &mp) const {
	mp.computeSymDensity(symData, cachedData);

	// Encourages approximations with symbols filling at least x% of their box.
	// The threshold x is provided by smallGlyphsCoverage.
	// Returns < 1 for glyphs under threshold;   >= 1 otherwise
	return pow(mp.symDensity.value() + 1. - cachedData.smallGlyphsCoverage, k);
}

void CachedData::useNewSymSize(unsigned sz_) {
	sz = sz_;
	sz_1 = sz - 1U;
	sz2 = (double)sz * sz;

	preferredMaxMcDist = sz / 8.;
	complPrefMaxMcDist = sz_1 * sqrt(2) - preferredMaxMcDist;
	patchCenter = cv::Point2d(sz_1, sz_1) / 2.;

	consec = cv::Mat(1, sz, CV_64FC1);
	iota(consec.begin<double>(), consec.end<double>(), 0.);
}

void CachedData::update(unsigned sz_, const FontEngine &fe_) {
	useNewSymSize(sz_);

	smallGlyphsCoverage = fe_.smallGlyphsCoverage();
}


MatchEngine::MatchEngine(const Settings &cfg_, FontEngine &fe_) :
	cfg(cfg_), fe(fe_), strSimMatch(cachedData, cfg_.matchSettings()),
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
					STILL_FG = 1. - STILL_BG;	// brightest shades
	symsSet.clear();
	symsSet.reserve(fe.symsSet().size());

	double minVal, maxVal;
	const unsigned sz = cfg.symSettings().getFontSz();
	const int szGlyph[] = { 2, sz, sz },
		szMasks[] = { 4, sz, sz };
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
								edgeMask,				// EDGE_MASK_IDX
								negGlyph,				// NEG_SYM_IDX
								groundedGlyph,			// GROUNDED_SYM_IDX
								blurOfGroundedGlyph,	// BLURRED_GR_SYM_IDX
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
	for(auto pAspect : getAvailAspects())
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

#ifndef UNIT_TESTING // UnitTesting project has a different implementation for this method
const vector<MatchAspect*>& MatchEngine::getAvailAspects() {
	static const vector<MatchAspect*> availAspects {
		&fgMatch, &bgMatch, &edgeMatch, &conMatch, &grMatch, &dirMatch, &lsMatch, &strSimMatch
	};
	return availAspects;
}
#endif // UNIT_TESTING

#ifdef _DEBUG
bool MatchEngine::usesUnicode() const {
	return fe.getEncoding().compare("UNICODE") == 0;
}
#endif // _DEBUG
