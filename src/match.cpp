/**********************************************************
 Project:     Pic2Sym
 File:        match.cpp

 Author:      Florin Tulba
 Created on:  2016-2-1
 
 Copyright (c) 2016 Florin Tulba
 **********************************************************/

#include "match.h"

using namespace std;
using namespace cv;

const wstring MatchParams::HEADER(L"#mcGlyphX,\t#mcGlyphY,\t#mcPatchX,\t#mcPatchY,\t"
								  L"#fg,\t#bg,\t#sdevFg,\t#sdevBg,\t#fg/all");
const wstring BestMatch::HEADER(wstring(L"#GlyphCode,\t#ChosenScore,\t") + MatchParams::HEADER);

wostream& operator<<(wostream &os, const MatchParams &mp) {
	os<<mp.mcGlyph.x<<",\t"<<mp.mcGlyph.y<<",\t"<<mp.mcPatch.x<<",\t"<<mp.mcPatch.y<<",\t"
		<<mp.fg<<",\t"<<mp.bg<<",\t"<<mp.sdevFg<<",\t"<<mp.sdevBg<<",\t"<<mp.glyphWeight;
	return os;
}

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

	os<<",\t"<<bm.score<<",\t"<<bm.params;
	return os;
}

Matcher::Matcher(unsigned fontSz, double smallGlyphsCoverage_) :
		fontSz_1(fontSz - 1U), smallGlyphsCoverage(smallGlyphsCoverage_),
		PREFERRED_RADIUS(3U * fontSz / 8.), MAX_MCS_OFFSET(sqrt(2) * fontSz_1),
		centerPatch(fontSz_1/2., fontSz_1/2.) {}

double Matcher::score(const Config &cfg) const {
	static const double SQRT2 = sqrt(2), TWO_SQRT2 = 2. - SQRT2;

	// for a histogram with just 2 equally large bins on 0 and 255^2 =>
	// mean = 255^2/2. = 32512.5; sdev = 255^2/(2*sqrt(2)) ~ 22989.81
	static const double SDEV_MAX = 255*255/(2.*SQRT2);
	static const double MIN_CONTRAST_BRIGHT = 2., // less contrast needed for bright tones
		MIN_CONTRAST_DARK = 5.; // more contrast needed for dark tones
	static const double CONTRAST_RATIO = (MIN_CONTRAST_DARK - MIN_CONTRAST_BRIGHT) / (2.*255);
	static const Point2d ORIGIN; // (0, 0)

	/////////////// CORRECTNESS FACTORS (Best Matching & Good Contrast) ///////////////
	// Range 0..1, acting just as penalty for bad standard deviations.
	// Closer to 1 for good sdev of fg;  tends to 0 otherwise.
	register const double fSdevFg = pow(1. - params.sdevFg / SDEV_MAX,
										cfg.get_kSdevFg());
	register const double fSdevBg = pow(1. - params.sdevBg / SDEV_MAX,
										cfg.get_kSdevBg());

	const double minimalContrast = // minimal contrast for the average brightness
		MIN_CONTRAST_BRIGHT + CONTRAST_RATIO * (params.fg + params.bg);
	// range 0 .. 255, best when large
	const double contrast = abs(params.bg - params.fg);
	// Encourage contrasts larger than minimalContrast:
	// <1 for low contrast;  1 for minimalContrast;  >1 otherwise
	register double fMinimalContrast = pow(contrast / minimalContrast,
										   cfg.get_kContrast());

	/////////////// SMOOTHNESS FACTORS (Similar gradient) ///////////////
	// best glyph location is when mc-s are near to each other
	// range 0 .. 1.42*fontSz_1, best when 0
	const double mcsOffset = norm(params.mcPatch - params.mcGlyph);
	// <=1 for mcsOffset >= PREFERRED_RADIUS;  >1 otherwise
	register const double fMinimalMCsOffset =
		pow(1. + (PREFERRED_RADIUS - mcsOffset) / MAX_MCS_OFFSET,
		cfg.get_kMCsOffset());

	const Point2d relMcPatch = params.mcPatch - centerPatch;
	const Point2d relMcGlyph = params.mcGlyph - centerPatch;

	// best gradient orientation when angle between mc-s is 0 => cos = 1
	// Maintaining the cosine of the angle is ok, as it stays near 1 for small angles.
	// -1..1 range, best when 1
	double cosAngleMCs = 0.;
	if(relMcGlyph != ORIGIN && relMcPatch != ORIGIN) // avoid DivBy0
		cosAngleMCs = relMcGlyph.dot(relMcPatch) /
		(norm(relMcGlyph) * norm(relMcPatch));

	// <=1 for |angleMCs| >= 45;  >1 otherwise
	// lessen the importance for small mcsOffset-s (< PREFERRED_RADIUS)
	register const double fMCsAngleLessThan45 = pow((1. + cosAngleMCs) * TWO_SQRT2,
													cfg.get_kCosAngleMCs() * min(mcsOffset, PREFERRED_RADIUS) / PREFERRED_RADIUS);

	/////////////// FANCINESS FACTOR (Larger glyphs) ///////////////
	// <=1 for glyphs considered small;   >1 otherwise
	register const double fGlyphWeight = pow(params.glyphWeight + 1. - smallGlyphsCoverage,
											 cfg.get_kGlyphWeight());

	const double result = fSdevFg * fSdevBg * fMinimalContrast *
		fMinimalMCsOffset * fMCsAngleLessThan45 * fGlyphWeight;

	return result;
}

BestMatch::BestMatch(bool isUnicode/* = true*/) : unicode(isUnicode) {
	reset();
}

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

void BestMatch::reset() {
	score = numeric_limits<double>::lowest();
	symIdx = UINT_MAX; // no best yet
	symCode = 32UL; // Space
}

void BestMatch::reset(double score_, unsigned symIdx_, unsigned long symCode_,
					  const MatchParams &params_) {
	score = score_;
	symIdx = symIdx_;
	symCode = symCode_;
	params = params_;
}

