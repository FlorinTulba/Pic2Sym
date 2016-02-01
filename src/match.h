/**********************************************************
 Project:     Pic2Sym
 File:        match.h

 Author:      Florin Tulba
 Created on:  2016-2-1
 
 Copyright (c) 2016 Florin Tulba
 **********************************************************/

#ifndef H_MATCH
#define H_MATCH

#include "config.h"

#include <string>
#include <iostream>

#include <opencv2/core/core.hpp>

// Holds relevant data during patch&glyph matching
struct MatchParams {
	static const std::wstring HEADER;	// table header when values are serialized

	cv::Point2d mcPatch;				// mass center for the patch
	cv::Point2d mcGlyph;				// glyph's mass center
	double glyphWeight;					// % of the box covered by the glyph (0..1)
	double fg, bg;						// color for fg / bg (range 0..255)

	// standard deviations for fg / bg
	// 0 .. 255^2/(2*sqrt(2)); best when 0
	double sdevFg, sdevBg;

	friend std::wostream& operator<<(std::wostream &os, const MatchParams &mp);
};

/*
Class for assessing a match based on various criteria.
*/
class Matcher {
	const unsigned fontSz_1;		// size of the font - 1
	const double smallGlyphsCoverage; // max ratio of glyph area / containing area for small symbols
	const double PREFERRED_RADIUS;	// allowed distance between the mc-s of patch & chosen glyph
	const double MAX_MCS_OFFSET;	// max distance possible between the mc-s of patch & chosen glyph
	// happens for the extremes of a diagonal

	const cv::Point2d centerPatch;		// center of the patch

public:
	MatchParams params;

	Matcher(unsigned fontSz, double smallGlyphsCoverage_);

	/*
	Returns a larger positive value for better correlations.
	Good correlation is a subjective matter.

	Several considered factors are mentioned below.
	* = mandatory;  + = nice to have

	A) Separately, each patch needs to:
	* 1. minimize the difference between the selected foreground glyph
	and the covered part from the original patch

	* 2. minimize the difference between the remaining background around the selected glyph
	and the corresponding part from the original patch

	* 3. have enough contrast between foreground & background of the selected glyph.

	B) Together, the patches should also preserve region's gradients:
	* 1. Prefer glyphs that respect the gradient of their corresponding patch.

	+ 2. Consider the gradient within a slightly extended patch
	Gain: Smoothness;		Drawback: Complexity

	C) Balance the accuracy with more artistic aspects:
	+ 1. use largest possible 'matching' glyph, not just dots, quotes, commas


	Points A1&2 minimize the standard deviation (or a similar measure) on each region.

	Point A3 encourages larger differences between the means of the fg & bg.

	Point B1 ensures the weight centers of the glyph and patch are close to each other
	as distance and as direction.

	Point C1 favors larger glyphs.

	The remaining points might be addressed in a future version.
	*/
	double score(const Config &cfg) const;
};

// Holds the best grayscale match found at a given time
struct BestMatch {
	double score;			// score of the best
	unsigned symIdx;		// index within vector<PixMapSym>
	unsigned long symCode;	// glyph code
	const bool unicode;		// is the charmap in Unicode?

	MatchParams params;		// parameters of the match for the best glyph

	BestMatch(bool isUnicode = true);
	BestMatch(const BestMatch&) = default;
	BestMatch& operator=(const BestMatch &other);

	void reset();

	void reset(double score_, unsigned symIdx_, unsigned long symCode_, const MatchParams &params_);

	static const std::wstring HEADER;

	friend std::wostream& operator<<(std::wostream &os, const BestMatch &bm);
};

#endif