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
#include "fontEngine.h"

#include <vector>
#include <array>
#include <string>
#include <iostream>

#include <boost/optional/optional.hpp>
#include <opencv2/core/core.hpp>

// Most symbol information
struct SymData final {
	const unsigned long code;		// the code of the symbol
	const double minVal = 0.;			// the value of darkest pixel, range 0..1
	const double diffMinMax = 1.;		// difference between brightest and darkest pixels, each in 0..1
	const double pixelSum;			// sum of the values of the pixels, each in 0..1
	const cv::Point2d &mc;			// mass center of the symbol given original fg & bg

	enum { // indices of each matrix type within a MatArray object
		GLYPH_IDX, FG_MASK_IDX, BG_MASK_IDX, EDGE_MASK_IDX, NEG_GLYPH_IDX, GROUNDED_GLYPH_IDX,
		MATRICES_COUNT // keep this last and don't use it as index in MatArray objects
	};

	// For each symbol from cmap, there'll be several additional helpful matrices to store
	// along with the one for the given glyph. The enum from above should be used for selection.
	typedef std::array< const cv::Mat, MATRICES_COUNT > MatArray;

	const MatArray symAndMasks;		// symbol + other matrices & masks

	SymData(unsigned long code_, double minVal_, double diffMinMax_, double pixelSum_,
			const cv::Point2d &mc_, const MatArray &symAndMasks_);
};

struct CachedData; // forward declaration

// Holds relevant data during patch&glyph matching
struct MatchParams final {
	// This param is computed only once, if necessary, when approximating the patch
	boost::optional<cv::Point2d> mcPatch;	// mass center for the patch

	// These params are evaluated for each symbol compared to the patch
	boost::optional<cv::Point2d> mcGlyph;	// glyph's mass center
	boost::optional<double> glyphWeight;	// % of the box covered by the glyph (0..1)
	boost::optional<double> fg, bg;			// color for fg / bg (range 0..255)

	// standard deviations for fg / bg / contour
	// 0 .. 127.5; best when 0
	boost::optional<double> sdevFg, sdevBg, sdevEdge;

	// Prepare for next symbol to match against patch
	void resetSymData(); // reset everything except mcPatch

	// Methods for computing each field
	void computeFg(const cv::Mat &patch, const SymData &symData);
	void computeBg(const cv::Mat &patch, const SymData &symData);
	void computeSdevFg(const cv::Mat &patch, const SymData &symData);
	void computeSdevBg(const cv::Mat &patch, const SymData &symData);
	void computeSdevEdge(const cv::Mat &patch, const SymData &symData);
	void computeRhoApproxSym(const SymData &symData, const CachedData &cachedData);
	void computeMcPatch(const cv::Mat &patch, const CachedData &cachedData);
	void computeMcApproxSym(const cv::Mat &patch, const SymData &symData,
							const CachedData &cachedData);

#if defined _DEBUG || defined UNIT_TESTING // Next members are necessary for logging
	static const std::wstring HEADER; // table header when values are serialized
	friend std::wostream& operator<<(std::wostream &os, const MatchParams &mp);
#endif

#ifndef UNIT_TESTING // UnitTesting project will still have following methods as public
private:
#endif
	// Both computeFg and computeBg simply call this
	static void computeMean(const cv::Mat &patch, const cv::Mat &mask, boost::optional<double> &miu);

	// Both computeSdevFg and computeSdevBg simply call this
	static void computeSdev(const cv::Mat &patch, const cv::Mat &mask,
					 boost::optional<double> &miu, boost::optional<double> &sdev);
};

// Holds the best grayscale match found at a given time
struct BestMatch final {
	unsigned symIdx = UINT_MAX;			// index within vector<PixMapSym>
	unsigned long symCode = ULONG_MAX;	// glyph code

	double score = std::numeric_limits<double>::lowest(); // score of the best match

	MatchParams params;		// parameters of the match for the best approximating glyph

	// called when finding a better match
	void update(double score_, unsigned symIdx_, unsigned long symCode_,
				const MatchParams &params_);

#if defined _DEBUG || defined UNIT_TESTING // Next members are necessary for logging
	static const std::wstring HEADER;
	friend std::wostream& operator<<(std::wostream &os, const BestMatch &bm);

	// Unicode symbols are logged in symbol format, while other encodings log their code
	const bool unicode;					// is the charmap in Unicode?

	BestMatch(bool isUnicode = true);
	BestMatch(const BestMatch&) = default;
	BestMatch& operator=(const BestMatch &other);
#endif
};

// Interface providing assessMatch method for MatchAspect classes and also for MatchEngine
struct IMatch abstract {
	// scores the match between a gray patch and a symbol
	virtual double assessMatch(const cv::Mat &patch,
							   const SymData &symData,
							   MatchParams &mp) const = 0;
	virtual ~IMatch() = 0 {}
};

// Base class for all considered aspects of matching.
class MatchAspect abstract : public IMatch {
protected:
	const CachedData &cachedData; // cached information from matching engine
	const double &k; // cached coefficient from cfg, corresponding to current aspect

public:
	MatchAspect(const CachedData &cachedData_, const double &k_) :
		cachedData(cachedData_), k(k_) {}

	// all aspects that are configured with coefficients > 0 are enabled; those with 0 are disabled
	bool enabled() const { return k > 0.; }
};

// Selecting a symbol with the scene underneath it as uniform as possible
class FgMatch final : public MatchAspect {
public:
	FgMatch(const CachedData &cachedData_, const Config &cfg) :
		MatchAspect(cachedData_, cfg.get_kSdevFg()) {}

	// scores the match between a gray patch and a symbol based on current aspect
	double assessMatch(const cv::Mat &patch,
					   const SymData &symData,
					   MatchParams &mp) const override; // IMatch override
};

// Aspect ensuring more uniform background scene around the selected symbol
class BgMatch final : public MatchAspect {
public:
	BgMatch(const CachedData &cachedData_, const Config &cfg) :
		MatchAspect(cachedData_, cfg.get_kSdevBg()) {}

	// scores the match between a gray patch and a symbol based on current aspect
	double assessMatch(const cv::Mat &patch,
					   const SymData &symData,
					   MatchParams &mp) const override; // IMatch override
};

// Aspect ensuring the edges of the selected symbol seem to appear also on the patch
class EdgeMatch final : public MatchAspect {
public:
	EdgeMatch(const CachedData &cachedData_, const Config &cfg) :
		MatchAspect(cachedData_, cfg.get_kSdevEdge()) {}

	// scores the match between a gray patch and a symbol based on current aspect
	double assessMatch(const cv::Mat &patch,
					   const SymData &symData,
					   MatchParams &mp) const override; // IMatch override
};

// Discouraging barely visible symbols
class BetterContrast final : public MatchAspect {
public:
	BetterContrast(const CachedData &cachedData_, const Config &cfg) :
		MatchAspect(cachedData_, cfg.get_kContrast()) {}

	// scores the match between a gray patch and a symbol based on current aspect
	double assessMatch(const cv::Mat &patch,
					   const SymData &symData,
					   MatchParams &mp) const override; // IMatch override
};

// Aspect concentrating on where's the center of gravity of the patch & its approximation
class GravitationalSmoothness final : public MatchAspect {
public:
	GravitationalSmoothness(const CachedData &cachedData_, const Config &cfg) :
		MatchAspect(cachedData_, cfg.get_kMCsOffset()) {}

	// scores the match between a gray patch and a symbol based on current aspect
	double assessMatch(const cv::Mat &patch,
					   const SymData &symData,
					   MatchParams &mp) const override; // IMatch override
};

// Aspect encouraging more accuracy while approximating the direction of the patch
class DirectionalSmoothness final : public MatchAspect {
public:
	DirectionalSmoothness(const CachedData &cachedData_, const Config &cfg) :
		MatchAspect(cachedData_, cfg.get_kCosAngleMCs()) {}

	// scores the match between a gray patch and a symbol based on current aspect
	double assessMatch(const cv::Mat &patch,
					   const SymData &symData,
					   MatchParams &mp) const override; // IMatch override
};

// Match aspect concerning user's preference for larger symbols as approximations
class LargerSym final : public MatchAspect {
public:
	LargerSym(const CachedData &cachedData_, const Config &cfg) :
		MatchAspect(cachedData_, cfg.get_kGlyphWeight()) {}

	// scores the match between a gray patch and a symbol based on current aspect
	double assessMatch(const cv::Mat &patch,
					   const SymData &symData,
					   MatchParams &mp) const override; // IMatch override
};

class MatchEngine; // forward declaration

// cached data for computing match parameters and evaluating match aspects
struct CachedData final {
	/*
	Max possible std dev = 127.5.
	Happens for an error matrix with a histogram with 2 equally large bins on 0 and 255.
	In that case, the mean is 127.5 and the std dev is:
	sqrt( ((-127.5)^2 * sz^2/2 + 127.5^2 * sz^2/2) /sz^2) = 127.5
	*/
	static const double sdevMax;

	unsigned sz;				// symbol size
	unsigned sz_1;				// sz - 1
	double sz2;					// sz^2
	double smallGlyphsCoverage;	// max density for symbols considered small
	double preferredMaxMcDist;	// acceptable distance between mass centers (3/8*sz)
	double complPrefMaxMcDist;	// max possible distance between mass centers (sz_1*sqrt(2)) - preferredMaxMcDist
	cv::Point2d patchCenter;	// position of the center of the patch (sz_1/2, sz_1/2)
	cv::Mat consec;				// row matrix with consecutive elements: 0..sz-1

private:
	friend class MatchEngine;
	void update(unsigned sz_, const FontEngine &fe_);
};

// MatchEngine finds best match for a patch based on current settings and symbols set.
class MatchEngine final : public IMatch {
public:
	// VSymData - vector with most information about each symbol
	typedef std::vector<SymData> VSymData;

	// Displaying the symbols requires dividing them into pages (ranges using iterators)
	typedef VSymData::const_iterator VSymDataCIt;
	typedef std::pair< VSymDataCIt, VSymDataCIt > VSymDataCItPair;

private:
	const Config &cfg;			// settings for the engine
	FontEngine &fe;				// symbols set manager
	std::string symsIdReady;	// type of symbols ready to use for transformation
	VSymData symsSet;			// set of most information on each symbol

	// matching aspects
	FgMatch fgMatch;
	BgMatch bgMatch;
	EdgeMatch edgeMatch;
	BetterContrast conMatch;
	GravitationalSmoothness grMatch;
	DirectionalSmoothness dirMatch;
	LargerSym lsMatch;

	// Returns a vector with the addresses of the matching aspects from above
	const std::vector<MatchAspect*>& getAvailAspects();

	std::vector<MatchAspect*> aspects;	// enabled aspects

	CachedData cachedData;	// data precomputed by getReady before performing the matching series

	// Determines best match of 'patch' compared to the elements from 'symsSet'
	void findBestMatch(const cv::Mat &patch, BestMatch &best);

public:
	MatchEngine(const Config &cfg_, FontEngine &fe_);

	std::string getIdForSymsToUse(); // type of the symbols determined by fe & cfg

	// Needed to display the cmap - returns a pair of symsSet iterators
	VSymDataCItPair getSymsRange(unsigned from, unsigned count) const;

	void updateSymbols();	// using different charmap - also useful for displaying these changes
	void getReady();		// called before a series of approxPatch

	// returns the approximation of 'patch_' plus other details in 'best'
	cv::Mat approxPatch(const cv::Mat &patch_, BestMatch &best);

	// scores the match between a gray patch and a symbol based on all enabled aspects
	double assessMatch(const cv::Mat &patch,
					   const SymData &symData,
					   MatchParams &mp) const override; // IMatch override
#ifdef _DEBUG
	bool usesUnicode() const; // Unicode glyphs are logged as symbols, the rest as their code
#endif // _DEBUG
};

#endif