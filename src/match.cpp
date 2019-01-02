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
 
 (c) 2016-2019 Florin Tulba <florintulba@yahoo.com>

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

#include "match.h"
#include "matchAspects.h"
#include "symDataBase.h"
#include "cachedData.h"
#include "matchParams.h"
#include "warnings.h"

using namespace std;
using namespace cv;

namespace {
	const Point2d ORIGIN; // (0, 0)
	const double TWOmSQRT2 = 2. - sqrt(2);

	// Early computation and avoids the Clique MatchParams - MatchAspect
	const IMatchParams &idealMatch = MatchParams::perfectMatch();
} // anonymous namespace

MatchAspect::MatchAspect(const double &k_) : k(k_) {}

double MatchAspect::assessMatch(const Mat &patch,
								const ISymData &symData,
								const CachedData &cachedData,
								IMatchParamsRW &mp) const {
	fillRequiredMatchParams(patch, symData, cachedData, mp);
	return score(mp, cachedData);
}

double MatchAspect::maxScore(const CachedData &cachedData) const {
	return score(idealMatch, cachedData);
}

bool MatchAspect::enabled() const {
	return k > 0.;
}

const vector<const stringType>& MatchAspect::aspectNames() {
	return registeredAspects();
}

vector<const stringType>& MatchAspect::registeredAspects() {
#pragma warning ( disable : WARN_THREAD_UNSAFE )
	static vector<const stringType> names;
#pragma warning ( default : WARN_THREAD_UNSAFE )

	return names;
}

MatchAspect::NameRegistrator::NameRegistrator(const stringType &aspectType) {
	registeredAspects().push_back(aspectType);
}

REGISTERED_MATCH_ASPECT(FgMatch);
REGISTERED_MATCH_ASPECT(BgMatch);
REGISTERED_MATCH_ASPECT(EdgeMatch);
REGISTERED_MATCH_ASPECT(BetterContrast);
REGISTERED_MATCH_ASPECT(GravitationalSmoothness);
REGISTERED_MATCH_ASPECT(DirectionalSmoothness);
REGISTERED_MATCH_ASPECT(LargerSym);

FgMatch::FgMatch(const IMatchSettings &ms) : MatchAspect(ms.get_kSdevFg()) {}

/**
Returned value discourages large std. devs.
For sdev =     0 (min) => returns 1 no matter k
For sdev = 127.5 (max) => returns 0 no matter k (For k=0, this matching aspect is disabled)
For other sdev-s       =>
	returns closer to 1 for k in (0..1)
	returns sdev for k=1
	returns closer to 0 for k>1 (Large k => higher penalty for large sdev-s)
*/
double FgMatch::score(const IMatchParams &mp, const CachedData&) const {
	return pow(1. - mp.getSdevFg().value() / CachedData::MaxSdev::forFgOrBg(), k);
}

void FgMatch::fillRequiredMatchParams(const Mat &patch,
									  const ISymData &symData,
									  const CachedData&,
									  IMatchParamsRW &mp) const {
	mp.computeSdevFg(patch, symData);
}

double FgMatch::relativeComplexity() const {
	// Simpler than BgMatch if considering that fg masks are typically smaller than bg ones,
	// so there are less values to consider
	return 3.1;
}

BgMatch::BgMatch(const IMatchSettings &ms) : MatchAspect(ms.get_kSdevBg()) {}

/**
Returned value discourages large std. devs.
For sdev =     0 (min) => returns 1 no matter k
For sdev = 127.5 (max) => returns 0 no matter k (For k=0, this matching aspect is disabled)
For other sdev-s       =>
	returns closer to 1 for k in (0..1)
	returns sdev for k=1
	returns closer to 0 for k>1 (Large k => higher penalty for large sdev-s)
*/
double BgMatch::score(const IMatchParams &mp, const CachedData&) const {
	return pow(1. - mp.getSdevBg().value() / CachedData::MaxSdev::forFgOrBg(), k);
}

void BgMatch::fillRequiredMatchParams(const Mat &patch,
									  const ISymData &symData,
									  const CachedData&,
									  IMatchParamsRW &mp) const {
	mp.computeSdevBg(patch, symData);
}

double BgMatch::relativeComplexity() const {
	// More complex than FgMatch if considering that fg masks are typically smaller than bg ones,
	// so bg has more values to consider
	return 3.2;
}

EdgeMatch::EdgeMatch(const IMatchSettings &ms) : MatchAspect(ms.get_kSdevEdge()) {}

/**
Returned value discourages large std. devs.
For sdev =     0 (min) => returns 1 no matter k
For sdev =	255 (max) => returns 0 no matter k (For k=0, this matching aspect is disabled)
For other sdev-s       =>
	returns closer to 1 for k in (0..1)
	returns sdev for k=1
	returns closer to 0 for k>1 (Large k => higher penalty for large sdev-s)
*/
double EdgeMatch::score(const IMatchParams &mp, const CachedData&) const {
	return pow(1. - mp.getSdevEdge().value() / CachedData::MaxSdev::forEdges(), k);
}

void EdgeMatch::fillRequiredMatchParams(const Mat &patch,
										const ISymData &symData,
										const CachedData&,
										IMatchParamsRW &mp) const {
	mp.computeSdevEdge(patch, symData);
}

double EdgeMatch::relativeComplexity() const {
	// Computes contrast, performs a norm and others => longer than FgMatch/BgMatch
	return 4.;
}

BetterContrast::BetterContrast(const IMatchSettings &ms) : MatchAspect(ms.get_kContrast()) {}

/**
Encourages larger contrasts:
0 for no contrast; 1 for max contrast (255)
*/
double BetterContrast::score(const IMatchParams &mp, const CachedData&) const {
	return pow(abs(mp.getContrast().value()) / 255., k);
}

void BetterContrast::fillRequiredMatchParams(const Mat &patch,
											 const ISymData &symData,
											 const CachedData&,
											 IMatchParamsRW &mp) const {
	mp.computeContrast(patch, symData);
}

double BetterContrast::relativeComplexity() const {
	// Simpler than FgMatch/BgMatch, since they compute not only the mean, but also the standard deviation
	// Normally, computing the 2 means from BetterContrast would be quicker than a mean plus a standard deviation
	return 2.;
}

GravitationalSmoothness::GravitationalSmoothness(const IMatchSettings &ms) :
	MatchAspect(ms.get_kMCsOffset()) {}

/**
Discourages mcsOffset larger than preferredMaxMcDist:
		returns 1 for mcsOffset == preferredMaxMcDist, no matter k
		returns 0 for mcsOffset == 1.42*(fontSz-1), no matter k (k > 0)
		returns in (0..1) for mcsOffset in (preferredMaxMcDist .. 1.42*(fontSz-1) )
		returns > 1 for mcsOffset < preferredMaxMcDist
Larger k induces larger penalty for large mcsOffset and
also larger reward for small mcsOffset
*/
double GravitationalSmoothness::score(const IMatchParams &mp, const CachedData&) const {
	return pow(1. + (CachedData::MassCenters::preferredMaxMcDist() - mp.getMcsOffset().value()) *
			   CachedData::MassCenters::invComplPrefMaxMcDist(), k);
}

void GravitationalSmoothness::fillRequiredMatchParams(const Mat &patch,
													  const ISymData &symData,
													  const CachedData &cachedData,
													  IMatchParamsRW &mp) const {
	mp.computeMcsOffset(patch, symData, cachedData);
}

double GravitationalSmoothness::relativeComplexity() const {
	// Computes contrast, mass centers, symbol density
	return 15.;
}

DirectionalSmoothness::DirectionalSmoothness(const IMatchSettings &ms) :
	MatchAspect(ms.get_kCosAngleMCs()) {}

/**
Penalizes large angle between mc-s, but no so much when they are close to each other.
The mc-s are consider close when the distance between them is < PreferredMaxMcDist
		(1. + cosAngleMCs) * (2-sqrt(2)) is <=1 for |angleMCs| >= 45  and  >1 otherwise
		mcsOffsetFactor  is:
		- <1 for mcsOffset > PreferredMaxMcDist
		- 1 for mcsOffset = PreferredMaxMcDist
		- >1 for mcsOffset < PreferredMaxMcDist
So, large k penalizes large (angles & mc-s offsets) and encourages small ones from both.
*/
double DirectionalSmoothness::score(const IMatchParams &mp, const CachedData&) const {
	const Point2d relMcPatch = mp.getMcPatch().value() - CachedData::MassCenters::unitSquareCenter();
	const Point2d relMcGlyph =
		mp.getMcPatchApprox().value() - CachedData::MassCenters::unitSquareCenter();

	// best gradient orientation when angle between mc-s is 0 => cos = 1	
	double cosAngleMCs = 0.; // -1..1 range, best when 1
	if(relMcGlyph != ORIGIN && relMcPatch != ORIGIN) // avoid DivBy0
		cosAngleMCs = relMcGlyph.dot(relMcPatch) / (norm(relMcGlyph) * norm(relMcPatch));

	/*
	angleFactor encourages angles between mc under 45 degrees:
	- <1 for |angleMCs| > 45
	- 1 for |angleMCs| = 45
	- >1 for |angleMCs| < 45
	*/
	const double angleFactor = (1. + cosAngleMCs) * TWOmSQRT2;

	/*
	mcsOffsetFactor encourages smaller offsets between mc-s
	- <1 for mcsOffset > PreferredMaxMcDist
	- 1 for mcsOffset = PreferredMaxMcDist
	- >1 for mcsOffset < PreferredMaxMcDist
	*/
	const double mcsOffsetFactor = 
		CachedData::MassCenters::a_mcsOffsetFactor() * mp.getMcsOffset().value() +
		CachedData::MassCenters::b_mcsOffsetFactor();

	return pow(angleFactor * mcsOffsetFactor, k);
}

void DirectionalSmoothness::fillRequiredMatchParams(const Mat &patch,
													const ISymData &symData,
													const CachedData &cachedData,
													IMatchParamsRW &mp) const {
	mp.computeMcsOffset(patch, symData, cachedData);
}

double DirectionalSmoothness::relativeComplexity() const {
	// Although both GravitationalSmoothness and DirectionalSmoothness call only
	// computeMcsOffset, DirectionalSmoothness has a more complex score method
	return 15.1;
}

LargerSym::LargerSym(const IMatchSettings &ms) : MatchAspect(ms.get_kSymDensity()) {}

/**
Encourages approximations with symbols filling at least x% of their box.
The threshold x is provided by smallGlyphsCoverage.
Returns < 1 for glyphs under threshold;   >= 1 otherwise
*/
double LargerSym::score(const IMatchParams &mp, const CachedData &cachedData) const {
	return pow(mp.getSymDensity().value() + 1. - cachedData.getSmallGlyphsCoverage(), k);
}

void LargerSym::fillRequiredMatchParams(const Mat&,
										const ISymData &symData,
										const CachedData&,
										IMatchParamsRW &mp) const {
	mp.computeSymDensity(symData);
}

double LargerSym::relativeComplexity() const {	
	return 0.001; // Performs only a value copy
}
