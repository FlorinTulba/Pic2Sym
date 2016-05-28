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

#include "match.h"
#include "matchAspects.h"
#include "symData.h"
#include "cachedData.h"
#include "matchParams.h"

using namespace std;
using namespace cv;

REGISTERED_MATCH_ASPECT(FgMatch);
REGISTERED_MATCH_ASPECT(BgMatch);
REGISTERED_MATCH_ASPECT(EdgeMatch);
REGISTERED_MATCH_ASPECT(BetterContrast);
REGISTERED_MATCH_ASPECT(GravitationalSmoothness);
REGISTERED_MATCH_ASPECT(DirectionalSmoothness);
REGISTERED_MATCH_ASPECT(LargerSym);

vector<const string>& MatchAspect::registeredAspects() {
	static vector<const string> names;
	return names;
}

double FgMatch::assessMatch(const Mat &patch,
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

double BgMatch::assessMatch(const Mat &patch,
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

double EdgeMatch::assessMatch(const Mat &patch,
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

double BetterContrast::assessMatch(const Mat &patch,
								   const SymData &symData,
								   MatchParams &mp) const {
	mp.computeContrast(patch, symData);
	
	// Encourages larger contrasts:
	// 0 for no contrast; 1 for max contrast (255)
	return pow( abs(mp.contrast.value()) / 255., k);
}

double GravitationalSmoothness::assessMatch(const Mat &patch,
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

double DirectionalSmoothness::assessMatch(const Mat &patch,
										  const SymData &symData,
										  MatchParams &mp) const {
	static const double SQRT2 = sqrt(2), TWOmSQRT2 = 2. - SQRT2;
	static const Point2d ORIGIN; // (0, 0)

	mp.computeMcsOffset(patch, symData, cachedData);

	const Point2d relMcPatch = mp.mcPatch.value() - cachedData.patchCenter;
	const Point2d relMcGlyph = mp.mcPatchApprox.value() - cachedData.patchCenter;

	// best gradient orientation when angle between mc-s is 0 => cos = 1	
	double cosAngleMCs = 0.; // -1..1 range, best when 1
	if(relMcGlyph != ORIGIN && relMcPatch != ORIGIN) // avoid DivBy0
		cosAngleMCs = relMcGlyph.dot(relMcPatch) / (norm(relMcGlyph) * norm(relMcPatch));

	// Penalizes large angle between mc-s, but no so much when they are close to each other.
	// The mc-s are consider close when the distance between them is < preferredMaxMcDist
	//		(1. + cosAngleMCs) * (2-sqrt(2)) is <=1 for |angleMCs| >= 45  and  >1 otherwise
	// So, large k generally penalizes large angles and encourages small ones,
	// but fades gradually for nearer mc-s or fades completely when the mc-s overlap.
	return pow((1. + cosAngleMCs) * TWOmSQRT2,
			   k * min(mp.mcsOffset.value(), cachedData.preferredMaxMcDist) /
			   cachedData.preferredMaxMcDist);
}

double LargerSym::assessMatch(const Mat&,
							  const SymData &symData,
							  MatchParams &mp) const {
	mp.computeSymDensity(symData, cachedData);

	// Encourages approximations with symbols filling at least x% of their box.
	// The threshold x is provided by smallGlyphsCoverage.
	// Returns < 1 for glyphs under threshold;   >= 1 otherwise
	return pow(mp.symDensity.value() + 1. - cachedData.smallGlyphsCoverage, k);
}
