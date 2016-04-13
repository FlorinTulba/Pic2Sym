/****************************************************************************************
 The application Pic2Sym approximates images by a
 grid of colored symbols with colored backgrounds.

 This file was created on 2016-4-13
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

#ifndef H_MATCH_ASPECTS
#define H_MATCH_ASPECTS

#include "match.h"

/**
Selecting a symbol with best structural similarity.

See https://ece.uwaterloo.ca/~z70wang/research/ssim for details.
*/
class StructuralSimilarity : public MatchAspect {
	REGISTER_MATCH_ASPECT(StructuralSimilarity);

public:
	static const cv::Size WIN_SIZE;	///< recommended window
	static const double SIGMA;		///< recommended standard deviation
	static const double C1;			///< the 1st used stabilizer coefficient
	static const double C2;			///< the 2nd used stabilizer coefficient

	/// scores the match between a gray patch and a symbol based on current aspect
	double assessMatch(const cv::Mat &patch,
					   const SymData &symData,
					   MatchParams &mp) const override; // IMatch override

#ifndef UNIT_TESTING // UNIT_TESTING needs the constructors as public
protected:
#endif

	StructuralSimilarity(const CachedData &cachedData_, const MatchSettings &cfg) :
		MatchAspect(cachedData_, cfg.get_kSsim()) {}
};

/// Selecting a symbol with the scene underneath it as uniform as possible
class FgMatch : public MatchAspect {
	REGISTER_MATCH_ASPECT(FgMatch);

public:
	/// scores the match between a gray patch and a symbol based on current aspect
	double assessMatch(const cv::Mat &patch,
					   const SymData &symData,
					   MatchParams &mp) const override; // IMatch override

#ifndef UNIT_TESTING // UNIT_TESTING needs the constructors as public
protected:
#endif

	FgMatch(const CachedData &cachedData_, const MatchSettings &cfg) :
		MatchAspect(cachedData_, cfg.get_kSdevFg()) {}
};

/// Aspect ensuring more uniform background scene around the selected symbol
class BgMatch : public MatchAspect {
	REGISTER_MATCH_ASPECT(BgMatch);

public:
	/// scores the match between a gray patch and a symbol based on current aspect
	double assessMatch(const cv::Mat &patch,
					   const SymData &symData,
					   MatchParams &mp) const override; // IMatch override

#ifndef UNIT_TESTING // UNIT_TESTING needs the constructors as public
protected:
#endif

	BgMatch(const CachedData &cachedData_, const MatchSettings &cfg) :
		MatchAspect(cachedData_, cfg.get_kSdevBg()) {}
};

/// Aspect ensuring the edges of the selected symbol seem to appear also on the patch
class EdgeMatch : public MatchAspect {
	REGISTER_MATCH_ASPECT(EdgeMatch);

public:
	/// scores the match between a gray patch and a symbol based on current aspect
	double assessMatch(const cv::Mat &patch,
					   const SymData &symData,
					   MatchParams &mp) const override; // IMatch override

#ifndef UNIT_TESTING // UNIT_TESTING needs the constructors as public
protected:
#endif

	EdgeMatch(const CachedData &cachedData_, const MatchSettings &cfg) :
		MatchAspect(cachedData_, cfg.get_kSdevEdge()) {}
};

/// Discouraging barely visible symbols
class BetterContrast : public MatchAspect {
	REGISTER_MATCH_ASPECT(BetterContrast);

public:
	/// scores the match between a gray patch and a symbol based on current aspect
	double assessMatch(const cv::Mat &patch,
					   const SymData &symData,
					   MatchParams &mp) const override; // IMatch override

#ifndef UNIT_TESTING // UNIT_TESTING needs the constructors as public
protected:
#endif

	BetterContrast(const CachedData &cachedData_, const MatchSettings &cfg) :
		MatchAspect(cachedData_, cfg.get_kContrast()) {}
};

/// Aspect concentrating on where's the center of gravity of the patch & its approximation
class GravitationalSmoothness : public MatchAspect {
	REGISTER_MATCH_ASPECT(GravitationalSmoothness);

public:
	/// scores the match between a gray patch and a symbol based on current aspect
	double assessMatch(const cv::Mat &patch,
					   const SymData &symData,
					   MatchParams &mp) const override; // IMatch override

#ifndef UNIT_TESTING // UNIT_TESTING needs the constructors as public
protected:
#endif

	GravitationalSmoothness(const CachedData &cachedData_, const MatchSettings &cfg) :
		MatchAspect(cachedData_, cfg.get_kMCsOffset()) {}
};

/// Aspect encouraging more accuracy while approximating the direction of the patch
class DirectionalSmoothness : public MatchAspect {
	REGISTER_MATCH_ASPECT(DirectionalSmoothness);

public:
	/// scores the match between a gray patch and a symbol based on current aspect
	double assessMatch(const cv::Mat &patch,
					   const SymData &symData,
					   MatchParams &mp) const override; // IMatch override

#ifndef UNIT_TESTING // UNIT_TESTING needs the constructors as public
protected:
#endif

	DirectionalSmoothness(const CachedData &cachedData_, const MatchSettings &cfg) :
		MatchAspect(cachedData_, cfg.get_kCosAngleMCs()) {}
};

/// Match aspect concerning user's preference for larger symbols as approximations
class LargerSym : public MatchAspect {
	REGISTER_MATCH_ASPECT(LargerSym);

public:
	/// scores the match between a gray patch and a symbol based on current aspect
	double assessMatch(const cv::Mat&,
					   const SymData &symData,
					   MatchParams &mp) const override; // IMatch override

#ifndef UNIT_TESTING // UNIT_TESTING needs the constructors as public
protected:
#endif

	LargerSym(const CachedData &cachedData_, const MatchSettings &cfg) :
		MatchAspect(cachedData_, cfg.get_kSymDensity()) {}
};

#endif