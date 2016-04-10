/****************************************************************************************
 The application Pic2Sym approximates images by a
 grid of colored symbols with colored backgrounds.

 This file was created on 2016-2-1
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

#ifndef H_MATCH
#define H_MATCH

#include "matchSettings.h"
#include "symData.h"
#include "cachedData.h"

#include <vector>
#include <string>
#include <iostream>

struct MatchParams; // forward declaration

/// Interface providing assessMatch method for MatchAspect classes and also for MatchEngine
struct IMatch /*abstract*/ {
	/// scores the match between a gray patch and a symbol
	virtual double assessMatch(const cv::Mat &patch,
							   const SymData &symData,
							   MatchParams &mp) const = 0;
	virtual ~IMatch() = 0 {}
};

/// Base class for all considered aspects of matching.
class MatchAspect /*abstract*/ : public IMatch {
protected:
	static std::vector<MatchAspect*> availAspects; ///< every created aspect gets registered in here

	const CachedData &cachedData; ///< cached information from matching engine
	const double &k; ///< cached coefficient from MatchSettings, corresponding to current aspect

	/// Base class constructor registers each created aspect in availAspects
	MatchAspect(const CachedData &cachedData_, const double &k_) :
			cachedData(cachedData_), k(k_) {
		availAspects.push_back(this);
	}

public:
	/// All aspects that are configured with coefficients > 0 are enabled; those with 0 are disabled
	bool enabled() const { return k > 0.; }

	/// Provides the list of created aspects
	static const std::vector<MatchAspect*>& getAvailAspects() { return availAspects; }

#ifdef UNIT_TESTING
	/// There is a global fixture that sets up a test context free of previously created aspects
	static void clearAvailAspects() { availAspects.clear(); }
#endif
};

/**
Selecting a symbol with best structural similarity.

See https://ece.uwaterloo.ca/~z70wang/research/ssim for details.
*/
class StructuralSimilarity : public MatchAspect {
public:
	static const cv::Size WIN_SIZE;	///< recommended window
	static const double SIGMA;		///< recommended standard deviation
	static const double C1;			///< the 1st used stabilizer coefficient
	static const double C2;			///< the 2nd used stabilizer coefficient

	StructuralSimilarity(const CachedData &cachedData_, const MatchSettings &cfg) :
		MatchAspect(cachedData_, cfg.get_kSsim()) {}

	/// scores the match between a gray patch and a symbol based on current aspect
	double assessMatch(const cv::Mat &patch,
					   const SymData &symData,
					   MatchParams &mp) const override; // IMatch override
};

/// Selecting a symbol with the scene underneath it as uniform as possible
class FgMatch : public MatchAspect {
public:
	FgMatch(const CachedData &cachedData_, const MatchSettings &cfg) :
		MatchAspect(cachedData_, cfg.get_kSdevFg()) {}

	/// scores the match between a gray patch and a symbol based on current aspect
	double assessMatch(const cv::Mat &patch,
					   const SymData &symData,
					   MatchParams &mp) const override; // IMatch override
};

/// Aspect ensuring more uniform background scene around the selected symbol
class BgMatch : public MatchAspect {
public:
	BgMatch(const CachedData &cachedData_, const MatchSettings &cfg) :
		MatchAspect(cachedData_, cfg.get_kSdevBg()) {}

	/// scores the match between a gray patch and a symbol based on current aspect
	double assessMatch(const cv::Mat &patch,
					   const SymData &symData,
					   MatchParams &mp) const override; // IMatch override
};

/// Aspect ensuring the edges of the selected symbol seem to appear also on the patch
class EdgeMatch : public MatchAspect {
public:
	EdgeMatch(const CachedData &cachedData_, const MatchSettings &cfg) :
		MatchAspect(cachedData_, cfg.get_kSdevEdge()) {}

	/// scores the match between a gray patch and a symbol based on current aspect
	double assessMatch(const cv::Mat &patch,
					   const SymData &symData,
					   MatchParams &mp) const override; // IMatch override
};

/// Discouraging barely visible symbols
class BetterContrast : public MatchAspect {
public:
	BetterContrast(const CachedData &cachedData_, const MatchSettings &cfg) :
		MatchAspect(cachedData_, cfg.get_kContrast()) {}

	/// scores the match between a gray patch and a symbol based on current aspect
	double assessMatch(const cv::Mat &patch,
					   const SymData &symData,
					   MatchParams &mp) const override; // IMatch override
};

/// Aspect concentrating on where's the center of gravity of the patch & its approximation
class GravitationalSmoothness : public MatchAspect {
public:
	GravitationalSmoothness(const CachedData &cachedData_, const MatchSettings &cfg) :
		MatchAspect(cachedData_, cfg.get_kMCsOffset()) {}

	/// scores the match between a gray patch and a symbol based on current aspect
	double assessMatch(const cv::Mat &patch,
					   const SymData &symData,
					   MatchParams &mp) const override; // IMatch override
};

/// Aspect encouraging more accuracy while approximating the direction of the patch
class DirectionalSmoothness : public MatchAspect {
public:
	DirectionalSmoothness(const CachedData &cachedData_, const MatchSettings &cfg) :
		MatchAspect(cachedData_, cfg.get_kCosAngleMCs()) {}

	/// scores the match between a gray patch and a symbol based on current aspect
	double assessMatch(const cv::Mat &patch,
					   const SymData &symData,
					   MatchParams &mp) const override; // IMatch override
};

/// Match aspect concerning user's preference for larger symbols as approximations
class LargerSym : public MatchAspect {
public:
	LargerSym(const CachedData &cachedData_, const MatchSettings &cfg) :
		MatchAspect(cachedData_, cfg.get_kSymDensity()) {}

	/// scores the match between a gray patch and a symbol based on current aspect
	double assessMatch(const cv::Mat&,
					   const SymData &symData,
					   MatchParams &mp) const override; // IMatch override
};

#endif