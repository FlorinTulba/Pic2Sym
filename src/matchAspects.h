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
 - (c) 1997-2002 OpenMP Architecture Review Board (www.openmp.org)
   (c) Microsoft Corporation (Visual C++ implementation for OpenMP C/C++ Version 2.0 March 2002)
   See: <https://msdn.microsoft.com/en-us/library/8y6825x5(v=vs.140).aspx>
 
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

/// Selecting a symbol with the scene underneath it as uniform as possible
class FgMatch : public MatchAspect {
	REGISTER_MATCH_ASPECT(FgMatch);

public:
	/// Providing a clue about how complex is this MatchAspect compared to the others
	double relativeComplexity() const override;

protected:
	/// Defines the scoring rule, based on all required fields computed already in MatchParams mp
	double score(const MatchParams &mp) const override;

	/// Prepares required fields from MatchParams mp to be able to assess the match
	void fillRequiredMatchParams(const cv::Mat &patch,
								 const SymData &symData,
								 MatchParams &mp) const override;

#ifdef UNIT_TESTING // UNIT_TESTING needs the constructors as public
public:
#endif

	FgMatch(const CachedData &cachedData_, const MatchSettings &cfg);
};

/// Aspect ensuring more uniform background scene around the selected symbol
class BgMatch : public MatchAspect {
	REGISTER_MATCH_ASPECT(BgMatch);

public:
	/// Providing a clue about how complex is this MatchAspect compared to the others
	double relativeComplexity() const override;

protected:
	/// Defines the scoring rule, based on all required fields computed already in MatchParams mp
	double score(const MatchParams &mp) const override;

	/// Prepares required fields from MatchParams mp to be able to assess the match
	void fillRequiredMatchParams(const cv::Mat &patch,
								 const SymData &symData,
								 MatchParams &mp) const override;

#ifdef UNIT_TESTING // UNIT_TESTING needs the constructors as public
public:
#endif

	BgMatch(const CachedData &cachedData_, const MatchSettings &cfg);
};

/// Aspect ensuring the edges of the selected symbol seem to appear also on the patch
class EdgeMatch : public MatchAspect {
	REGISTER_MATCH_ASPECT(EdgeMatch);

public:
	/// Providing a clue about how complex is this MatchAspect compared to the others
	double relativeComplexity() const override;

protected:
	/// Defines the scoring rule, based on all required fields computed already in MatchParams mp
	double score(const MatchParams &mp) const override;

	/// Prepares required fields from MatchParams mp to be able to assess the match
	void fillRequiredMatchParams(const cv::Mat &patch,
								 const SymData &symData,
								 MatchParams &mp) const override;

#ifdef UNIT_TESTING // UNIT_TESTING needs the constructors as public
public:
#endif

	EdgeMatch(const CachedData &cachedData_, const MatchSettings &cfg);
};

/// Discouraging barely visible symbols
class BetterContrast : public MatchAspect {
	REGISTER_MATCH_ASPECT(BetterContrast);

public:
	/// Providing a clue about how complex is this MatchAspect compared to the others
	double relativeComplexity() const override;

protected:
	/// Defines the scoring rule, based on all required fields computed already in MatchParams mp
	double score(const MatchParams &mp) const override;

	/// Prepares required fields from MatchParams mp to be able to assess the match
	void fillRequiredMatchParams(const cv::Mat &patch,
								 const SymData &symData,
								 MatchParams &mp) const override;

#ifdef UNIT_TESTING // UNIT_TESTING needs the constructors as public
public:
#endif

	BetterContrast(const CachedData &cachedData_, const MatchSettings &cfg);
};

/// Aspect concentrating on where's the center of gravity of the patch & its approximation
class GravitationalSmoothness : public MatchAspect {
	REGISTER_MATCH_ASPECT(GravitationalSmoothness);

public:
	/// Providing a clue about how complex is this MatchAspect compared to the others
	double relativeComplexity() const override;

protected:
	/// Defines the scoring rule, based on all required fields computed already in MatchParams mp
	double score(const MatchParams &mp) const override;

	/// Prepares required fields from MatchParams mp to be able to assess the match
	void fillRequiredMatchParams(const cv::Mat &patch,
								 const SymData &symData,
								 MatchParams &mp) const override;

#ifdef UNIT_TESTING // UNIT_TESTING needs the constructors as public
public:
#endif

	GravitationalSmoothness(const CachedData &cachedData_, const MatchSettings &cfg);
};

/// Aspect encouraging more accuracy while approximating the direction of the patch
class DirectionalSmoothness : public MatchAspect {
	REGISTER_MATCH_ASPECT(DirectionalSmoothness);

public:
	/// Providing a clue about how complex is this MatchAspect compared to the others
	double relativeComplexity() const override;

protected:
	/// Defines the scoring rule, based on all required fields computed already in MatchParams mp
	double score(const MatchParams &mp) const override;

	/// Prepares required fields from MatchParams mp to be able to assess the match
	void fillRequiredMatchParams(const cv::Mat &patch,
								 const SymData &symData,
								 MatchParams &mp) const override;

#ifdef UNIT_TESTING // UNIT_TESTING needs the constructors as public
public:
#endif

	DirectionalSmoothness(const CachedData &cachedData_, const MatchSettings &cfg);
};

/// Match aspect concerning user's preference for larger symbols as approximations
class LargerSym : public MatchAspect {
	REGISTER_MATCH_ASPECT(LargerSym);

public:
	/// Providing a clue about how complex is this MatchAspect compared to the others
	double relativeComplexity() const override;

protected:
	/// Defines the scoring rule, based on all required fields computed already in MatchParams mp
	double score(const MatchParams &mp) const override;

	/// Prepares required fields from MatchParams mp to be able to assess the match
	void fillRequiredMatchParams(const cv::Mat &patch,
								 const SymData &symData,
								 MatchParams &mp) const override;

#ifdef UNIT_TESTING // UNIT_TESTING needs the constructors as public
public:
#endif

	LargerSym(const CachedData &cachedData_, const MatchSettings &cfg);
};

#endif