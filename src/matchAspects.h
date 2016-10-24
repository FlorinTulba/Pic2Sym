/************************************************************************************************
 The application Pic2Sym approximates images by a
 grid of colored symbols with colored backgrounds.

 This file belongs to the Pic2Sym project.

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
 ***********************************************************************************************/

#ifndef H_MATCH_ASPECTS
#define H_MATCH_ASPECTS

#include "match.h"

/// Selecting a symbol with the scene underneath it as uniform as possible
class FgMatch : public MatchAspect {
public:
	/// Providing a clue about how complex is this MatchAspect compared to the others
	double relativeComplexity() const override;

protected:
	/// Defines the scoring rule, based on all required fields computed already in MatchParams mp
	double score(const MatchParams &mp, const CachedData &cachedData) const override;

	/// Prepares required fields from MatchParams mp to be able to assess the match
	void fillRequiredMatchParams(const cv::Mat &patch,
								 const SymData &symData,
								 const CachedData &cachedData,
								 MatchParams &mp) const override;

#ifdef UNIT_TESTING // UNIT_TESTING needs the constructors as public
public:
#endif

	FgMatch(const MatchSettings &cfg);

	REGISTER_MATCH_ASPECT(FgMatch);
};

/// Aspect ensuring more uniform background scene around the selected symbol
class BgMatch : public MatchAspect {
public:
	/// Providing a clue about how complex is this MatchAspect compared to the others
	double relativeComplexity() const override;

protected:
	/// Defines the scoring rule, based on all required fields computed already in MatchParams mp
	double score(const MatchParams &mp, const CachedData &cachedData) const override;

	/// Prepares required fields from MatchParams mp to be able to assess the match
	void fillRequiredMatchParams(const cv::Mat &patch,
								 const SymData &symData,
								 const CachedData &cachedData,
								 MatchParams &mp) const override;

#ifdef UNIT_TESTING // UNIT_TESTING needs the constructors as public
public:
#endif

	BgMatch(const MatchSettings &cfg);

	REGISTER_MATCH_ASPECT(BgMatch);
};

/// Aspect ensuring the edges of the selected symbol seem to appear also on the patch
class EdgeMatch : public MatchAspect {
public:
	/// Providing a clue about how complex is this MatchAspect compared to the others
	double relativeComplexity() const override;

protected:
	/// Defines the scoring rule, based on all required fields computed already in MatchParams mp
	double score(const MatchParams &mp, const CachedData &cachedData) const override;

	/// Prepares required fields from MatchParams mp to be able to assess the match
	void fillRequiredMatchParams(const cv::Mat &patch,
								 const SymData &symData,
								 const CachedData &cachedData,
								 MatchParams &mp) const override;

#ifdef UNIT_TESTING // UNIT_TESTING needs the constructors as public
public:
#endif

	EdgeMatch(const MatchSettings &cfg);

	REGISTER_MATCH_ASPECT(EdgeMatch);
};

/// Discouraging barely visible symbols
class BetterContrast : public MatchAspect {
public:
	/// Providing a clue about how complex is this MatchAspect compared to the others
	double relativeComplexity() const override;

protected:
	/// Defines the scoring rule, based on all required fields computed already in MatchParams mp
	double score(const MatchParams &mp, const CachedData &cachedData) const override;

	/// Prepares required fields from MatchParams mp to be able to assess the match
	void fillRequiredMatchParams(const cv::Mat &patch,
								 const SymData &symData,
								 const CachedData &cachedData,
								 MatchParams &mp) const override;

#ifdef UNIT_TESTING // UNIT_TESTING needs the constructors as public
public:
#endif

	BetterContrast(const MatchSettings &cfg);

	REGISTER_MATCH_ASPECT(BetterContrast);
};

/// Aspect concentrating on where's the center of gravity of the patch & its approximation
class GravitationalSmoothness : public MatchAspect {
public:
	/// Providing a clue about how complex is this MatchAspect compared to the others
	double relativeComplexity() const override;

protected:
	/// Defines the scoring rule, based on all required fields computed already in MatchParams mp
	double score(const MatchParams &mp, const CachedData &cachedData) const override;

	/// Prepares required fields from MatchParams mp to be able to assess the match
	void fillRequiredMatchParams(const cv::Mat &patch,
								 const SymData &symData,
								 const CachedData &cachedData,
								 MatchParams &mp) const override;

#ifdef UNIT_TESTING // UNIT_TESTING needs the constructors as public
public:
#endif

	GravitationalSmoothness(const MatchSettings &cfg);

	REGISTER_MATCH_ASPECT(GravitationalSmoothness);
};

/// Aspect encouraging more accuracy while approximating the direction of the patch
class DirectionalSmoothness : public MatchAspect {
public:
	/// Providing a clue about how complex is this MatchAspect compared to the others
	double relativeComplexity() const override;

protected:
	/// Defines the scoring rule, based on all required fields computed already in MatchParams mp
	double score(const MatchParams &mp, const CachedData &cachedData) const override;

	/// Prepares required fields from MatchParams mp to be able to assess the match
	void fillRequiredMatchParams(const cv::Mat &patch,
								 const SymData &symData,
								 const CachedData &cachedData,
								 MatchParams &mp) const override;

#ifdef UNIT_TESTING // UNIT_TESTING needs the constructors as public
public:
#endif

	DirectionalSmoothness(const MatchSettings &cfg);

	REGISTER_MATCH_ASPECT(DirectionalSmoothness);
};

/// Match aspect concerning user's preference for larger symbols as approximations
class LargerSym : public MatchAspect {
public:
	/// Providing a clue about how complex is this MatchAspect compared to the others
	double relativeComplexity() const override;

protected:
	/// Defines the scoring rule, based on all required fields computed already in MatchParams mp
	double score(const MatchParams &mp, const CachedData &cachedData) const override;

	/// Prepares required fields from MatchParams mp to be able to assess the match
	void fillRequiredMatchParams(const cv::Mat &patch,
								 const SymData &symData,
								 const CachedData &cachedData,
								 MatchParams &mp) const override;

#ifdef UNIT_TESTING // UNIT_TESTING needs the constructors as public
public:
#endif

	LargerSym(const MatchSettings &cfg);

	REGISTER_MATCH_ASPECT(LargerSym);
};

#endif