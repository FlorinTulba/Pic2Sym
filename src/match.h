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
 
 (c) 2016, 2017 Florin Tulba <florintulba@yahoo.com>

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

#ifndef H_MATCH
#define H_MATCH

#include "matchSettings.h"
#include "symData.h"
#include "cachedData.h"

#pragma warning ( push, 0 )

#include <vector>
#include <string>

#pragma warning ( pop )

struct MatchParams; // forward declaration

/// Interface providing assessMatch method for MatchAspect classes and also for MatchEngine
struct IMatch /*abstract*/ {
	/// scores the match between a gray patch and a symbol
	virtual double assessMatch(const cv::Mat &patch,
							   const SymData &symData,
							   const CachedData &cachedData,
							   MatchParams &mp) const = 0;
	virtual ~IMatch() = 0 {}
};

/**
Base class for all considered aspects of matching.

Derived classes should have protected constructors and
objects should be created only by the MatchAspectsFactory class.

UNIT_TESTING should still have the constructors of the derived classes as public.
*/
class MatchAspect /*abstract*/ : public IMatch {
protected:
	/// Provides a list of names of the already registered aspects
	static std::vector<const std::string>& registeredAspects();

	/// Helper class to populate registeredAspects.
	/// Define a static private field of this type in each subclass using
	/// REGISTER_MATCH_ASPECT and REGISTERED_MATCH_ASPECT defined below
	struct NameRegistrator {
		/// adds a new aspect name to registeredAspects
		NameRegistrator(const std::string &aspectType);
	};

	const double &k; ///< cached coefficient from MatchSettings, corresponding to current aspect

	/// Defines the scoring rule, based on all required fields computed already in MatchParams mp
	virtual double score(const MatchParams &mp, const CachedData &cachedData) const = 0;

	/// Prepares required fields from MatchParams mp to be able to assess the match
	virtual void fillRequiredMatchParams(const cv::Mat &patch,
										 const SymData &symData,
										 const CachedData &cachedData,
										 MatchParams &mp) const = 0;

	/// Base class constructor
	MatchAspect(const double &k_);
	void operator=(const MatchAspect&) = delete;

public:
	virtual ~MatchAspect() = 0 {}

	/// Scores the match between a gray patch and a symbol based on current aspect (IMatch override)
	double assessMatch(const cv::Mat &patch,
					   const SymData &symData,
					   const CachedData &cachedData,
					   MatchParams &mp) const override final; // Template method (reason to set it final)

	/// Computing max score of a this MatchAspect
	double maxScore(const CachedData &cachedData) const;

	/// Providing a clue about how complex is this MatchAspect compared to the others
	virtual double relativeComplexity() const = 0;

	virtual const std::string& name() const = 0; ///< provides aspect's name

	/// All aspects that are configured with coefficients > 0 are enabled; those with 0 are disabled
	bool enabled() const;

	/// Provides the list of names of all registered aspects
	static const std::vector<const std::string>& aspectNames();
};

/// Place this call at the end of an aspect class to register (HEADER file).
/// Definitions of 'NAME', 'nameRegistrator' and name() are provided by REGISTERED_MATCH_ASPECT.
#define REGISTER_MATCH_ASPECT(AspectName) \
	public: \
		const std::string& name() const override;		/** provides aspect's name */ \
	\
	protected: \
		static const std::string NAME;					/** aspect's name */ \
		static const NameRegistrator nameRegistrator;	/** Instance that registers this Aspect */ \
		/** '_Ref_count_obj' helps 'make_shared' create the object to point to */ \
		friend class std::_Ref_count_obj<AspectName>

/// Place this call in a SOURCE file, to define the entities declared by REGISTER_MATCH_ASPECT
/// This is the definition of 'nameRegistrator'.
#define REGISTERED_MATCH_ASPECT(AspectName) \
	const std::string					AspectName::NAME(#AspectName); \
	const AspectName::NameRegistrator	AspectName::nameRegistrator(#AspectName); \
	\
	const std::string& AspectName::name() const { return NAME; }

/*
STEPS TO CREATE A NEW 'MatchAspect' (<NewAspect>):
==================================================

(1) Create a class for it using the template:

	/// Class Details
	class <NewAspect> : public MatchAspect {
	public:
		/// scores the match between a gray patch and a symbol based on current aspect
		double assessMatch(const cv::Mat &patch,
						   const SymData &symData,
						   MatchParams &mp,
						   const CachedData &cachedData) const override; // IMatch override

	#ifndef UNIT_TESTING // UNIT_TESTING needs the constructors as public
	protected:
	#endif // UNIT_TESTING defined
		/// Constructor Details
		<NewAspect>(const MatchSettings &ms);

		REGISTER_MATCH_ASPECT(<NewAspect>);
	};

(2) Place following call in a cpp unit:

	REGISTERED_MATCH_ASPECT(<NewAspect>)

(3) Include the file declaring <NewAspect> in 'matchAspectsFactory.cpp' and
	add the line below to 'MatchAspectsFactory::create()'.

	HANDLE_MATCH_ASPECT(<NewAspect>);

*/

#endif // H_MATCH