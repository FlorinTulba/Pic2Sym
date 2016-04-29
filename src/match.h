/****************************************************************************************
 The application Pic2Sym approximates images by a
 grid of colored symbols with colored backgrounds.

 This file was created on 2016-3-1
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
		NameRegistrator(const std::string &aspectType) {
			registeredAspects().push_back(aspectType);
		}
	};

	const CachedData &cachedData; ///< cached information from matching engine
	const double &k; ///< cached coefficient from MatchSettings, corresponding to current aspect

	/// Base class constructor
	MatchAspect(const CachedData &cachedData_, const double &k_) :
			cachedData(cachedData_), k(k_) {}

public:
	virtual ~MatchAspect() = 0 {}

	/// All aspects that are configured with coefficients > 0 are enabled; those with 0 are disabled
	bool enabled() const { return k > 0.; }

	/// Provides the list of names of all registered aspects
	static const std::vector<const std::string>& aspectNames() { return registeredAspects(); }
};

/// Place this call in a private region of an aspect class to register (HEADER file).
/// This is only the declaration of 'nameRegistrator'.
#define REGISTER_MATCH_ASPECT(AspectName) \
	static const NameRegistrator nameRegistrator; /** Instance that registers this Aspect */ \
	/** '_Ref_count_obj' helps 'make_shared' create the object to point to */ \
	friend class std::_Ref_count_obj<AspectName>

/// Place this call in a SOURCE file, to define the declared static 'nameRegistrator'.
/// This is the definition of 'nameRegistrator'.
#define REGISTERED_MATCH_ASPECT(AspectName) \
const AspectName::NameRegistrator AspectName::nameRegistrator(#AspectName)

/*
STEPS TO CREATE A NEW 'MatchAspect' (<NewAspect>):
==================================================

(1) Create a class for it using the template:

	/// Class Details
	class <NewAspect> : public MatchAspect {
		REGISTER_MATCH_ASPECT(<NewAspect>);

	public:
		/// scores the match between a gray patch and a symbol based on current aspect
		double assessMatch(const cv::Mat &patch,
						   const SymData &symData,
						   MatchParams &mp) const override; // IMatch override

	#ifndef UNIT_TESTING // UNIT_TESTING needs the constructors as public
	protected:
	#endif
		/// Constructor Details
		<NewAspect>(const CachedData &cachedData_, const MatchSettings &ms);
	};

(2) Place following call in a cpp unit:

	REGISTERED_MATCH_ASPECT(<NewAspect>)

(3) Include the file declaring <NewAspect> in 'matchAspectsFactory.cpp' and
	add the line below to 'MatchAspectsFactory::create()'.

	HANDLE_MATCH_ASPECT(<NewAspect>);

*/

#endif