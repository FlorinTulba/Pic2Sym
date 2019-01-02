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

#ifndef H_SYM_FILTER
#define H_SYM_FILTER

#include "symFilterBase.h"

#pragma warning ( push, 0 )

#include "std_string.h"
#include "std_memory.h"
#include <unordered_map>

#pragma warning ( pop )

/// Base class of the template class TSymFilter from below, to keep the template as thin as possible
struct SymFilter /*abstract*/ : ISymFilter {
	/// Returns the name of the filter identified by filterId_
	static const std::stringType& filterName(unsigned filterId_);

private: // SymFilter cannot be directly derived, except by the friend TSymFilter declared below
	template<class T> friend class TSymFilter;

	/// filterId - filterName associations
	static std::unordered_map<unsigned, const std::stringType> filterTypes;

	const std::uniquePtr<ISymFilter> nextFilter;	///< DefSymFilter or a derivate from SymFilter
	const unsigned filterId;						///< id of the filter

	/**
	Constructs a new SymFilter with the provided id and name, which both must be unique.

	@param filterId_ id of the filter
	@param filterName name of the filter
	@param nextFilter_ optional successor filter

	@throw invalid_argument for a non-unique filterId_ or for a non-unique filterName
	*/
	SymFilter(unsigned filterId_, const std::stringType &filterName,
			  std::uniquePtr<ISymFilter> nextFilter_);
	SymFilter(const SymFilter&) = delete;
	void operator=(const SymFilter&) = delete;

	~SymFilter() {}
};

/**
Base class for any filters which can be applied on symbols.
TSymFilter is a template layer over SymFilter to use CRTP.
Normal polymorphism isn't necessary, as the used filters don't have state,
so static methods would be enough - that is static polymorphism does the job.

As a consequence, derived classes from TSymFilter must have 2 public methods with following signature:
	static bool isEnabled()
	static bool isDisposable(const IPixMapSym &pms, const SymFilterCache &sfc)
*/
template<class DerivedFromTSymFilter>
class TSymFilter /*abstract*/ : public SymFilter {
protected:
	/**
	Constructs a new TSymFilter with the provided id and name, which both must be unique.

	@param filterId_ id of the filter
	@param filterName name of the filter
	@param nextFilter_ optional successor filter

	@throw invalid_argument for a non-unique filterId_ or for a non-unique filterName
	*/
	TSymFilter(unsigned filterId_, const std::stringType &filterName,
			   std::uniquePtr<ISymFilter> nextFilter_) :
		SymFilter(filterId_, filterName, std::move(nextFilter_)) {}

	TSymFilter(const TSymFilter&) = delete;
	void operator=(const TSymFilter&) = delete;

public:
	/**
	Returns the id of the filter which detected that the symbol exhibits some undesired features.

	Derived classes from TSymFilter must have 2 public methods with following signature:
		static bool isEnabled()
		static bool isDisposable(const IPixMapSym &pms, const SymFilterCache &sfc)
	*/
	boost::optional<unsigned> matchingFilterId(const IPixMapSym &pms, const SymFilterCache &sfc) const override {
		// Using static polymorphism
		if(DerivedFromTSymFilter::isEnabled() && DerivedFromTSymFilter::isDisposable(pms, sfc))
			return filterId;

		return nextFilter->matchingFilterId(pms, sfc);
	}
};

/// Macro for defining DerivedFromTSymFilter::isEnabled() method within each DerivedFromTSymFilter class
#define CHECK_ENABLED_SYM_FILTER(DerivedFromTSymFilter) \
	static bool isEnabled() { \
		extern const bool DerivedFromTSymFilter##Enabled; \
		return DerivedFromTSymFilter##Enabled; \
	}			

#endif // H_SYM_FILTER
