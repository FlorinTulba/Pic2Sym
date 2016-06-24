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

#ifndef H_SYM_FILTER
#define H_SYM_FILTER

#include "symFilterBase.h"

#include <map>
#include <memory>
#include <string>

/// Base class of the template class TSymFilter from below, to keep the template as thin as possible
struct SymFilter /*abstract*/ : ISymFilter {
	/// Returns the name of the filter identified by filterId_
	static const std::string& filterName(unsigned filterId_);

private: // SymFilter cannot be directly derived, except by the friend TSymFilter declared below
	template<class T> friend struct TSymFilter;

	/// filterId - filterName associations
	static std::map<unsigned, const std::string> filterTypes;

	std::unique_ptr<ISymFilter> nextFilter;	///< null or a derivate from SymFilter
	const unsigned filterId;				///< id of the filter (must be &gt; 0)

	/**
	Constructs a new SymFilter with the provided id and name, which both must be unique.

	@param filterId_ id of the filter
	@param filterName name of the filter
	@param nextFilter_ optional successor filter

	@throw invalid_argument for a 0 / non-unique filterId_ or for a non-unique filterName
	*/
	SymFilter(unsigned filterId_, const std::string &filterName,
			  std::unique_ptr<ISymFilter> nextFilter_);
	virtual ~SymFilter() = 0 {}
};

/**
Base class for any filters which can be applied on symbols.
TSymFilter is a template layer over SymFilter to use CRTP.
Normal polymorphism isn't necessary, as the used filters don't have state,
so static methods would be enough - that is static polymorphism does the job.

As a consequence, derived classes from TSymFilter must have a public method with following signature:
	static bool isDisposable(const PixMapSym &pms, const SymFilterCache &sfc)
*/
template<class DerivedFromTSymFilter>
struct TSymFilter : SymFilter {
	/**
	Constructs a new TSymFilter with the provided id and name, which both must be unique.

	@param filterId_ id of the filter
	@param filterName name of the filter
	@param nextFilter_ optional successor filter

	@throw invalid_argument for a 0 / non-unique filterId_ or for a non-unique filterName
	*/
	TSymFilter(unsigned filterId_, const std::string &filterName,
			   std::unique_ptr<ISymFilter> nextFilter_) :
		SymFilter(filterId_, filterName, std::move(nextFilter_)) {}

	/**
	Returns the id of the filter which detected that the symbol exhibits some undesired features.
	0 means no filters considered the glyph as disposable.

	Derived classes from TSymFilter must have a public method with following signature:
	static bool isDisposable(const PixMapSym &pms, const SymFilterCache &sfc)
	*/
	unsigned matchingFilterId(const PixMapSym &pms, const SymFilterCache &sfc) const override {
		if(DerivedFromTSymFilter::isDisposable(pms, sfc)) // using static polymorphism
			return filterId;

		return nextFilter->matchingFilterId(pms, sfc);
	}
};

#endif