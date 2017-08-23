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

#include "symFilter.h"
#include "symFilterCache.h"
#include "pixMapSymBase.h"
#include "misc.h"

#pragma warning ( push, 0 )

#include <set>

#pragma warning ( pop )

using namespace std;

void SymFilterCache::setFontSz(unsigned sz) {
	szD = szU = sz;
	areaD = areaU = szU * szU;
}

void SymFilterCache::setBoundingBox(unsigned height, unsigned width) {
	bbAreaD = bbAreaU = height * width;
}

map<unsigned, const stringType> SymFilter::filterTypes;

SymFilter::SymFilter(unsigned filterId_, const stringType &filterName,
					 uniquePtr<ISymFilter> nextFilter_) : ISymFilter(),
				nextFilter(nextFilter_ ? std::move(nextFilter_) : makeUnique<DefSymFilter>()),
				filterId(filterId_) {
#pragma warning ( disable : WARN_THREAD_UNSAFE )
	static set<const stringType> filterNames;
#pragma warning ( default : WARN_THREAD_UNSAFE )

#ifndef UNIT_TESTING
	if(filterTypes.find(filterId_) != filterTypes.end())
		THROW_WITH_VAR_MSG(__FUNCTION__ " called with non-unique filterId_: " + to_string(filterId_), invalid_argument);

	if(filterNames.find(filterName) != filterNames.end())
		THROW_WITH_VAR_MSG(__FUNCTION__ " called with non-unique filterName: " + filterName, invalid_argument);
#endif // UNIT_TESTING not defined

	filterTypes.emplace(filterId_, filterName);
	filterNames.insert(filterName);
}

const stringType& SymFilter::filterName(unsigned filterId_) {
	auto it = filterTypes.find(filterId_), itEnd = filterTypes.end();
	if(it != itEnd)
		return it->second;

	THROW_WITH_VAR_MSG(__FUNCTION__ " : received an invalid filterId: " + to_string(filterId_), invalid_argument);
}
