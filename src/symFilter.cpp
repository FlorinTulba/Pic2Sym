/******************************************************************************
 The application Pic2Sym approximates images by a
 grid of colored symbols with colored backgrounds.

 Copyrights from the libraries used by the program:
 - (c) 2003 Boost (www.boost.org)
     License: doc/licenses/Boost.lic
     http://www.boost.org/LICENSE_1_0.txt
 - (c) 2015-2016 OpenCV (www.opencv.org)
     License: doc/licenses/OpenCV.lic
     http://opencv.org/license/
 - (c) 1996-2002, 2006 The FreeType Project (www.freetype.org)
     License: doc/licenses/FTL.txt
     http://git.savannah.gnu.org/cgit/freetype/freetype2.git/plain/docs/FTL.TXT
 - (c) 1997-2002 OpenMP Architecture Review Board (www.openmp.org)
   (c) Microsoft Corporation (implementation for OpenMP C/C++ v2.0 March 2002)
     See: https://msdn.microsoft.com/en-us/library/8y6825x5.aspx
 - (c) 1995-2017 zlib software (Jean-loup Gailly and Mark Adler - www.zlib.net)
     License: doc/licenses/zlib.lic
     http://www.zlib.net/zlib_license.html


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
 If not, see: http://www.gnu.org/licenses/agpl-3.0.txt .
 *****************************************************************************/

#include "precompiled.h"

#include "pixMapSymBase.h"
#include "symFilter.h"
#include "symFilterCache.h"
#include "warnings.h"

#pragma warning(push, 0)

#include <unordered_set>

#pragma warning(pop)

using namespace std;

extern template class unordered_set<string, hash<string>>;

unordered_map<unsigned, const string> SymFilter::filterTypes;

#pragma warning(disable : WARN_THROWS_ALTHOUGH_NOEXCEPT)
SymFilter::SymFilter(unsigned filterId_,
                     const string& filterName,
                     unique_ptr<ISymFilter> nextFilter_) noexcept(!UT)
    : ISymFilter(),
      nextFilter(nextFilter_ ? move(nextFilter_) : make_unique<DefSymFilter>()),
      filterId(filterId_) {
  static unordered_set<string, hash<string>> filterNames;

#ifndef UNIT_TESTING
  if (filterTypes.find(filterId_) != filterTypes.end())
    THROW_WITH_VAR_MSG(__FUNCTION__ " called with alreay existing filterId_: " +
                           to_string(filterId_),
                       invalid_argument);

  if (filterNames.find(filterName) != filterNames.end())
    THROW_WITH_VAR_MSG(
        __FUNCTION__ " called with alreay existing filterName: " + filterName,
        invalid_argument);
#endif  // UNIT_TESTING not defined

  filterTypes.emplace(filterId_, filterName);
  filterNames.insert(filterName);
}
#pragma warning(default : WARN_THROWS_ALTHOUGH_NOEXCEPT)

#pragma warning(disable : WARN_THROWS_ALTHOUGH_NOEXCEPT)
const string& SymFilter::filterName(unsigned filterId_) noexcept(!UT) {
  if (auto it = filterTypes.find(filterId_), itEnd = filterTypes.end();
      it != itEnd)
    return it->second;

  THROW_WITH_VAR_MSG(
      __FUNCTION__ " : received an invalid filterId: " + to_string(filterId_),
      invalid_argument);
}
#pragma warning(default : WARN_THROWS_ALTHOUGH_NOEXCEPT)
