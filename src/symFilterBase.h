/******************************************************************************
 The application Pic2Sym approximates images by a
 grid of colored symbols with colored backgrounds.

 Copyrights from the libraries used by the program:
 - (c) 2003-2021 Boost (www.boost.org)
     License: doc/licenses/Boost.lic
     http://www.boost.org/LICENSE_1_0.txt
 - (c) 2015-2021 OpenCV (www.opencv.org)
     License: doc/licenses/OpenCV.lic
     http://opencv.org/license/
 - (c) 1996-2021 The FreeType Project (www.freetype.org)
     License: doc/licenses/FTL.txt
     http://git.savannah.gnu.org/cgit/freetype/freetype2.git/plain/docs/FTL.TXT
 - (c) 1997-2021 OpenMP Architecture Review Board (www.openmp.org)
   (c) Microsoft Corporation (implementation for OpenMP C/C++ v2.0 March 2002)
     See: https://msdn.microsoft.com/en-us/library/8y6825x5.aspx
 - (c) 1995-2021 zlib software (Jean-loup Gailly and Mark Adler - www.zlib.net)
     License: doc/licenses/zlib.lic
     http://www.zlib.net/zlib_license.html
 - (c) 2015-2021 Microsoft Guidelines Support Library - github.com/microsoft/GSL
     License: doc/licenses/MicrosoftGSL.lic
     https://raw.githubusercontent.com/microsoft/GSL/main/LICENSE


 (c) 2016-2021 Florin Tulba <florintulba@yahoo.com>

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

#ifndef H_SYM_FILTER_BASE
#define H_SYM_FILTER_BASE

#include "pixMapSymBase.h"
#include "symFilterCache.h"

#pragma warning(push, 0)

#include <optional>

#pragma warning(pop)

extern template class std::optional<unsigned>;

namespace pic2sym::syms::inline filter {

/// Interface used for filtering out some of the symbols from the charmap
class ISymFilter /*abstract*/ {
 public:
  virtual ~ISymFilter() noexcept = 0 {}

  /**
  Returns the id of the filter which detected that the symbol exhibits some
  undesired features.
  */
  virtual std::optional<unsigned> matchingFilterId(
      const IPixMapSym&,
      const SymFilterCache&) const noexcept = 0;
};

/// Implicit Symbol Filter, which just approves any symbol and is enabled by
/// default
class DefSymFilter final : public ISymFilter {
 public:
  DefSymFilter() noexcept {}

  /**
  Returns the id of the filter which detected that the symbol exhibits some
  undesired features.
  */
  std::optional<unsigned> matchingFilterId(
      const IPixMapSym&,
      const SymFilterCache&) const noexcept final {
    return std::nullopt;
  }
};

}  // namespace pic2sym::syms::inline filter

#endif  // H_SYM_FILTER_BASE
