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

#ifndef H_SYM_FILTER
#define H_SYM_FILTER

#include "misc.h"
#include "symFilterBase.h"

#pragma warning(push, 0)

#include <memory>
#include <string>
#include <unordered_map>

#include <gsl/gsl>

#pragma warning(pop)

namespace pic2sym::syms::inline filter {

/// Base class of the template class TSymFilter from below, to keep the template
/// as thin as possible
class SymFilter /*abstract*/ : public ISymFilter {
 public:
  /**
  @return the name of the filter identified by filterId_
  @throw invalid_argument for an invalid filterId_

  Exception to be only reported, not handled
  */
  static const std::string& filterName(unsigned filterId_) noexcept(!UT);

  // Slicing prevention
  SymFilter(const SymFilter&) = delete;
  SymFilter(SymFilter&&) = delete;
  void operator=(const SymFilter&) = delete;
  void operator=(SymFilter&&) = delete;

 private:
  // SymFilter cannot be directly derived, except by the friend TSymFilter
  // declared below
  template <class T>
  friend class TSymFilter;

  /**
  Constructs a new SymFilter with the provided id and name, which both must be
  unique.

  @param filterId_ id of the filter
  @param filterName name of the filter
  @param nextFilter_ optional successor filter

  @throw invalid_argument for duplicate filterId_ or filterName

  Exception to be only reported, not handled
  */
  SymFilter(unsigned filterId_,
            const std::string& filterName,
            std::unique_ptr<ISymFilter> nextFilter_) noexcept(!UT);

  /// filterId - filterName associations
  static std::unordered_map<unsigned, const std::string> filterTypes;

  /// DefSymFilter or a derivate from SymFilter
  std::unique_ptr<ISymFilter> nextFilter;

  unsigned filterId;  ///< id of the filter
};

/**
Base class for any filters which can be applied on symbols.
TSymFilter is a template layer over SymFilter to use CRTP.
Normal polymorphism isn't necessary, as the used filters don't have state,
so static methods would be enough - that is static polymorphism does the job.

As a consequence, derived classes from TSymFilter must have 2 public methods
with following signature:
- static bool isEnabled() noexcept (using SYM_FILTER_DECLARE_IS_ENABLED define)
- static bool isDisposable(const IPixMapSym &pms, const SymFilterCache &sfc)
noexcept
*/
template <class DerivedFromTSymFilter>
class TSymFilter /*abstract*/ : public SymFilter {
 public:
  /**
  Returns the id of the filter which detected that the symbol exhibits some
  undesired features.

  Derived classes from TSymFilter must have 2 public methods with following
  signature:
  - static bool isEnabled() noexcept (using SYM_FILTER_DECLARE_IS_ENABLED
  define)
  - static bool isDisposable(const IPixMapSym& pms, const SymFilterCache& sfc)
  noexcept
  */
  std::optional<unsigned> matchingFilterId(
      const IPixMapSym& pms,
      const SymFilterCache& sfc) const noexcept override {
    // Using static polymorphism
    if (DerivedFromTSymFilter::isDisposable(pms, sfc))
      return filterId;

    return nextFilter->matchingFilterId(pms, sfc);
  }

 protected:
  /**
  Constructs a new TSymFilter with the provided id and name, which both must be
  unique.

  @param filterId_ id of the filter
  @param filterName name of the filter
  @param nextFilter_ optional successor filter

  @throw invalid_argument only in UnitTesting for a non-unique filterId_ or for
  a non-unique filterName

  Exceptions caught only in UnitTesting
  */
  TSymFilter(unsigned filterId_,
             const std::string& filterName,
             std::unique_ptr<ISymFilter> nextFilter_) noexcept(!UT)
      : SymFilter{filterId_, filterName, std::move(nextFilter_)} {}
};

/// MacroS for defining DerivedFromTSymFilter::isEnabled() method within each
/// DerivedFromTSymFilter class
#define SYM_FILTER_DECLARE_IS_ENABLED(DerivedFromTSymFilter) \
  static bool isEnabled() noexcept

#define SYM_FILTER_DEFINE_IS_ENABLED(DerivedFromTSymFilter)        \
  extern const bool DerivedFromTSymFilter##Enabled;                \
                                                                   \
  bool syms::filter::DerivedFromTSymFilter::isEnabled() noexcept { \
    return DerivedFromTSymFilter##Enabled;                         \
  }

}  // namespace pic2sym::syms::inline filter

#endif  // H_SYM_FILTER
