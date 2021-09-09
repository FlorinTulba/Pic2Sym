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

#ifndef H_MATCH_SUPPORT_WITH_PRESELECTION
#define H_MATCH_SUPPORT_WITH_PRESELECTION

#include "matchSupport.h"

#include "bestMatchBase.h"
#include "cachedData.h"
#include "matchAssessment.h"
#include "matchSettingsBase.h"
#include "preselCandidates.h"
#include "symDataBase.h"

namespace pic2sym::match {

/// Polymorphic support for the MatchEngine and Transformer classes reflecting
/// the preselection mode.
class MatchSupportWithPreselection : public MatchSupport {
 public:
  /// Filling in the rest of the data required when PreselectionByTinySyms ==
  /// true
  MatchSupportWithPreselection(
      transform::CachedDataRW& cd_,
      syms::VSymData& symsSet_,
      MatchAssessor& matchAssessor_,
      const cfg::IMatchSettings& matchSettings_) noexcept;

  /// cached data corresponding to the tiny size symbols
  const transform::CachedData& cachedData() const noexcept override;

  /// update cached data corresponding to the tiny size symbols
  void updateCachedData(unsigned fontSz,
                        const syms::IFontEngine& fe) noexcept override;

  /**
  @return true if a new better match is found within this short list
  @throw invalid_argument if the draftMatch corresponds to a uniform patch

  Exception to be only reported, not handled
  */
  bool improvesBasedOnBatchShortList(
      transform::CandidatesShortList&&
          shortList,          ///< most promising candidates from
                              ///< current batch of symbols
      IBestMatch& draftMatch  ///< draft for normal symbols (hopefully improved
                              ///< by a match with a symbol from the shortList)
  ) const noexcept(!UT);

  /// No const lvalue parameters accepted for improvesBasedOnBatchShortList
  bool improvesBasedOnBatchShortList(
      const transform::CandidatesShortList& shortList,
      IBestMatch& draftMatch) const = delete;

 private:
  transform::CachedDataRW
      cdPresel;  ///< cached data corresponding to tiny size symbols
  gsl::not_null<syms::VSymData*> symsSet;  ///< the set of normal-size symbols

  /// Match manager based on the enabled matching aspects
  gsl::not_null<MatchAssessor*> matchAssessor;

  gsl::not_null<const cfg::IMatchSettings*> matchSettings;  ///< match settings
};

}  // namespace pic2sym::match

#endif  // H_MATCH_SUPPORT_WITH_PRESELECTION
