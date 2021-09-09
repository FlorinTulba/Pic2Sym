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

#ifndef H_MATCH_ENGINE
#define H_MATCH_ENGINE

#include "matchEngineBase.h"

#include "cachedData.h"
#include "clusterEngineBase.h"
#include "cmapPerspectiveBase.h"
#include "fontEngineBase.h"
#include "match.h"
#include "settingsBase.h"
#include "symDataBase.h"

namespace pic2sym::match {

/// MatchEngine finds best match for a patch based on current settings and
/// symbols set.
class MatchEngine : public IMatchEngine {
 public:
  MatchEngine(const cfg::ISettings& cfg_,
              syms::IFontEngine& fe_,
              ui::ICmapPerspective& cmP_) noexcept;

  // Slicing prevention
  MatchEngine(const MatchEngine&) = delete;
  MatchEngine(MatchEngine&&) = delete;
  void operator=(const MatchEngine&) = delete;
  void operator=(MatchEngine&&) = delete;

  /**
  @return the type of the symbols determined by fe & cfg
  @throw logic_error only in UnitTesting for incomplete font configuration
  @throw domain_error for invalid font size

  Exceptions to be only reported, not handled
  */
  std::string getIdForSymsToUse() const noexcept(!UT) override;

  /// To be displayed in CmapView's status bar
  unsigned getSymsCount() const noexcept override;

  /// Access to the const methods of the matchAssessor
  const MatchAssessor& assessor() const noexcept override;

  /// Access to all the methods of the matchAssessor
  MatchAssessor& mutableAssessor() const noexcept override;

  IMatchSupport& support() noexcept override;  ///< access to matchSupport

  /**
  Using different charmap - also useful for displaying these changes

  @throw logic_error for incomplete font configuration
  @throw NormalSymsLoadingFailure
  @throw TinySymsLoadingFailure
  @throw AbortedJob

  Exceptions handled, so no rapid termination via noexcept
  */
  void updateSymbols() override;

  /**
  Called before a series of improvesBasedOnBatch
  @throw logic_error for incomplete font configuration
  @throw NormalSymsLoadingFailure
  @throw TinySymsLoadingFailure

  Let the exceptions be handled, so no noexcept
  */
  void getReady() override;

  /**
  @return true if a new better match is found within the new batch of symbols

  @throw invalid_argument if upperSymIdx is too large or if draftMatch is
  uniform

  Exception to be only reported, not handled
  */
  bool improvesBasedOnBatch(
      unsigned fromSymIdx,   ///< start of the batch
      unsigned upperSymIdx,  ///< end of the batch (exclusive)
      IBestMatch&
          draftMatch,  ///< draft for normal/tiny symbols (hopefully improved by
                       ///< a match with a symbol from the new batch)
      MatchProgress&
          matchProgress  ///< observer notified for each new improved match
  ) const noexcept(!UT) override;

  /**
  Unicode glyphs are logged as symbols, the rest as their code
  @throw logic_error only in UnitTesting for incomplete font configuration

  Exception to be only reported, not handled
  */
  bool usesUnicode() const noexcept(!UT) override;

  /// Clustering should be avoided when the obtained clusters are really small
  bool isClusteringUseful() const noexcept override;

  /// Setting the symbols monitor
  MatchEngine& useSymsMonitor(
      ui::AbsJobMonitor& symsMonitor_) noexcept override;

  PRIVATE :

      /// settings for the engine
      gsl::not_null<const cfg::ISettings*>
          cfg;

  gsl::not_null<syms::IFontEngine*> fe;  ///< symbols set manager

  /// Reorganized symbols to be visualized within the cmap viewer
  gsl::not_null<ui::ICmapPerspective*> cmP;

  /// observer of the symbols' loading, filtering and clustering, who reports
  /// their progress
  ui::AbsJobMonitor* symsMonitor = nullptr;

  std::string symsIdReady;  ///< type of symbols ready to use for transformation

  syms::VSymData symsSet;  ///< set of most information on each symbol

  /// Data precomputed by matchSupport before performing the matching series
  transform::CachedDataRW cachedData;

  /// Match manager based on the enabled matching aspects
  gsl::not_null<MatchAssessor*> matchAssessor;

  std::unique_ptr<syms::cluster::IClusterEngine> ce;  ///< clusters manager

  // Keep this below the fields, as it depends on them
  /// Cached data management
  std::unique_ptr<IMatchSupport> matchSupport;

  /// All the available aspects.
  /// vector<not_null<unique_ptr>> not possible because not_null needs
  /// copy-constructible types. not-nullness checked instead when appended.
  std::vector<std::unique_ptr<const MatchAspect>> availAspects;
};

}  // namespace pic2sym::match

#endif  // H_MATCH_ENGINE
