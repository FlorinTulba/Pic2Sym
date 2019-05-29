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

#ifndef H_MATCH_ENGINE_BASE
#define H_MATCH_ENGINE_BASE

#include "misc.h"

#pragma warning(push, 0)

#include <string>

#pragma warning(pop)

// Forward declarations
class IBestMatch;
class IMatchSupport;
class MatchAssessor;
class MatchProgress;
class AbsJobMonitor;

/// MatchEngine finds best match for a patch based on current settings and
/// symbols set.
class IMatchEngine /*abstract*/ {
 public:
  /**
  @return the type of the symbols determined by fe & cfg
  @throw logic_error only in UnitTesting for incomplete font configuration
  @throw domain_error for invalid font size

  Exceptions to be only reported, not handled
  */
  virtual std::string getIdForSymsToUse() const noexcept(!UT) = 0;

  /// To be displayed in CmapView's status bar
  virtual unsigned getSymsCount() const noexcept = 0;

  /// Access to the const methods of the matchAssessor
  virtual const MatchAssessor& assessor() const noexcept = 0;

  /// Access to all the methods of the matchAssessor
  virtual MatchAssessor& mutableAssessor() const noexcept = 0;

  virtual IMatchSupport& support() noexcept = 0;  ///< access to matchSupport

  /**
  Using different charmap - also useful for displaying these changes

  @throw logic_error for incomplete font configuration
  @throw NormalSymsLoadingFailure
  @throw TinySymsLoadingFailure

  Exception handled, so no rapid termination via noexcept
  */
  virtual void updateSymbols() = 0;

  /**
  Called before a series of improvesBasedOnBatch
  @throw logic_error for incomplete font configuration
  @throw NormalSymsLoadingFailure
  @throw TinySymsLoadingFailure

  Let the exceptions be handled, so no noexcept
  */
  virtual void getReady() = 0;

  /**
  @return true if a new better match is found within the new batch of symbols

  @throw invalid_argument if upperSymIdx is too large or if draftMatch is
  uniformous

  Exception to be only reported, not handled
  */
  virtual bool improvesBasedOnBatch(
      unsigned fromSymIdx,   ///< start of the batch
      unsigned upperSymIdx,  ///< end of the batch (exclusive)
      IBestMatch&
          draftMatch,  ///< draft for normal/tiny symbols (hopefully improved by
                       ///< a match with a symbol from the new batch)
      MatchProgress&
          matchProgress  ///< observer notified for each new improved match
      ) const noexcept(!UT) = 0;

  /**
  Unicode glyphs are logged as symbols, the rest as their code
  @throw logic_error only in UnitTesting for incomplete font configuration

  Exception to be only reported, not handled
  */
  virtual bool usesUnicode() const noexcept(!UT) = 0;

  /// Clustering should be avoided when the obtained clusters are really small
  virtual const bool& isClusteringUseful() const noexcept = 0;

  /// Setting the symbols monitor
  virtual IMatchEngine& useSymsMonitor(
      AbsJobMonitor& symsMonitor_) noexcept = 0;

  virtual ~IMatchEngine() noexcept {}

  // Slicing prevention
  IMatchEngine(const IMatchEngine&) = delete;
  IMatchEngine(IMatchEngine&&) = delete;
  IMatchEngine& operator=(const IMatchEngine&) = delete;
  IMatchEngine& operator=(IMatchEngine&&) = delete;

 protected:
  constexpr IMatchEngine() noexcept {}
};

#endif  // H_MATCH_ENGINE_BASE
