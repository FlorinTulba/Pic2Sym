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

#include "precompiled.h"
// This keeps precompiled.h first; Otherwise header sorting might move it

#include "matchEngine.h"

#include "bestMatchBase.h"
#include "clusterEngine.h"
#include "clusterSupport.h"
#include "cmapPerspectiveBase.h"
#include "fontEngineBase.h"
#include "jobMonitorBase.h"
#include "matchAspectsFactory.h"
#include "matchAssessment.h"
#include "matchParamsBase.h"
#include "matchProgress.h"
#include "matchSupportBase.h"
#include "misc.h"
#include "ompTrace.h"
#include "patchBase.h"
#include "preselectManager.h"
#include "scoreThresholds.h"
#include "settingsBase.h"
#include "symSettingsBase.h"
#include "symbolsSupportBase.h"
#include "taskMonitor.h"

#ifndef UNIT_TESTING

#include "warnings.h"

#pragma warning(push, 0)

// The project uses parallelism
#include <omp.h>

#pragma warning(pop)

#else  // UNIT_TESTING defined

// Unit Tests don't use parallelism, to ensure that at least the sequential code
// works as expected

#pragma warning(disable : WARN_INCONSISTENT_DLL_LINKAGE)

// returns 0 - the index of the unique thread used
extern int __cdecl omp_get_thread_num(void);

#pragma warning(default : WARN_INCONSISTENT_DLL_LINKAGE)

#endif  // UNIT_TESTING

using namespace std;
using namespace cv;
using namespace gsl;

namespace pic2sym {

extern const bool UseSkipMatchAspectsHeuristic;
extern unsigned TinySymsSz();
extern const bool PrepareMoreGlyphsAtOnce;
extern const double InvestigateClusterEvenForInferiorScoreFactor;

using namespace syms;

namespace match {

namespace {
/// Returns a configured instance of MatchAssessorNoSkip or MatchAssessorSkip,
/// depending on UseSkipMatchAspectsHeuristic
MatchAssessor& specializedInstance(
    const vector<unique_ptr<const MatchAspect>>& availAspects_) noexcept {
  if (UseSkipMatchAspectsHeuristic) {
    static MatchAssessorSkip instSkip;
    return instSkip.availableAspects(availAspects_);
  }

  static MatchAssessorNoSkip instNoSkip;

  return instNoSkip.availableAspects(availAspects_);
}

/// Determines the cluster and the symbol within it corresponding to symIdx
void locateIdx(const set<unsigned>& clusterOffsets,
               unsigned symIdx,
               unsigned& clusterIdx,
               unsigned& symIdxWithinCluster) noexcept {
  const auto it = --clusterOffsets.upper_bound(symIdx);
  clusterIdx = narrow_cast<unsigned>(distance(clusterOffsets.cbegin(), it));
  symIdxWithinCluster = symIdx - *it;
}

/// Checks if the provided symbol range within the current cluster contains a
/// better match
bool checkRangeWithinCluster(unsigned fromIdx,
                             unsigned lastIdx,
                             const IMatchEngine& me,
                             const Mat& toApprox,
                             const VSymData& symsSet,
                             const p2s::transform::CachedData& cd,
                             IScoreThresholds& scoresToBeatBySyms,
                             IBestMatch& draftMatch) noexcept {
  // method called only locally from MatchEngine::improvesBasedOnBatch(), where
  // the draftMatch is checked to correspond to a non-uniform patch, which has
  // mp non-nullptr
  IMatchParamsRW& mp = draftMatch.refParams();

  bool betterMatchFound{false};
  const MatchAssessor& assessor = me.assessor();
  for (unsigned idx{fromIdx}; idx <= lastIdx; ++idx) {
    mp.reset();  // preserves patch-invariant fields
    const ISymData& symData = *symsSet[(size_t)idx];
    double score;
    if (assessor.isBetterMatch(toApprox, symData, cd, scoresToBeatBySyms, mp,
                               score)) {
      draftMatch.update(score, symData.getCode(), idx, symData);
      assessor.scoresToBeat(score, scoresToBeatBySyms);
      betterMatchFound = true;
    }
  }
  return betterMatchFound;
}
}  // anonymous namespace

MatchEngine::MatchEngine(const p2s::cfg::ISettings& cfg_,
                         IFontEngine& fe_,
                         p2s::ui::ICmapPerspective& cmP_) noexcept
    : IMatchEngine(),
      cfg(&cfg_),
      fe(&fe_),
      cmP(&cmP_),
      matchAssessor(&specializedInstance(availAspects)),
      ce(make_unique<ClusterEngine>(fe_, symsSet)),
      matchSupport{
          p2s::transform::IPreselManager::concrete().createMatchSupport(
              cachedData,
              symsSet,
              *matchAssessor,
              cfg_.getMS())} {
  Ensures(matchSupport);  // with preselection or not, but not nullptr

  for (const string& aspectName : MatchAspect::aspectNames())
    availAspects.emplace_back(
        not_null(owner<const MatchAspect*>(
                     MatchAspectsFactory::create(aspectName, cfg_.getMS())))
            .get());

  matchAssessor->updateEnabledMatchAspectsCount();
}

void MatchEngine::updateSymbols() {
  const string idForSymsToUse{getIdForSymsToUse()};

  if (symsIdReady == idForSymsToUse)
    return;  // already up to date

  static p2s::ui::TaskMonitor fieldsComputations{
      "computing specific symbol-related values", *symsMonitor};

  symsSet.clear();
  const p2s::syms::VPixMapSym& rawSyms = fe->symsSet();
  const int symsCount{narrow_cast<int>(size(rawSyms))};
  symsSet.reserve((size_t)symsCount);

  fieldsComputations.setTotalSteps((size_t)symsCount);

  const unsigned sz{cfg->getSS().getFontSz()};

#pragma warning(disable : WARN_CODE_ANALYSIS_IGNORES_OPENMP)
#pragma omp parallel if (PrepareMoreGlyphsAtOnce)
#pragma omp for schedule(static, 1) nowait
  // ordered would be useful above only for debugging (OMP_PRINTF)

  for (int i{}; i < symsCount; ++i) {
    OMP_PRINTF(PrepareMoreGlyphsAtOnce, "glyph %d", i);

    {
      // Computing SymData fields separately, to keep the critical push_back
      // from below as short as possible
      unique_ptr<const ISymData> newSym =
          make_unique<const SymData>(*rawSyms[(size_t)i], sz, false);

#pragma omp critical
      // ordered instead of critical would be useful above only for debugging

      symsSet.push_back(move(newSym));
    }

    // #pragma omp master not allowed in for
    if (!omp_get_thread_num())
      fieldsComputations.taskAdvanced((size_t)i);
  }
#pragma warning(default : WARN_CODE_ANALYSIS_IGNORES_OPENMP)

  fieldsComputations.taskDone();

  ce->support().groupSyms(fe->getFontType());
  cmP->reset(symsSet, ce->getSymsIndicesPerCluster());

  symsIdReady = idForSymsToUse;  // ready to use the new cmap&size
}

unsigned MatchEngine::getSymsCount() const noexcept {
  return narrow_cast<unsigned>(size(symsSet));
}

const MatchAssessor& MatchEngine::assessor() const noexcept {
  return *matchAssessor;
}

MatchAssessor& MatchEngine::mutableAssessor() const noexcept {
  return *matchAssessor;
}

IMatchSupport& MatchEngine::support() noexcept {
  return *matchSupport;
}

void MatchEngine::getReady() {
  updateSymbols();
  // might throw logic_error, NormalSymsLoadingFailure, TinySymsLoadingFailure

  matchSupport->updateCachedData(cfg->getSS().getFontSz(), *fe);
  matchAssessor->getReady(cachedData);
}

bool MatchEngine::isClusteringUseful() const noexcept {
  return ce->worthGrouping();
}

#pragma warning(disable : WARN_THROWS_ALTHOUGH_NOEXCEPT)
bool MatchEngine::improvesBasedOnBatch(unsigned fromSymIdx,
                                       unsigned upperSymIdx,
                                       IBestMatch& draftMatch,
                                       MatchProgress& matchProgress) const
    noexcept(!UT) {
  EXPECTS_OR_REPORT_AND_THROW_CONST_MSG(
      upperSymIdx <= getSymsCount(), invalid_argument,
      HERE.function_name() +
          " - upperSymIdx cannot point beyond the set of used symbols!"s);
  EXPECTS_OR_REPORT_AND_THROW_CONST_MSG(
      draftMatch.getPatch().nonUniform(), invalid_argument,
      HERE.function_name() + " called for uniform patch!"s);

  IMatchParamsRW& mp = draftMatch.refParams();

  if (!matchAssessor->enabledMatchAspectsCount())
    return false;  // Makes no sense comparing without enabled matching aspects

  const p2s::transform::CachedData& cd = matchSupport->cachedData();
  const VSymData& inspectedSet = ce->support().clusteredSyms();

  const Mat& toApprox = draftMatch.getPatch().matrixToApprox();
  ScoreThresholds scoresToBeatBySyms;
  matchAssessor->scoresToBeat(draftMatch.getScore(), scoresToBeatBySyms);

  double score{};
  bool betterMatchFound{false};
  if (ce->worthGrouping()) {
    // Matching is performed first with clusters and only afterwards with
    // individual symbols
    unsigned fromCluster, firstSymIdxWithinFromCluster, lastCluster,
        lastSymIdxWithinLastCluster;
    locateIdx(ce->getClusterOffsets(), fromSymIdx, fromCluster,
              firstSymIdxWithinFromCluster);
    locateIdx(ce->getClusterOffsets(), upperSymIdx - 1, lastCluster,
              lastSymIdxWithinLastCluster);

    const VClusterData& clusters = ce->getClusters();
    const bool previouslyQualified{
        (clusters[(size_t)fromCluster]->getSz() > 1U) &&
        (draftMatch.getLastPromisingNontrivialCluster() == fromCluster)};

    // Multi-element clusters still qualify with slightly inferior scores,
    // as individual symbols within the cluster might deliver a superior score.
    ScoreThresholds scoresToBeatByClusters{
        InvestigateClusterEvenForInferiorScoreFactor, scoresToBeatBySyms};

    // 1st cluster might have already been qualified for thorough examination
    if (previouslyQualified) {  // cluster already qualified
      const unsigned upperLimit{(fromCluster < lastCluster)
                                    ? clusters[(size_t)fromCluster]->getSz()
                                    : lastSymIdxWithinLastCluster};
      if (checkRangeWithinCluster(firstSymIdxWithinFromCluster, upperLimit,
                                  *this, toApprox, inspectedSet, cd,
                                  scoresToBeatBySyms, draftMatch)) {
        scoresToBeatByClusters.update(
            InvestigateClusterEvenForInferiorScoreFactor, scoresToBeatBySyms);
        matchProgress.remarkedMatch(*draftMatch.getSymIdx(),
                                    draftMatch.getScore());

        betterMatchFound = true;
      }

      ++fromCluster;  // nothing else to investigate from this cluster

    } else if (firstSymIdxWithinFromCluster > 0U) {
      // If cluster fromCluster was already analyzed, but wasn't considered
      // worthy enough to investigate symbol by symbol, increment fromCluster
      ++fromCluster;
    }

    // Examine all remaining unchecked clusters (if any) within current batch
    for (unsigned clusterIdx{fromCluster}; clusterIdx <= lastCluster;
         ++clusterIdx) {
      const unique_ptr<const IClusterData>& cluster =
          clusters[(size_t)clusterIdx];

      // Current cluster attempts qualification - it computes its own score
      mp.reset();  // preserves patch-invariant fields

      if (cluster->getSz() == 1U) {  // Trivial cluster
        // Single element clusters have same score as their content.
        const unsigned symIdx{cluster->getIdxOfFirstSym()};
        const ISymData& symData = *inspectedSet[(size_t)symIdx];
        if (matchAssessor->isBetterMatch(toApprox, symData, cd,
                                         scoresToBeatBySyms, mp, score)) {
          draftMatch.update(score, symData.getCode(), symIdx, symData);
          matchAssessor->scoresToBeat(score, scoresToBeatBySyms);
          scoresToBeatByClusters.update(
              InvestigateClusterEvenForInferiorScoreFactor, scoresToBeatBySyms);
          matchProgress.remarkedMatch(symIdx, score);

          betterMatchFound = true;
        }

      } else {  // Nontrivial cluster
        if (matchAssessor->isBetterMatch(toApprox, *cluster, cd,
                                         scoresToBeatByClusters, mp, score)) {
          // cluster is a selected candidate
          draftMatch.setLastPromisingNontrivialCluster(clusterIdx);

          const unsigned upperLimit{(clusterIdx < lastCluster)
                                        ? cluster->getSz()
                                        : lastSymIdxWithinLastCluster};
          if (checkRangeWithinCluster(0U, upperLimit, *this, toApprox,
                                      inspectedSet, cd, scoresToBeatBySyms,
                                      draftMatch)) {
            scoresToBeatByClusters.update(
                InvestigateClusterEvenForInferiorScoreFactor,
                scoresToBeatBySyms);
            matchProgress.remarkedMatch(*draftMatch.getSymIdx(),
                                        draftMatch.getScore());

            betterMatchFound = true;
          }
        }
      }
    }

  } else {
    // Matching is performed directly with individual symbols, not with clusters

    // Examine all remaining symbols within current batch
    for (unsigned symIdx{fromSymIdx}; symIdx < upperSymIdx; ++symIdx) {
      const ISymData& symData = *inspectedSet[(size_t)symIdx];

      mp.reset();  // preserves patch-invariant fields

      if (matchAssessor->isBetterMatch(toApprox, symData, cd,
                                       scoresToBeatBySyms, mp, score)) {
        draftMatch.update(score, symData.getCode(), symIdx, symData);
        matchAssessor->scoresToBeat(score, scoresToBeatBySyms);
        matchProgress.remarkedMatch(symIdx, score);

        betterMatchFound = true;
      }
    }
  }

  if (betterMatchFound)
    draftMatch.updatePatchApprox(cfg->getMS());

  return betterMatchFound;
}
#pragma warning(default : WARN_THROWS_ALTHOUGH_NOEXCEPT)

MatchEngine& MatchEngine::useSymsMonitor(
    p2s::ui::AbsJobMonitor& symsMonitor_) noexcept {
  symsMonitor = &symsMonitor_;
  ce->useSymsMonitor(symsMonitor_);
  return *this;
}

#ifdef _DEBUG

bool MatchEngine::usesUnicode() const noexcept(!UT) {
  return fe->getEncoding() == "UNICODE";
  // getEncoding() throws logic_error for incomplete font configuration
}

#else  // _DEBUG not defined

bool MatchEngine::usesUnicode() const noexcept(!UT) {
  return true;
}

#endif  // _DEBUG

}  // namespace match
}  // namespace pic2sym
