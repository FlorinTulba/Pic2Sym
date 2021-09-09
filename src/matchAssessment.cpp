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

#include "matchAssessment.h"

using namespace std;
using namespace gsl;
using namespace cv;

namespace pic2sym {

extern const double EnableSkipAboveMatchRatio;

using transform::CachedData;

namespace match {

MatchAssessor& MatchAssessor::availableAspects(
    const vector<unique_ptr<const MatchAspect>>& availAspects_) noexcept {
  availAspects = &availAspects_;
  return *this;
}

void MatchAssessor::newlyEnabledMatchAspect() noexcept {
  enabledAspectsCountM1 = enabledAspectsCount++;
}

void MatchAssessor::newlyDisabledMatchAspect() noexcept {
  enabledAspectsCount = enabledAspectsCountM1--;
}

void MatchAssessor::updateEnabledMatchAspectsCount() noexcept {
  enabledAspectsCount = 0ULL;
  if (availAspects) {
    for (const auto& pAspect : *availAspects)
      if (pAspect->enabled())
        ++enabledAspectsCount;
  }

  enabledAspectsCountM1 = enabledAspectsCount - 1ULL;
}

std::size_t MatchAssessor::enabledMatchAspectsCount() const noexcept {
  return enabledAspectsCount;
}

void MatchAssessor::getReady(const CachedData& /*cachedData*/) noexcept {
  enabledAspects.clear();
  if (availAspects) {
    for (const auto& pAspect : *availAspects)
      if (pAspect->enabled())
        enabledAspects.push_back(&*pAspect);
  }

  // Desired effects
  Ensures(enabledAspectsCount == size(enabledAspects));
  Ensures(enabledAspectsCountM1 == enabledAspectsCount - 1ULL);
}

bool MatchAssessor::isBetterMatch(const Mat& patch,
                                  const p2s::syms::ISymData& symData,
                                  const CachedData& cd,
                                  const IScoreThresholds& scoresToBeat,
                                  IMatchParamsRW& mp,
                                  double& score) const noexcept {
  Expects(enabledAspectsCount == size(enabledAspects));  // check the sync

  if (!enabledAspectsCount)
    return false;  // comparing makes no sense when no enabled matching aspects

  score = enabledAspects[0ULL]->assessMatch(patch, symData, cd, mp);
  for (size_t i{1ULL}; i < enabledAspectsCount; ++i)
    score *= enabledAspects[i]->assessMatch(patch, symData, cd, mp);
  return score > scoresToBeat.overall();
}

MatchAssessorNoSkip::MatchAssessorNoSkip() noexcept : MatchAssessor() {}

void MatchAssessorNoSkip::scoresToBeat(
    double draftScore,
    IScoreThresholds& scoresToBeat) const noexcept {
  scoresToBeat.update(draftScore);
}

MatchAssessorSkip::MatchAssessorSkip() noexcept : MatchAssessor() {}

void MatchAssessorSkip::getReady(const CachedData& cachedData) noexcept {
  MatchAssessor::getReady(cachedData);

  ranges::sort(enabledAspects, [&](not_null<const MatchAspect*> a,
                                   not_null<const MatchAspect*> b) noexcept {
    const double relComplexityA{a->relativeComplexity()};
    const double relComplexityB{b->relativeComplexity()};
    // Ascending by complexity
    if (relComplexityA < relComplexityB)
      return true;

    if (relComplexityA > relComplexityB)
      return false;

    // Equal complexity here already

    const double maxScoreA{a->maxScore(cachedData)};
    const double maxScoreB{b->maxScore(cachedData)};

    // Descending by max score
    return maxScoreA >= maxScoreB;
  });

#if defined(MONITOR_SKIPPED_MATCHING_ASPECTS) && !defined(UNIT_TESTING)
  totalIsBetterMatchCalls = 0ULL;
  skippedAspects.assign(enabledAspectsCount, 0ULL);
#endif  // defined(MONITOR_SKIPPED_MATCHING_ASPECTS) && !defined(UNIT_TESTING)

  // Adjust max increase factors for every enabled aspect except last
  invMaxIncreaseFactors.resize(enabledAspectsCountM1);
  double maxIncreaseFactor{1.};
  for (int ip1{narrow_cast<int>(enabledAspectsCountM1)}, i{ip1 - 1}; i >= 0;
       ip1 = i--) {
    maxIncreaseFactor *= enabledAspects[(size_t)ip1]->maxScore(cachedData);
    invMaxIncreaseFactors[(size_t)i] = 1. / maxIncreaseFactor;
  }

  // Setting the threshold as a % from the ideal score
  thresholdDraftScore = maxIncreaseFactor *

                        // Max score for the ideal match
                        enabledAspects[0ULL]->maxScore(cachedData) *
                        EnableSkipAboveMatchRatio;
}

void MatchAssessorSkip::scoresToBeat(
    double draftScore,
    IScoreThresholds& scoresToBeat) const noexcept {
  if (draftScore < thresholdDraftScore) {
    // For bad matches intermediary results won't be used
    scoresToBeat.inferiorMatch();
    scoresToBeat.update(draftScore);
  } else {
    scoresToBeat.update(draftScore, invMaxIncreaseFactors);
  }
}

bool MatchAssessorSkip::isBetterMatch(const Mat& patch,
                                      const p2s::syms::ISymData& symData,
                                      const CachedData& cd,
                                      const IScoreThresholds& scoresToBeat,
                                      IMatchParamsRW& mp,
                                      double& score) const noexcept {
  Expects(enabledAspectsCount == size(enabledAspects));  // check the sync

  if (!enabledAspectsCount)
    return false;  // comparing makes no sense when no enabled matching aspects

#if defined(MONITOR_SKIPPED_MATCHING_ASPECTS) && !defined(UNIT_TESTING)
#pragma omp atomic
  /*
  We're within a `parallel for` here!!

  However this region is suppressed in the final release, so the speed isn't
  that relevant. It matters only to get correct values for the count of skipped
  aspects.

  That's why the `parallel for` doesn't need:
  reduction(+ : totalIsBetterMatchCalls)
  */
  ++totalIsBetterMatchCalls;
#endif  // defined(MONITOR_SKIPPED_MATCHING_ASPECTS) && !defined(UNIT_TESTING)

  // Until finding a good match, Aspects Skipping heuristic won't be used
  if (scoresToBeat.representsInferiorMatch())
    return MatchAssessor::isBetterMatch(patch, symData, cd, scoresToBeat, mp,
                                        score);

  score = enabledAspects[0ULL]->assessMatch(patch, symData, cd, mp);

  for (size_t im1{}, i{1ULL}; i <= enabledAspectsCountM1; im1 = i++) {
    if (score < scoresToBeat[im1]) {
#if defined(MONITOR_SKIPPED_MATCHING_ASPECTS) && !defined(UNIT_TESTING)

      for (size_t j{i}; j <= enabledAspectsCountM1; ++j) {
#pragma omp atomic  // See comment from previous

        ++skippedAspects[j];
      }

#endif  // defined(MONITOR_SKIPPED_MATCHING_ASPECTS) && !defined(UNIT_TESTING)

      // Skip further aspects checking when score can't beat best match score
      return false;
    }
    score *= enabledAspects[i]->assessMatch(patch, symData, cd, mp);
  }

  return score > scoresToBeat.overall();
}

}  // namespace match
}  // namespace pic2sym
