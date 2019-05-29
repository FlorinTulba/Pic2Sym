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

#include "matchAspects.h"
#include "matchAssessment.h"
#include "misc.h"
#include "scoreThresholdsBase.h"

using namespace std;
using namespace cv;

extern const double EnableSkipAboveMatchRatio;

MatchAssessor& MatchAssessor::availableAspects(
    const vector<std::unique_ptr<const MatchAspect>>& availAspects_) noexcept {
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
  if (availAspects != nullptr) {
    for (const unique_ptr<const MatchAspect>& pAspect : *availAspects)
      if (pAspect->enabled())
        ++enabledAspectsCount;
  }

  enabledAspectsCountM1 = enabledAspectsCount - 1ULL;
}

size_t MatchAssessor::enabledMatchAspectsCount() const noexcept {
  return enabledAspectsCount;
}

void MatchAssessor::getReady(const CachedData& /*cachedData*/) noexcept {
  enabledAspects.clear();
  if (availAspects != nullptr) {
    for (const unique_ptr<const MatchAspect>& pAspect : *availAspects)
      if (pAspect->enabled())
        enabledAspects.push_back(&*pAspect);
  }

  // Desired effects
  assert(enabledAspectsCount == enabledAspects.size());
  assert(enabledAspectsCountM1 == enabledAspectsCount - 1ULL);
}

bool MatchAssessor::isBetterMatch(const Mat& patch,
                                  const ISymData& symData,
                                  const CachedData& cd,
                                  const IScoreThresholds& scoresToBeat,
                                  IMatchParamsRW& mp,
                                  double& score) const noexcept {
  if (enabledAspectsCount == 0ULL)
    return false;  // comparing makes no sense when no enabled matching aspects

  assert(enabledAspectsCount == enabledAspects.size());  // check their sync

  score = enabledAspects[0ULL]->assessMatch(patch, symData, cd, mp);
  for (size_t i = 1ULL; i < enabledAspectsCount; ++i)
    score *= enabledAspects[i]->assessMatch(patch, symData, cd, mp);
  return score > scoresToBeat.overall();
}

MatchAssessorNoSkip::MatchAssessorNoSkip() noexcept : MatchAssessor() {}

void MatchAssessorNoSkip::scoresToBeat(double draftScore,
                                       IScoreThresholds& scoresToBeat) const
    noexcept {
  scoresToBeat.update(draftScore);
}

MatchAssessorSkip::MatchAssessorSkip() noexcept : MatchAssessor() {}

void MatchAssessorSkip::getReady(const CachedData& cachedData) noexcept {
  MatchAssessor::getReady(cachedData);

  sort(BOUNDS(enabledAspects), [&](const MatchAspect* a,
                                   const MatchAspect* b) noexcept->bool {
    const double relComplexityA = a->relativeComplexity(),
                 relComplexityB = b->relativeComplexity();
    // Ascending by complexity
    if (relComplexityA < relComplexityB)
      return true;

    if (relComplexityA > relComplexityB)
      return false;

    // Equal complexity here already

    const double maxScoreA = a->maxScore(cachedData),
                 maxScoreB = b->maxScore(cachedData);

    // Descending by max score
    return maxScoreA >= maxScoreB;
  });

#ifdef MONITOR_SKIPPED_MATCHING_ASPECTS
  totalIsBetterMatchCalls = 0ULL;
  skippedAspects.assign(enabledAspectsCount, 0ULL);
#endif  // MONITOR_SKIPPED_MATCHING_ASPECTS

  // Adjust max increase factors for every enabled aspect except last
  invMaxIncreaseFactors.resize(enabledAspectsCountM1);
  double maxIncreaseFactor = 1.;
  for (int ip1 = (int)enabledAspectsCountM1, i = ip1 - 1; i >= 0; ip1 = i--) {
    maxIncreaseFactor *= enabledAspects[(size_t)ip1]->maxScore(cachedData);
    invMaxIncreaseFactors[(size_t)i] = 1. / maxIncreaseFactor;
  }

  // Setting the threshold as a % from the ideal score
  thresholdDraftScore = maxIncreaseFactor *

                        // Max score for the ideal match
                        enabledAspects[0ULL]->maxScore(cachedData) *
                        EnableSkipAboveMatchRatio;
}

void MatchAssessorSkip::scoresToBeat(double draftScore,
                                     IScoreThresholds& scoresToBeat) const
    noexcept {
  if (draftScore < thresholdDraftScore) {
    // For bad matches intermediary results won't be used
    scoresToBeat.inferiorMatch();
    scoresToBeat.update(draftScore);
  } else {
    scoresToBeat.update(draftScore, invMaxIncreaseFactors);
  }
}

bool MatchAssessorSkip::isBetterMatch(const Mat& patch,
                                      const ISymData& symData,
                                      const CachedData& cd,
                                      const IScoreThresholds& scoresToBeat,
                                      IMatchParamsRW& mp,
                                      double& score) const noexcept {
  if (enabledAspectsCount == 0ULL)
    return false;  // comparing makes no sense when no enabled matching aspects

  assert(enabledAspectsCount == enabledAspects.size());  // check their sync

#ifdef MONITOR_SKIPPED_MATCHING_ASPECTS
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
#endif  // MONITOR_SKIPPED_MATCHING_ASPECTS

  // Until finding a good match, Aspects Skipping heuristic won't be used
  if (scoresToBeat.representsInferiorMatch())
    return MatchAssessor::isBetterMatch(patch, symData, cd, scoresToBeat, mp,
                                        score);

  score = enabledAspects[0ULL]->assessMatch(patch, symData, cd, mp);

  for (size_t im1 = 0ULL, i = 1ULL; i <= enabledAspectsCountM1; im1 = i++) {
    if (score < scoresToBeat[im1]) {
#ifdef MONITOR_SKIPPED_MATCHING_ASPECTS
      for (size_t j = i; j <= enabledAspectsCountM1; ++j) {
#pragma omp atomic  // See comment from previous

        // MONITOR_SKIPPED_MATCHING_ASPECTS region
        ++skippedAspects[j];
      }
#endif  // MONITOR_SKIPPED_MATCHING_ASPECTS

      // Skip further aspects checking when score can't beat best match score
      return false;
    }
    score *= enabledAspects[i]->assessMatch(patch, symData, cd, mp);
  }

  return score > scoresToBeat.overall();
}
