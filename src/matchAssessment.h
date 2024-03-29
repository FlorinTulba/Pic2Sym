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

#ifndef H_MATCH_ASSESSMENT
#define H_MATCH_ASSESSMENT

#include "cachedData.h"
#include "match.h"
#include "matchParamsBase.h"
#include "scoreThresholdsBase.h"
#include "symbolsSupportBase.h"

#ifndef UNIT_TESTING
#include "countSkippedAspects.h"
#endif  // UNIT_TESTING not defined

#pragma warning(push, 0)

#include <memory>
#include <vector>

#include <gsl/gsl>

#include <opencv2/core/core.hpp>

#pragma warning(pop)

extern template class std::vector<size_t>;
extern template class std::vector<double>;

namespace pic2sym::match {

/**
Match manager based on the enabled matching aspects:
- selects and rearranges the enabled matching aspects
- applies them while approximating each patch
*/
class MatchAssessor /*abstract*/ {
 public:
  MatchAssessor() noexcept {}

  virtual ~MatchAssessor() noexcept = 0 {}

  // Slicing prevention
  MatchAssessor(const MatchAssessor&) = delete;
  MatchAssessor(MatchAssessor&&) = delete;
  void operator=(const MatchAssessor&) = delete;
  void operator=(MatchAssessor&&) = delete;

  MatchAssessor& availableAspects(
      const std::vector<std::unique_ptr<const MatchAspect>>&
          availAspects_) noexcept;

  void newlyEnabledMatchAspect() noexcept;   ///< increments enabledAspectsCount
  void newlyDisabledMatchAspect() noexcept;  ///< decrements enabledAspectsCount

  /// Updates enabledAspectsCount by checking which aspects are enabled
  void updateEnabledMatchAspectsCount() noexcept;

  /// Provides enabledAspectsCount
  std::size_t enabledMatchAspectsCount() const noexcept;

  /**
  Prepares the enabled aspects for an image transformation process.

  @param cachedData some precomputed values
  */
  virtual void getReady(const transform::CachedData& cachedData) noexcept;

  /// Determines if symData is a better match for patch than previous matching
  /// symbol
  virtual bool isBetterMatch(
      const cv::Mat& patch,  ///< the patch whose approximation through a symbol
                             ///< is performed
      const syms::ISymData&
          symData,  ///< data of the new symbol/cluster compared to the patch
      const transform::CachedData& cd,       ///< precomputed values
      const IScoreThresholds& scoresToBeat,  ///< scores after each aspect that
                                             ///< beat the current best match
      IMatchParamsRW& mp,  ///< matching parameters resulted from the comparison
      double& score        ///< achieved score of the new assessment
  ) const noexcept;

  /**
  Sets the threshold scores which might spare a symbol from evaluating further
  matching aspects.

  @param draftScore a new reference score
  @param scoresToBeat the mentioned threshold scores
  */
  virtual void scoresToBeat(double draftScore,
                            IScoreThresholds& scoresToBeat) const noexcept = 0;

#if defined(MONITOR_SKIPPED_MATCHING_ASPECTS) && !defined(UNIT_TESTING)

  /// MatchAssessorSkip will report all skipped matching aspects
  virtual void reportSkippedAspects() const noexcept {}

#endif  // defined(MONITOR_SKIPPED_MATCHING_ASPECTS) && !defined(UNIT_TESTING)

 protected:
  /// The available matching aspects, enabled or not
  const std::vector<std::unique_ptr<const MatchAspect>>* availAspects = nullptr;

  /// the enabled aspects
  std::vector<gsl::not_null<const MatchAspect*>> enabledAspects;

  size_t enabledAspectsCount{};  ///< count of the enabled aspects

  /// Count of the enabled aspects minus 1
  size_t enabledAspectsCountM1{(size_t)-1LL};

#if defined(MONITOR_SKIPPED_MATCHING_ASPECTS) && !defined(UNIT_TESTING)

  /// Used for reporting skipped aspects
  mutable size_t totalIsBetterMatchCalls{};

  /// Used for reporting skipped aspects
  mutable std::vector<size_t> skippedAspects;

#endif  // defined(MONITOR_SKIPPED_MATCHING_ASPECTS) && !defined(UNIT_TESTING)
};

/// MatchAssessor version when UseSkipMatchAspectsHeuristic is false
class MatchAssessorNoSkip : public MatchAssessor {
 public:
  MatchAssessorNoSkip() noexcept;

  /**
  @param draftScore a new reference score
  @param scoresToBeat the mentioned threshold scores
  */
  void scoresToBeat(double draftScore,
                    IScoreThresholds& scoresToBeat) const noexcept override;
};

/// MatchAssessor version when UseSkipMatchAspectsHeuristic is true
class MatchAssessorSkip : public MatchAssessor {
 public:
  MatchAssessorSkip() noexcept;

  /**
  Prepares the enabled aspects for an image transformation process.

  @param cachedData some precomputed values

  It also rearranges these enabled aspects based on their complexity and their
  max score.

  The goal is to skip as many complex aspects as possible while checking if a
  new symbol is a better match for a given patch than a previous symbol with a
  given score. As soon as it's obvious that the remaining aspects can't raise
  the score above current best score, the assessment is aborted.

  The strategy is most beneficial if the skipped aspects are more complex than
  those evaluated already. Thus, it reorders the enabled aspects as follows:
  - rearrange aspects in increasing order of their complexity
  - for equally complex aspects, consider first those with a higher max score,
    to reduce the chance that other aspects are needed
  */
  void getReady(const transform::CachedData& cachedData) noexcept override;

  /**
  Sets the threshold scores which might spare a symbol from evaluating further
  matching aspects.

  @param draftScore a new reference score
  @param scoresToBeat the mentioned threshold scores
  */
  void scoresToBeat(double draftScore,
                    IScoreThresholds& scoresToBeat) const noexcept override;

  /// Determines if symData is a better match for patch than previous matching
  /// symbol
  bool isBetterMatch(
      const cv::Mat& patch,  ///< the patch whose approximation through a symbol
                             ///< is performed
      const syms::ISymData&
          symData,  ///< data of the new symbol/cluster compared to the patch
      const transform::CachedData& cd,       ///< precomputed values
      const IScoreThresholds& scoresToBeat,  ///< scores after each aspect that
                                             ///< beat the current best match
      IMatchParamsRW& mp,  ///< matching parameters resulted from the comparison
      double& score        ///< achieved score of the new assessment
  ) const noexcept override;

#if defined(MONITOR_SKIPPED_MATCHING_ASPECTS) && !defined(UNIT_TESTING)

  /// Reports all skipped matching aspects
  void reportSkippedAspects() const noexcept override;

#endif  // defined(MONITOR_SKIPPED_MATCHING_ASPECTS) && !defined(UNIT_TESTING)

 private:
  /// 1 over (max possible increase of the score based on remaining aspects)
  std::vector<double> invMaxIncreaseFactors;

  /**
  The Aspects Skipping heuristic comes with a slight cost which becomes
  noticeable when:
  - there are barely any several skipped matching aspects
  - these very few skipped aspects are not complex

  In such cases the heuristic probably won't shorten the transformation time.

  The heuristic starts getting more efficient only after finding a really good
  draft match for the patch. This justifies introducing a threshold for the
  score of a draft (the field thresholdDraftScore). As long as the draft matches
  for a given patch score under this threshold, the heuristic isn't used. The
  first draft match above the threshold enables the heuristic.
  */
  double thresholdDraftScore{};
};

}  // namespace pic2sym::match

#endif  // H_MATCH_ASSESSMENT
