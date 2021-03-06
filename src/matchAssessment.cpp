/************************************************************************************************
 The application Pic2Sym approximates images by a
 grid of colored symbols with colored backgrounds.

 Copyrights from the libraries used by the program:
 - (c) 2016 Boost (www.boost.org)
		License: <http://www.boost.org/LICENSE_1_0.txt>
			or doc/licenses/Boost.lic
 - (c) 2015 OpenCV (www.opencv.org)
		License: <http://opencv.org/license.html>
            or doc/licenses/OpenCV.lic
 - (c) 2015 The FreeType Project (www.freetype.org)
		License: <http://git.savannah.gnu.org/cgit/freetype/freetype2.git/plain/docs/FTL.TXT>
	        or doc/licenses/FTL.txt
 - (c) 1997-2002 OpenMP Architecture Review Board (www.openmp.org)
   (c) Microsoft Corporation (Visual C++ implementation for OpenMP C/C++ Version 2.0 March 2002)
		See: <https://msdn.microsoft.com/en-us/library/8y6825x5(v=vs.140).aspx>
 - (c) 1995-2013 zlib software (Jean-loup Gailly and Mark Adler - see: www.zlib.net)
		License: <http://www.zlib.net/zlib_license.html>
            or doc/licenses/zlib.lic
 
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
 If not, see <http://www.gnu.org/licenses/agpl-3.0.txt>.
 ***********************************************************************************************/

#include "matchAssessment.h"
#include "matchAspects.h"
#include "scoreThresholdsBase.h"
#include "misc.h"

using namespace std;
using namespace cv;

extern const double EnableSkipAboveMatchRatio;

MatchAssessor& MatchAssessor::availableAspects(const vector<const std::uniquePtr<const MatchAspect>> &availAspects_) {
	availAspects = &availAspects_;
	return *this;
}

void MatchAssessor::newlyEnabledMatchAspect() {
	enabledAspectsCountM1 = enabledAspectsCount++;
}

void MatchAssessor::newlyDisabledMatchAspect() {
	enabledAspectsCount = enabledAspectsCountM1--;
}

void MatchAssessor::updateEnabledMatchAspectsCount() {
	enabledAspectsCount = 0ULL;
	if(availAspects != nullptr) {
		for(const uniquePtr<const MatchAspect> &pAspect : *availAspects)
			if(pAspect->enabled())
				++enabledAspectsCount;
	}

	enabledAspectsCountM1 = enabledAspectsCount - 1ULL;
}

size_t MatchAssessor::enabledMatchAspectsCount() const {
	return enabledAspectsCount;
}	

void MatchAssessor::getReady(const CachedData &/*cachedData*/) {
	enabledAspects.clear();
	if(availAspects != nullptr) {
		for(const uniquePtr<const MatchAspect> &pAspect : *availAspects)
			if(pAspect->enabled())
				enabledAspects.push_back(&*pAspect);
	}

	assert(enabledAspectsCount == enabledAspects.size());
	assert(enabledAspectsCountM1 == enabledAspectsCount - 1ULL);
}

bool MatchAssessor::isBetterMatch(const Mat &patch, const ISymData &symData, const CachedData &cd,
								  const IScoreThresholds &scoresToBeat,
								  IMatchParamsRW &mp, double &score) const {
	// There is at least one enabled match aspect,
	// since Controller::performTransformation() prevents further calls when there are no enabled aspects.
	assert(enabledAspectsCount > 0ULL && enabledAspectsCount == enabledAspects.size());

	score = enabledAspects[0ULL]->assessMatch(patch, symData, cd, mp);
	for(size_t i = 1ULL; i < enabledAspectsCount; ++i)
		score *= enabledAspects[i]->assessMatch(patch, symData, cd, mp);
	return score > scoresToBeat.overall();
}

MatchAssessorNoSkip::MatchAssessorNoSkip() : MatchAssessor() {}

void MatchAssessorNoSkip::scoresToBeat(double draftScore, IScoreThresholds &scoresToBeat) const {
	scoresToBeat.update(draftScore);
}

MatchAssessorSkip::MatchAssessorSkip() : MatchAssessor() {}

void MatchAssessorSkip::getReady(const CachedData &cachedData) {
	MatchAssessor::getReady(cachedData);

#ifndef AI_REVIEWER_CHECK // AI Reviewer might not parse correctly such lambda-s
	sort(BOUNDS(enabledAspects), [&] (const MatchAspect *a, const MatchAspect *b) -> bool {
		const double relComplexityA = a->relativeComplexity(),
					relComplexityB = b->relativeComplexity();
		// Ascending by complexity
		if(relComplexityA < relComplexityB)
			return true;

		if(relComplexityA > relComplexityB)
			return false;

		// Equal complexity here already

		const double maxScoreA = a->maxScore(cachedData),
					maxScoreB = b->maxScore(cachedData);

		// Descending by max score
		return maxScoreA >= maxScoreB;
	});

#else // AI_REVIEWER_CHECK defined
	// Let AI Reviewer know that following methods were used within the lambda above
	enabledAspects[0ULL]->relativeComplexity();
	enabledAspects[0ULL]->maxScore(cachedData);
#endif // AI_REVIEWER_CHECK

#ifdef MONITOR_SKIPPED_MATCHING_ASPECTS
	totalIsBetterMatchCalls = 0ULL;
	skippedAspects.assign(enabledAspectsCount, 0ULL);
#endif // MONITOR_SKIPPED_MATCHING_ASPECTS

	// Adjust max increase factors for every enabled aspect except last
	invMaxIncreaseFactors.resize(enabledAspectsCountM1);
	double maxIncreaseFactor = 1.;
	for(int ip1 = (int)enabledAspectsCountM1, i = ip1 - 1; i >= 0; ip1 = i--) {
		maxIncreaseFactor *= enabledAspects[(size_t)ip1]->maxScore(cachedData);
		invMaxIncreaseFactors[(size_t)i] = 1. / maxIncreaseFactor;
	}

	thresholdDraftScore =
		maxIncreaseFactor * enabledAspects[0ULL]->maxScore(cachedData) // Max score for the ideal match
		* EnableSkipAboveMatchRatio; // setting the threshold as a % from the ideal score
}

void MatchAssessorSkip::scoresToBeat(double draftScore, IScoreThresholds &scoresToBeat) const {
	if(draftScore < thresholdDraftScore) { // For bad matches intermediary results won't be used
		scoresToBeat.inferiorMatch();
		scoresToBeat.update(draftScore);
	} else {
		scoresToBeat.update(draftScore, invMaxIncreaseFactors);
	}
}

bool MatchAssessorSkip::isBetterMatch(const Mat &patch, const ISymData &symData, const CachedData &cd,
									  const IScoreThresholds &scoresToBeat,
									  IMatchParamsRW &mp, double &score) const {
#ifdef MONITOR_SKIPPED_MATCHING_ASPECTS
#pragma omp atomic
	/*
	We're within a `parallel for` here!!

	However this region is suppressed in the final release, so the speed isn't that relevant.
	It matters only to get correct values for the count of skipped aspects.

	That's why the `parallel for` doesn't need:
	reduction(+ : totalIsBetterMatchCalls)
	*/
	++totalIsBetterMatchCalls;
#endif // MONITOR_SKIPPED_MATCHING_ASPECTS

	// Until finding a good match, Aspects Skipping heuristic won't be used
	if(scoresToBeat.representsInferiorMatch())
		return MatchAssessor::isBetterMatch(patch, symData, cd, scoresToBeat, mp, score);

	// There is at least one enabled match aspect,
	// since Controller::performTransformation() prevents further calls when there are no enabled aspects.
	assert(enabledAspectsCount > 0ULL && enabledAspectsCount == enabledAspects.size());

	score = enabledAspects[0ULL]->assessMatch(patch, symData, cd, mp);

	for(size_t im1 = 0ULL, i = 1ULL; i <= enabledAspectsCountM1; im1 = i++) {
		if(score < scoresToBeat[im1]) {
#ifdef MONITOR_SKIPPED_MATCHING_ASPECTS
			for(size_t j = i; j <= enabledAspectsCountM1; ++j) {
#pragma omp atomic // See comment from previous MONITOR_SKIPPED_MATCHING_ASPECTS region
				++skippedAspects[j];
			}
#endif // MONITOR_SKIPPED_MATCHING_ASPECTS
			return false; // skip further aspects checking when score can't beat best match score
		}
		score *= enabledAspects[i]->assessMatch(patch, symData, cd, mp);
	}

	return score > scoresToBeat.overall();
}
