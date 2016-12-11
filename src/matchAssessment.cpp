/************************************************************************************************
 The application Pic2Sym approximates images by a
 grid of colored symbols with colored backgrounds.

 This file belongs to the Pic2Sym project.

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
 
 (c) 2016 Florin Tulba <florintulba@yahoo.com>

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
#include "misc.h"

using namespace std;
using namespace cv;

extern const bool UseSkipMatchAspectsHeuristic;
extern const double EnableSkipAboveMatchRatio;

MatchAssessor& MatchAssessor::availableAspects(const vector<std::shared_ptr<MatchAspect>> &availAspects_) {
	availAspects = &availAspects_;
	return *this;
}

MatchAssessor& MatchAssessor::specializedInstance(const vector<std::shared_ptr<MatchAspect>> &availAspects_) {
	if(UseSkipMatchAspectsHeuristic) {
#pragma warning ( disable : WARN_THREAD_UNSAFE )
		static MatchAssessorSkip instSkip;
#pragma warning ( default : WARN_THREAD_UNSAFE )

		return instSkip.availableAspects(availAspects_);
	}

#pragma warning ( disable : WARN_THREAD_UNSAFE )
	static MatchAssessorNoSkip instNoSkip;
#pragma warning ( default : WARN_THREAD_UNSAFE )

	return instNoSkip.availableAspects(availAspects_);
}

MatchAssessor::MatchAssessor() {}

void MatchAssessor::newlyEnabledMatchAspect() {
	enabledAspectsCountM1 = enabledAspectsCount++;
}

void MatchAssessor::newlyDisabledMatchAspect() {
	enabledAspectsCount = enabledAspectsCountM1--;
}

void MatchAssessor::updateEnabledMatchAspectsCount() {
	enabledAspectsCount = 0ULL;
	if(availAspects != nullptr) {
		for(auto pAspect : *availAspects)
			if(pAspect->enabled())
				++enabledAspectsCount;
	}

	enabledAspectsCountM1 = enabledAspectsCount - 1ULL;
}

size_t MatchAssessor::enabledMatchAspectsCount() const {
	return enabledAspectsCount;
}	

void MatchAssessor::getReady(const CachedData &cachedData) {
	enabledAspects.clear();
	if(availAspects != nullptr) {
		for(auto pAspect : *availAspects)
			if(pAspect->enabled())
				enabledAspects.push_back(&*pAspect);
	}

	assert(enabledAspectsCount == enabledAspects.size());
	assert(enabledAspectsCountM1 == enabledAspectsCount - 1ULL);
}

bool MatchAssessor::isBetterMatch(const Mat &patch, const SymData &symData, const CachedData &cd,
								  const valarray<double> &scoresToBeat,
								  MatchParams &mp, double &score) const {
	// There is at least one enabled match aspect,
	// since Controller::performTransformation() prevents further calls when there are no enabled aspects.
	assert(enabledAspectsCount > 0ULL && enabledAspectsCount == enabledAspects.size());

	score = enabledAspects[0ULL]->assessMatch(patch, symData, cd, mp);
	for(size_t i = 1ULL; i < enabledAspectsCount; ++i)
		score *= enabledAspects[i]->assessMatch(patch, symData, cd, mp);
	return score > scoresToBeat[0ULL];
}

MatchAssessorNoSkip::MatchAssessorNoSkip() : MatchAssessor() {}

const valarray<double> MatchAssessorNoSkip::scoresToBeat(double draftScore) const {
	return std::move(valarray<double>(draftScore, 1ULL));
}

MatchAssessorSkip::MatchAssessorSkip() : MatchAssessor() {}

void MatchAssessorSkip::getReady(const CachedData &cachedData) {
	MatchAssessor::getReady(cachedData);

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

#ifdef MONITOR_SKIPPED_MATCHING_ASPECTS
	totalIsBetterMatchCalls = 0ULL;
	skippedAspects.assign(enabledAspectsCount, 0ULL);
#endif // MONITOR_SKIPPED_MATCHING_ASPECTS

	// Adjust max increase factors for every enabled aspect
	invMaxIncreaseFactors.resize(enabledAspectsCount);
	const int lastIdx = (int)enabledAspectsCountM1;
	double maxIncreaseFactor = invMaxIncreaseFactors[enabledAspectsCountM1] = 1.;
	for(int i = lastIdx - 1, ip1 = lastIdx; i >= 0; ip1 = i--) {
		maxIncreaseFactor *= enabledAspects[(size_t)ip1]->maxScore(cachedData);
		invMaxIncreaseFactors[(size_t)i] = 1. / maxIncreaseFactor;
	}

	thresholdDraftScore =
		maxIncreaseFactor * enabledAspects[0ULL]->maxScore(cachedData) // Max score for the ideal match
		* EnableSkipAboveMatchRatio; // setting the threshold as a % from the ideal score
}

const valarray<double> MatchAssessorSkip::scoresToBeat(double draftScore) const {	
	if(draftScore < thresholdDraftScore) // Returning a trivial valarray for bad matches
		return std::move(valarray<double>(draftScore, 1ULL));

	return std::move(draftScore * invMaxIncreaseFactors);
}

bool MatchAssessorSkip::isBetterMatch(const Mat &patch, const SymData &symData, const CachedData &cd,
									  const valarray<double> &scoresToBeat,
									  MatchParams &mp, double &score) const {
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

	// Until finding a good match, scoresToBeat has just 1 element and Aspects Skipping heuristic won't be used
	if(scoresToBeat.size() == 1ULL)
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

	return score > scoresToBeat[enabledAspectsCountM1];
}
