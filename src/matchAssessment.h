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
 
 (c) 2016, 2017 Florin Tulba <florintulba@yahoo.com>

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

#ifndef H_MATCH_ASSESSMENT
#define H_MATCH_ASSESSMENT

#include "countSkippedAspects.h"

#pragma warning ( push, 0 )

#include "std_memory.h"
#include <vector>

#include <opencv2/core/core.hpp>

#pragma warning ( pop )

// Forward declarations
class CachedData;
struct ISymData;
struct IMatchParamsRW;
struct IScoreThresholds;
class MatchAspect;

/**
Match manager based on the enabled matching aspects:
- selects and rearranges the enabled matching aspects
- applies them while approximating each patch
*/
class MatchAssessor /*abstract*/ {
protected:
	/// The available matching aspects, enabled or not
	const std::vector<const std::uniquePtr<const MatchAspect>> *availAspects = nullptr;

	// matching aspects
	std::vector<const MatchAspect*> enabledAspects;	///< the enabled aspects

	size_t enabledAspectsCount = 0ULL;				///< count of the enabled aspects
	size_t enabledAspectsCountM1 = (size_t)-1LL;	///< count of the enabled aspects minus 1

	MatchAssessor() = default;
	MatchAssessor(const MatchAssessor&) = delete;
	void operator=(const MatchAssessor&) = delete;

public:
	virtual ~MatchAssessor() = 0 {}

	MatchAssessor& availableAspects(const std::vector<const std::uniquePtr<const MatchAspect>> &availAspects_);

	void newlyEnabledMatchAspect();		///< increments enabledAspectsCount
	void newlyDisabledMatchAspect();	///< decrements enabledAspectsCount
	void updateEnabledMatchAspectsCount();		///< updates enabledAspectsCount by checking which aspects are enabled
	size_t enabledMatchAspectsCount() const;	///< provides enabledAspectsCount

	/**
	Prepares the enabled aspects for an image transformation process.

	@param cachedData some precomputed values
	*/
	virtual void getReady(const CachedData &cachedData);

	/// Determines if symData is a better match for patch than previous matching symbol
	virtual bool isBetterMatch(const cv::Mat &patch,	///< the patch whose approximation through a symbol is performed
							   const ISymData &symData,	///< data of the new symbol/cluster compared to the patch
							   const CachedData &cd,	///< precomputed values
							   const IScoreThresholds &scoresToBeat,///< scores after each aspect that beat the current best match
							   IMatchParamsRW &mp,			///< matching parameters resulted from the comparison
							   double &score			///< achieved score of the new assessment
							   ) const;

	/**
	Sets the threshold scores which might spare a symbol from evaluating further matching aspects.

	@param draftScore a new reference score
	@param scoresToBeat the mentioned threshold scores
	*/
	virtual void scoresToBeat(double draftScore, IScoreThresholds &scoresToBeat) const = 0;

#ifdef MONITOR_SKIPPED_MATCHING_ASPECTS
	mutable size_t totalIsBetterMatchCalls = 0ULL; ///< used for reporting skipped aspects
	mutable std::vector<size_t> skippedAspects; ///< used for reporting skipped aspects

	/// While reporting, the particular aspects that were used during the transformation are required
	inline const std::vector<const MatchAspect*>& getEnabledAspects() const { return enabledAspects; }

	/// MatchAssessorSkip will report all skipped matching aspects
	virtual void reportSkippedAspects() const {}
#endif // MONITOR_SKIPPED_MATCHING_ASPECTS
};

/// MatchAssessor version when UseSkipMatchAspectsHeuristic is false
struct MatchAssessorNoSkip : MatchAssessor {
	MatchAssessorNoSkip();
	MatchAssessorNoSkip(const MatchAssessorNoSkip&) = delete;
	void operator=(const MatchAssessorNoSkip&) = delete;

	/**
	@param draftScore a new reference score
	@param scoresToBeat the mentioned threshold scores
	*/
	void scoresToBeat(double draftScore, IScoreThresholds &scoresToBeat) const override;
};

/// MatchAssessor version when UseSkipMatchAspectsHeuristic is true
class MatchAssessorSkip : public MatchAssessor {
protected:
	std::vector<double> invMaxIncreaseFactors; ///< 1 over (max possible increase of the score based on remaining aspects)

	/**
	The Aspects Skipping heuristic comes with a slight cost which becomes noticeable when:
	- there are barely any several skipped matching aspects
	- these very few skipped aspects are not complex

	In such cases the heuristic probably won't shorten the transformation time.

	The heuristic starts getting more efficient only after finding a really good draft match for the patch.
	This justifies introducing a threshold for the score of a draft (the field thresholdDraftScore).
	As long as the draft matches for a given patch score under this threshold, the heuristic isn't used.
	The first draft match above the threshold enables the heuristic.
	*/
	double thresholdDraftScore = 0.;

public:
	MatchAssessorSkip();
	MatchAssessorSkip(const MatchAssessorSkip&) = delete;
	void operator=(const MatchAssessorSkip&) = delete;

	/**
	Prepares the enabled aspects for an image transformation process.

	@param cachedData some precomputed values

	It also rearranges these enabled aspects based on their complexity and their max score.

	The goal is to skip as many complex aspects as possible while checking if a new symbol
	is a better match for a given patch than a previous symbol with a given score.
	As soon as it's obvious that the remaining aspects can't raise the score above current best score,
	the assessment is aborted.

	The strategy is most beneficial if the skipped aspects are more complex than those evaluated already.
	Thus, it reorders the enabled aspects as follows:
	- rearrange aspects in increasing order of their complexity
	- for equally complex aspects, consider first those with a higher max score,
	  to reduce the chance that other aspects are needed
	*/
	void getReady(const CachedData &cachedData) override;

	/**
	Sets the threshold scores which might spare a symbol from evaluating further matching aspects.

	@param draftScore a new reference score
	@param scoresToBeat the mentioned threshold scores
	*/
	void scoresToBeat(double draftScore, IScoreThresholds &scoresToBeat) const override;

	/// Determines if symData is a better match for patch than previous matching symbol
	bool isBetterMatch(const cv::Mat &patch,	///< the patch whose approximation through a symbol is performed
					   const ISymData &symData,	///< data of the new symbol/cluster compared to the patch
					   const CachedData &cd,	///< precomputed values
					   const IScoreThresholds &scoresToBeat,///< scores after each aspect that beat the current best match
					   IMatchParamsRW &mp,			///< matching parameters resulted from the comparison
					   double &score			///< achieved score of the new assessment
					   ) const override;

#ifdef MONITOR_SKIPPED_MATCHING_ASPECTS

	/// Reports all skipped matching aspects
	void reportSkippedAspects() const override;

#endif // MONITOR_SKIPPED_MATCHING_ASPECTS
};

#endif // H_MATCH_ASSESSMENT
