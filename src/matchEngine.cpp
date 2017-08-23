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

#include "matchEngine.h"
#include "matchAssessment.h"
#include "clusterEngine.h"
#include "matchAspectsFactory.h"
#include "matchParamsBase.h"
#include "patchBase.h"
#include "bestMatchBase.h"
#include "preselectManager.h"
#include "preselectSyms.h"
#include "symbolsSupport.h"
#include "clusterSupport.h"
#include "matchSupport.h"
#include "cmapPerspective.h"
#include "settingsBase.h"
#include "symSettingsBase.h"
#include "jobMonitorBase.h"
#include "taskMonitor.h"
#include "ompTrace.h"
#include "misc.h"

#ifndef UNIT_TESTING

#pragma warning ( push, 0 )

// The project uses parallelism
#include <omp.h>

#pragma warning ( pop )

#else // UNIT_TESTING defined
// Unit Tests don't use parallelism, to ensure that at least the sequential code works as expected
extern int __cdecl omp_get_thread_num(void); // returns 0 - the index of the unique thread used

#endif // UNIT_TESTING

using namespace std;
using namespace cv;

extern const bool UseSkipMatchAspectsHeuristic;
extern unsigned TinySymsSz();

namespace {
	/// Returns a configured instance of MatchAssessorNoSkip or MatchAssessorSkip,
	/// depending on UseSkipMatchAspectsHeuristic
	MatchAssessor& specializedInstance(const vector<std::sharedPtr<MatchAspect>> &availAspects_) {
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

	/// Determines the cluster and the symbol within it corresponding to symIdx
	void locateIdx(const set<unsigned> &clusterOffsets, unsigned symIdx,
				   unsigned &clusterIdx, unsigned &symIdxWithinCluster) {
		auto it = --clusterOffsets.upper_bound(symIdx);
		clusterIdx = (unsigned)distance(clusterOffsets.cbegin(), it);
		symIdxWithinCluster = symIdx - *it;
	}

	/// Checks if the provided symbol range within the current cluster contains a better match
	bool checkRangeWithinCluster(unsigned fromIdx, unsigned lastIdx,
								 const MatchEngine &me, const Mat &toApprox,
								 const VSymData &symsSet,
								 const CachedData &cd,
								 ScoreThresholds &scoresToBeatBySyms,
								 IBestMatch &draftMatch) {
		bool betterMatchFound = false;
		const MatchAssessor &assessor = me.assessor();
		const uniquePtr<IMatchParamsRW> &mp = draftMatch.refParams();
		assert(mp);
		for(unsigned idx = fromIdx; idx <= lastIdx; ++idx) {
			mp->reset(); // preserves patch-invariant fields
			const ISymData &symData = *symsSet[(size_t)idx];
			double score;
			if(assessor.isBetterMatch(toApprox, symData, cd, scoresToBeatBySyms, *mp, score)) {
				draftMatch.update(score, symData.getCode(), idx, symData);
				assessor.scoresToBeat(score, scoresToBeatBySyms);
				betterMatchFound = true;
			}
		}
		return betterMatchFound;
	}
} // anonymous namespace

MatchEngine::MatchEngine(const ISettings &cfg_, FontEngine &fe_, CmapPerspective &cmP_) :
			cfg(cfg_), fe(fe_), cmP(cmP_),
			matchAssessor(specializedInstance(availAspects)),
			ce(makeUnique<ClusterEngine>(fe_, symsSet)),
			matchSupport(IPreselManager::concrete().createMatchSupport(
				cachedData, symsSet, matchAssessor, cfg_.getMS())) {
	std::sharedPtr<MatchAspect> aspect;
	for(const stringType &aspectName: MatchAspect::aspectNames())
		availAspects.push_back(MatchAspectsFactory::create(aspectName, cfg_.getMS()));

	matchAssessor.updateEnabledMatchAspectsCount();
}

void MatchEngine::updateSymbols() {
	const stringType idForSymsToUse = getIdForSymsToUse(); // throws for invalid cmap/size
	if(symsIdReady.compare(idForSymsToUse) == 0)
		return; // already up to date

#pragma warning ( disable : WARN_THREAD_UNSAFE )
	static TaskMonitor fieldsComputations("computing specific symbol-related values", *symsMonitor);
#pragma warning ( default : WARN_THREAD_UNSAFE )

	extern const bool PrepareMoreGlyphsAtOnce;

	symsSet.clear();
	const VPixMapSym &rawSyms = fe.symsSet();
	const int symsCount = (int)rawSyms.size();
	symsSet.reserve((size_t)symsCount);

	fieldsComputations.setTotalSteps((size_t)symsCount);

	const unsigned sz = cfg.getSS().getFontSz();

#pragma omp parallel if(PrepareMoreGlyphsAtOnce)
#pragma omp for schedule(static, 1) nowait // ordered would be useful only for debugging (ompPrintf)
	for(int i = 0; i<symsCount; ++i) {
		ompPrintf(PrepareMoreGlyphsAtOnce, "glyph %d", i);

		// Computing SymData fields separately, to keep the critical emplace from below as short as possible
		uniquePtr<const ISymData> newSym = makeUnique<const SymData>(*rawSyms[(size_t)i], sz, false);

#pragma omp critical // ordered instead of critical would be useful only for debugging
		symsSet.emplace_back(move(newSym));

		// #pragma omp master not allowed in for
		if(omp_get_thread_num() == 0)
			fieldsComputations.taskAdvanced((size_t)i);
	}

	fieldsComputations.taskDone();

	ce->support().groupSyms(fe.getFontType());
	cmP.reset(symsSet, ce->getSymsIndicesPerCluster());

	symsIdReady = idForSymsToUse; // ready to use the new cmap&size
}

unsigned MatchEngine::getSymsCount() const {
	return (unsigned)symsSet.size();
}

const vector<std::sharedPtr<MatchAspect>>& MatchEngine::availMatchAspects() const {
	return availAspects;
}

const MatchAssessor& MatchEngine::assessor() const { 
	return matchAssessor;
}

IMatchSupport& MatchEngine::support() {
	assert(matchSupport);
	return *matchSupport;
}

void MatchEngine::getReady() {
	updateSymbols();

	matchSupport->updateCachedData(cfg.getSS().getFontSz(), fe);
	matchAssessor.getReady(cachedData);
}

const bool& MatchEngine::isClusteringUseful() const {
	return ce->worthGrouping();
}

bool MatchEngine::improvesBasedOnBatch(unsigned fromSymIdx, unsigned upperSymIdx,
									   IBestMatch &draftMatch, MatchProgress &matchProgress) const {
	assert(matchAssessor.enabledMatchAspectsCount() > 0ULL);
	assert(draftMatch.getPatch().nonUniform());
	assert(upperSymIdx <= getSymsCount());

	const CachedData &cd = matchSupport->cachedData();
	const VSymData &inspectedSet = ce->support().clusteredSyms();

	const Mat &toApprox = draftMatch.getPatch().matrixToApprox();
	const uniquePtr<IMatchParamsRW> &mp = draftMatch.refParams();
	assert(mp);
	ScoreThresholds scoresToBeatBySyms;
	matchAssessor.scoresToBeat(draftMatch.getScore(), scoresToBeatBySyms);

	double score;
	bool betterMatchFound = false;
	if(ce->worthGrouping()) { // Matching is performed first with clusters and only afterwards with individual symbols
		unsigned fromCluster, firstSymIdxWithinFromCluster, lastCluster, lastSymIdxWithinLastCluster;
		locateIdx(ce->getClusterOffsets(), fromSymIdx, fromCluster, firstSymIdxWithinFromCluster);
		locateIdx(ce->getClusterOffsets(), upperSymIdx-1, lastCluster, lastSymIdxWithinLastCluster);

		const VClusterData &clusters = ce->getClusters();
		const bool previouslyQualified = (clusters[(size_t)fromCluster]->getSz() > 1U) &&
						(draftMatch.getLastPromisingNontrivialCluster() == fromCluster);

		// Multi-element clusters still qualify with slightly inferior scores,
		// as individual symbols within the cluster might deliver a superior score.
		extern const double InvestigateClusterEvenForInferiorScoreFactor;
		ScoreThresholds scoresToBeatByClusters(InvestigateClusterEvenForInferiorScoreFactor,
											   scoresToBeatBySyms);

		// 1st cluster might have already been qualified for thorough examination
		if(previouslyQualified) { // cluster already qualified
			const unsigned upperLimit = (fromCluster < lastCluster) ?
				clusters[(size_t)fromCluster]->getSz() : lastSymIdxWithinLastCluster;
			if(checkRangeWithinCluster(firstSymIdxWithinFromCluster, upperLimit, *this, toApprox, 
									inspectedSet, cd, scoresToBeatBySyms, draftMatch)) {
				scoresToBeatByClusters.update(InvestigateClusterEvenForInferiorScoreFactor,
											  scoresToBeatBySyms);
				matchProgress.remarkedMatch(*draftMatch.getSymIdx(), draftMatch.getScore());

				betterMatchFound = true;
			}

			++fromCluster; // nothing else to investigate from this cluster

		} else if(firstSymIdxWithinFromCluster > 0U) {
			// If cluster fromCluster was already analyzed, but wasn't considered worthy enough
			// to investigate symbol by symbol, increment fromCluster
			++fromCluster;
		}

		// Examine all remaining unchecked clusters (if any) within current batch
		for(unsigned clusterIdx = fromCluster; clusterIdx <= lastCluster; ++clusterIdx) {
			const uniquePtr<const IClusterData> &cluster = clusters[(size_t)clusterIdx];

			// Current cluster attempts qualification - it computes its own score
			mp->reset(); // preserves patch-invariant fields

			if(cluster->getSz() == 1U) { // Trivial cluster
				// Single element clusters have same score as their content.
				const unsigned symIdx = cluster->getIdxOfFirstSym();
				const ISymData &symData = *inspectedSet[(size_t)symIdx];
				if(matchAssessor.isBetterMatch(toApprox, symData, cd, scoresToBeatBySyms, *mp, score)) {
					draftMatch.update(score, symData.getCode(), symIdx, symData);
					matchAssessor.scoresToBeat(score, scoresToBeatBySyms);
					scoresToBeatByClusters.update(InvestigateClusterEvenForInferiorScoreFactor,
												  scoresToBeatBySyms);
					matchProgress.remarkedMatch(symIdx, score);

					betterMatchFound = true;
				}
			
			} else { // Nontrivial cluster
				if(matchAssessor.isBetterMatch(toApprox, *cluster, cd,
											scoresToBeatByClusters, *mp, score)) {
					draftMatch.setLastPromisingNontrivialCluster(clusterIdx); // cluster is a selected candidate

					const unsigned upperLimit = (clusterIdx < lastCluster) ?
						cluster->getSz() : lastSymIdxWithinLastCluster;
					if(checkRangeWithinCluster(0U, upperLimit, *this, toApprox, inspectedSet, cd,
												scoresToBeatBySyms, draftMatch)) {
						scoresToBeatByClusters.update(InvestigateClusterEvenForInferiorScoreFactor,
													  scoresToBeatBySyms);
						matchProgress.remarkedMatch(*draftMatch.getSymIdx(), draftMatch.getScore());

						betterMatchFound = true;
					}
				}
			}
		}

	} else { // Matching is performed directly with individual symbols, not with clusters
		// Examine all remaining symbols within current batch
		for(unsigned symIdx = fromSymIdx; symIdx < upperSymIdx; ++symIdx) {
			const ISymData &symData = *inspectedSet[(size_t)symIdx];

			mp->reset(); // preserves patch-invariant fields

			if(matchAssessor.isBetterMatch(toApprox, symData, cd, scoresToBeatBySyms, *mp, score)) {
				draftMatch.update(score, symData.getCode(), symIdx, symData);
				matchAssessor.scoresToBeat(score, scoresToBeatBySyms);
				matchProgress.remarkedMatch(symIdx, score);

				betterMatchFound = true;
			}
		}
	}

	if(betterMatchFound)
		draftMatch.updatePatchApprox(cfg.getMS());

	return betterMatchFound;
}

MatchEngine& MatchEngine::useSymsMonitor(AbsJobMonitor &symsMonitor_) {
	symsMonitor = &symsMonitor_;
	ce->useSymsMonitor(symsMonitor_);
	return *this;
}

#ifdef _DEBUG

bool MatchEngine::usesUnicode() const {
	return fe.getEncoding().compare("UNICODE") == 0;
}

#else // _DEBUG not defined

bool MatchEngine::usesUnicode() const { return true; }

#endif // _DEBUG
