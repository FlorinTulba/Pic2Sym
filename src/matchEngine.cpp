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

#include "matchEngine.h"
#include "matchAssessment.h"
#include "clusterEngine.h"
#include "matchAspectsFactory.h"
#include "matchParams.h"
#include "patch.h"
#include "preselectManager.h"
#include "preselectSyms.h"
#include "clusterSupport.h"
#include "matchSupport.h"
#include "cmapPerspective.h"
#include "settings.h"
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

extern unsigned TinySymsSz();

namespace {
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
								 BestMatch &draftMatch, MatchParams &mp) {
		bool betterMatchFound = false;
		const auto &assessor = me.assessor();
		for(unsigned idx = fromIdx; idx <= lastIdx; ++idx) {
			mp.reset(); // preserves patch-invariant fields
			const auto &symData = symsSet[idx];
			double score;
			if(assessor.isBetterMatch(toApprox, symData, cd, scoresToBeatBySyms, mp, score)) {
				draftMatch.update(score, symData.code, idx, symData, mp);
				assessor.scoresToBeat(score, scoresToBeatBySyms);
				betterMatchFound = true;
			}
		}
		return betterMatchFound;
	}
} // anonymous namespace

MatchEngine::MatchEngine(const Settings &cfg_, FontEngine &fe_, CmapPerspective &cmP_) :
			cfg(cfg_), fe(fe_), cmP(cmP_), ce(fe_),
			matchAssessor(MatchAssessor::specializedInstance(availAspects)) {
	std::shared_ptr<MatchAspect> aspect;
	for(const auto &aspectName: MatchAspect::aspectNames())
		availAspects.push_back(MatchAspectsFactory::create(aspectName, cfg_.matchSettings()));

	matchAssessor.updateEnabledMatchAspectsCount();
}

void MatchEngine::updateSymbols() {
	const string idForSymsToUse = getIdForSymsToUse(); // throws for invalid cmap/size
	if(symsIdReady.compare(idForSymsToUse) == 0)
		return; // already up to date

#pragma warning ( disable : WARN_THREAD_UNSAFE )
	static TaskMonitor fieldsComputations("computing specific symbol-related values", *symsMonitor);
#pragma warning ( default : WARN_THREAD_UNSAFE )

	extern const bool PrepareMoreGlyphsAtOnce;

	symsSet.clear();
	const auto &rawSyms = fe.symsSet();
	const int symsCount = (int)rawSyms.size();
	symsSet.reserve((size_t)symsCount);

	fieldsComputations.setTotalSteps((size_t)symsCount);

	const unsigned sz = cfg.symSettings().getFontSz();

#pragma omp parallel if(PrepareMoreGlyphsAtOnce)
#pragma omp for schedule(static, 1) nowait // ordered would be useful only for debugging (ompPrintf)
	for(int i = 0; i<symsCount; ++i) {
		ompPrintf(PrepareMoreGlyphsAtOnce, "glyph %d", i);

		const auto &pms = rawSyms[(size_t)i];
		const Mat glyph = pms.toMatD01(sz),
				negGlyph = pms.toMat(sz, true);
		Mat fgMask, bgMask, edgeMask, groundedGlyph, blurOfGroundedGlyph, varianceOfGroundedGlyph;
		double minVal, diffMinMax; // for very small fonts, minVal might be > 0 and diffMinMax might be < 255

		// Computing SymData fields separately, to keep the critical emplace from below as short as possible
		SymData::computeFields(glyph, fgMask, bgMask, edgeMask,
							   groundedGlyph, blurOfGroundedGlyph, varianceOfGroundedGlyph,
							   minVal, diffMinMax, false);

#pragma omp critical // ordered instead of critical would be useful only for debugging
		symsSet.emplace_back(negGlyph,
							 pms.symCode,
							 pms.symIdx,
							 minVal, diffMinMax,
							 pms.avgPixVal, pms.mc,
							 SymData::MatArray { {
										fgMask,					// FG_MASK_IDX
										bgMask,					// BG_MASK_IDX
										edgeMask,				// EDGE_MASK_IDX
										groundedGlyph,			// GROUNDED_SYM_IDX
										blurOfGroundedGlyph,	// BLURRED_GR_SYM_IDX
										varianceOfGroundedGlyph	// VARIANCE_GR_SYM_IDX
								} },
								pms.removable);

		// #pragma omp master not allowed in for
		if(omp_get_thread_num() == 0)
			fieldsComputations.taskAdvanced((size_t)i);
	}

	fieldsComputations.taskDone();

	if(preselManager == nullptr)
		THROW_WITH_CONST_MSG("Please call 'usePreselManager()' before using " __FUNCTION__, logic_error);

	preselManager->clustersSupport().groupSyms(fe.getFontType());
	cmP.reset(symsSet, ce.getSymsIndicesPerCluster());

	symsIdReady = idForSymsToUse; // ready to use the new cmap&size
}

unsigned MatchEngine::getSymsCount() const {
	return (unsigned)symsSet.size();
}

const vector<std::shared_ptr<MatchAspect>>& MatchEngine::availMatchAspects() const {
	return availAspects;
}

const MatchAssessor& MatchEngine::assessor() const { 
	return matchAssessor;
}

void MatchEngine::getReady() {
	updateSymbols();

	if(preselManager!=nullptr)
		preselManager->matchSupport().updateCachedData(cfg.symSettings().getFontSz(), fe);
	else
		THROW_WITH_CONST_MSG("Please call 'usePreselManager()' before using " __FUNCTION__, logic_error);

	matchAssessor.getReady(cachedData);
}

bool MatchEngine::improvesBasedOnBatch(unsigned fromSymIdx, unsigned upperSymIdx,
									   BestMatch &draftMatch, MatchProgress &matchProgress) const {
	assert(matchAssessor.enabledMatchAspectsCount() > 0ULL);
	assert(draftMatch.patch.needsApproximation);
	assert(upperSymIdx <= getSymsCount());
	assert(preselManager != nullptr);

	auto &matchSupport = preselManager->matchSupport();
	const CachedData &cd = matchSupport.cachedData();
	const VSymData &inspectedSet = preselManager->clustersSupport().clusteredSyms();

	const Mat &toApprox = draftMatch.patch.matrixToApprox();
	MatchParams &mp = draftMatch.bestVariant.params;
	ScoreThresholds scoresToBeatBySyms;
	matchAssessor.scoresToBeat(draftMatch.score, scoresToBeatBySyms);

	double score;
	bool betterMatchFound = false;
	if(ce.worthGrouping()) { // Matching is performed first with clusters and only afterwards with individual symbols
		unsigned fromCluster, firstSymIdxWithinFromCluster, lastCluster, lastSymIdxWithinLastCluster;
		locateIdx(ce.getClusterOffsets(), fromSymIdx, fromCluster, firstSymIdxWithinFromCluster);
		locateIdx(ce.getClusterOffsets(), upperSymIdx-1, lastCluster, lastSymIdxWithinLastCluster);

		const auto &clusters = ce.getClusters();
		const bool previouslyQualified = (clusters[fromCluster].sz > 1U) &&
						draftMatch.lastPromisingNontrivialCluster.is_initialized() &&
						(*draftMatch.lastPromisingNontrivialCluster == fromCluster);

		// Multi-element clusters still qualify with slightly inferior scores,
		// as individual symbols within the cluster might deliver a superior score.
		extern const double InvestigateClusterEvenForInferiorScoreFactor;
		ScoreThresholds scoresToBeatByClusters(InvestigateClusterEvenForInferiorScoreFactor, scoresToBeatBySyms);

		// 1st cluster might have already been qualified for thorough examination
		if(previouslyQualified) { // cluster already qualified
			const unsigned upperLimit = (fromCluster < lastCluster) ?
											clusters[fromCluster].sz : lastSymIdxWithinLastCluster;
			if(checkRangeWithinCluster(firstSymIdxWithinFromCluster, upperLimit,
									*this, toApprox, inspectedSet, cd,
									scoresToBeatBySyms,
									draftMatch, mp)) {
				scoresToBeatByClusters.update(InvestigateClusterEvenForInferiorScoreFactor, scoresToBeatBySyms);
				matchProgress.remarkedMatch(*draftMatch.symIdx, draftMatch.score);

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
			const auto &cluster = clusters[clusterIdx];

			// Current cluster attempts qualification - it computes its own score
			mp.reset(); // preserves patch-invariant fields

			if(cluster.sz == 1U) { // Trivial cluster
				// Single element clusters have same score as their content.
				const unsigned symIdx = cluster.idxOfFirstSym;
				const auto &symData = inspectedSet[symIdx];
				if(matchAssessor.isBetterMatch(toApprox, symData, cd, scoresToBeatBySyms, mp, score)) {
					draftMatch.update(score, symData.code, symIdx, symData, mp);
					matchAssessor.scoresToBeat(score, scoresToBeatBySyms);
					scoresToBeatByClusters.update(InvestigateClusterEvenForInferiorScoreFactor, scoresToBeatBySyms);
					matchProgress.remarkedMatch(symIdx, score);

					betterMatchFound = true;
				}
			
			} else { // Nontrivial cluster
				if(matchAssessor.isBetterMatch(toApprox, cluster, cd, scoresToBeatByClusters, mp, score)) {
					draftMatch.lastPromisingNontrivialCluster = clusterIdx; // cluster is a selected candidate

					const unsigned upperLimit = (clusterIdx < lastCluster) ?
						cluster.sz : lastSymIdxWithinLastCluster;
					if(checkRangeWithinCluster(0U, upperLimit, 
												*this, toApprox, inspectedSet, cd,
												scoresToBeatBySyms,
												draftMatch, mp)) {
						scoresToBeatByClusters.update(InvestigateClusterEvenForInferiorScoreFactor, scoresToBeatBySyms);
						matchProgress.remarkedMatch(*draftMatch.symIdx, draftMatch.score);

						betterMatchFound = true;
					}
				}
			}
		}

	} else { // Matching is performed directly with individual symbols, not with clusters
		// Examine all remaining symbols within current batch
		for(unsigned symIdx = fromSymIdx; symIdx < upperSymIdx; ++symIdx) {
			const auto &symData = inspectedSet[symIdx];

			mp.reset(); // preserves patch-invariant fields

			if(matchAssessor.isBetterMatch(toApprox, symData, cd, scoresToBeatBySyms, mp, score)) {
				draftMatch.update(score, symData.code, symIdx, symData, mp);
				matchAssessor.scoresToBeat(score, scoresToBeatBySyms);
				matchProgress.remarkedMatch(symIdx, score);

				betterMatchFound = true;
			}
		}
	}

	if(betterMatchFound)
		draftMatch.updatePatchApprox(cfg.matchSettings());

	return betterMatchFound;
}

MatchEngine& MatchEngine::useSymsMonitor(AbsJobMonitor &symsMonitor_) {
	symsMonitor = &symsMonitor_;
	ce.useSymsMonitor(symsMonitor_);
	return *this;
}

MatchEngine& MatchEngine::usePreselManager(PreselManager &preselManager_) {
	preselManager = &preselManager_;
	return *this;
}

#ifdef _DEBUG

bool MatchEngine::usesUnicode() const {
	return fe.getEncoding().compare("UNICODE") == 0;
}

#else // _DEBUG not defined

bool MatchEngine::usesUnicode() const { return true; }

#endif // _DEBUG
