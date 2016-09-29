/****************************************************************************************
 The application Pic2Sym approximates images by a
 grid of colored symbols with colored backgrounds.

 This file belongs to the Pic2Sym project.

 Copyrights from the libraries used by the program:
 - (c) 2015 Boost (www.boost.org)
   License: <http://www.boost.org/LICENSE_1_0.txt>
            or doc/licenses/Boost.lic
 - (c) 2015 The FreeType Project (www.freetype.org)
   License: <http://git.savannah.gnu.org/cgit/freetype/freetype2.git/plain/docs/FTL.TXT>
	        or doc/licenses/FTL.txt
 - (c) 2015 OpenCV (www.opencv.org)
   License: <http://opencv.org/license.html>
            or doc/licenses/OpenCV.lic
 - (c) 1997-2002 OpenMP Architecture Review Board (www.openmp.org)
   (c) Microsoft Corporation (Visual C++ implementation for OpenMP C/C++ Version 2.0 March 2002)
   See: <https://msdn.microsoft.com/en-us/library/8y6825x5(v=vs.140).aspx>
 
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
 ****************************************************************************************/

#include "matchEngine.h"
#include "clusterEngine.h"
#include "matchAspectsFactory.h"
#include "matchParams.h"
#include "patch.h"
#include "settings.h"
#include "taskMonitor.h"
#include "ompTrace.h"
#include "misc.h"

#include <omp.h>

using namespace std;
using namespace cv;

namespace {
	/// Determines the cluster and the symbol within it corresponding to symIdx
	void locateIdx(const set<unsigned> &clusterOffsets, unsigned symIdx,
				   unsigned &clusterIdx, unsigned &symIdxWithinCluster) {
		auto it = --clusterOffsets.upper_bound(symIdx);
		clusterIdx = (unsigned)distance(clusterOffsets.cbegin(), it);
		symIdxWithinCluster = symIdx - *it;
	}

	/// Checks if the provided symbol range within the current cluster contains a better match
	bool checkRangeWithinCluster(unsigned from, unsigned upperLimit,
								 const MatchEngine &me, const Mat &toApprox,
								 const VSymData &symsSet, const valarray<double> &invMaxIncreaseFactors,
								 BestMatch &draftMatch, MatchParams &mp) {
		bool betterMatchFound = false;
		valarray<double> scoresToBeatBySyms(draftMatch.score * invMaxIncreaseFactors);
		for(unsigned i = from; i < upperLimit; ++i) {
			mp.reset(); // preserves patch-invariant fields
			const auto &symData = symsSet[i];
			double score;
			if(me.isBetterMatch(toApprox, symData, mp, scoresToBeatBySyms, score)) {
				draftMatch.update(score, symData.code, i, symData, mp);
				scoresToBeatBySyms = draftMatch.score * invMaxIncreaseFactors;
				betterMatchFound = true;
			}
		}
		return betterMatchFound;
	}
} // anonymous namespace

MatchEngine::MatchEngine(const Settings &cfg_, FontEngine &fe_) : cfg(cfg_), fe(fe_), ce() {
	for(const auto &aspectName: MatchAspect::aspectNames())
		availAspects.push_back(
			MatchAspectsFactory::create(aspectName, cachedData, cfg_.matchSettings()));
}

void MatchEngine::updateSymbols() {
	const string idForSymsToUse = getIdForSymsToUse(); // throws for invalid cmap/size
	if(symsIdReady.compare(idForSymsToUse) == 0)
		return; // already up to date

	static TaskMonitor fieldsComputations("computing specific symbol-related values", *symsMonitor);

	extern const bool PrepareMoreGlyphsAtOnce;

	symsSet.clear();
	const auto &rawSyms = fe.symsSet();
	const int symsCount = (int)rawSyms.size();
	symsSet.reserve(symsCount);

	fieldsComputations.setTotalSteps((size_t)symsCount);

	const unsigned sz = cfg.symSettings().getFontSz();

#pragma omp parallel if(PrepareMoreGlyphsAtOnce)
#pragma omp for schedule(static, 1) nowait ordered
	for(int i = 0; i<symsCount; ++i) {
		ompPrintf(PrepareMoreGlyphsAtOnce, "glyph %d", i);

		const auto &pms = rawSyms[i];
		const Mat glyph = pms.toMatD01(sz),
				negGlyph = pms.toMat(sz, !pms.removable);
		Mat fgMask, bgMask, edgeMask, groundedGlyph, blurOfGroundedGlyph, varianceOfGroundedGlyph;
		double minVal, maxVal; // for very small fonts, minVal might be > 0 and maxVal might be < 255
		SymData::computeFields(glyph, fgMask, bgMask, edgeMask,
							   groundedGlyph, blurOfGroundedGlyph, varianceOfGroundedGlyph,
							   minVal, maxVal);

#pragma omp ordered
//#pragma omp critical - implied by ordered from above
		symsSet.emplace_back(pms.symCode,
							 minVal, maxVal-minVal,
							 pms.glyphSum, pms.mc,
							 SymData::MatArray { {
										fgMask,					// FG_MASK_IDX
										bgMask,					// BG_MASK_IDX
										edgeMask,				// EDGE_MASK_IDX
										negGlyph,				// NEG_SYM_IDX
										groundedGlyph,			// GROUNDED_SYM_IDX
										blurOfGroundedGlyph,	// BLURRED_GR_SYM_IDX
										varianceOfGroundedGlyph	// VARIANCE_GR_SYM_IDX
									} });

		// #pragma omp master not allowed in for
		if(omp_get_thread_num() == 0)
			fieldsComputations.taskAdvanced((size_t)i);
	}

	fieldsComputations.taskDone();

	// Clustering symsSet (which gets reordered) - clusterOffsets will point where each cluster starts
	ce.process(symsSet);

	symsIdReady = idForSymsToUse; // ready to use the new cmap&size
}

MatchEngine::VSymDataCItPair MatchEngine::getSymsRange(unsigned from, unsigned count) const {
	const unsigned sz = (unsigned)symsSet.size();
	const VSymDataCIt itEnd = symsSet.cend();
	if(from >= sz)
		return make_pair(itEnd, itEnd);

	const VSymDataCIt itStart = next(symsSet.cbegin(), from);
	if(from + count >= sz)
		return make_pair(itStart, itEnd);

	return make_pair(itStart, next(itStart, count));
}

unsigned MatchEngine::getSymsCount() const {
	return (unsigned)symsSet.size();
}

const set<unsigned>& MatchEngine::getClusterOffsets() const {
	return ce.getClusterOffsets();
}

const vector<std::shared_ptr<MatchAspect>>& MatchEngine::availMatchAspects() const {
	return availAspects;
}

void MatchEngine::getReady() {
	updateSymbols();

	cachedData.update(cfg.symSettings().getFontSz(), fe);

	enabledAspects.clear();
	for(auto pAspect : availAspects)
		if(pAspect->enabled())
			enabledAspects.push_back(&*pAspect);

	/*
	Reorder the aspects based on their complexity and their max score.
	The goal is to skip as many complex aspects as possible while checking if a new symbol
	is a better match for a given patch than a previous symbol with a given score.
	As soon as it's obvious that the remaining aspects can't raise the score above current best score,
	the assessment is aborted.
	The strategy is most beneficial if the skipped aspects are more complex than those evaluated already.
	Thus, reorder the enabled aspects as follows:
	- rearrange aspects in increasing order of their complexity
	- for equally complex aspects, consider first those with a higher max score,
	  to reduce the chance that other aspects are needed
	*/
	sort(BOUNDS(enabledAspects), [] (const MatchAspect *a, const MatchAspect *b) -> bool {
		const double relComplexityA = a->relativeComplexity(),
					relComplexityB = b->relativeComplexity();
		// Ascending by complexity
		if(relComplexityA < relComplexityB)
			return true;

		if(relComplexityA > relComplexityB)
			return false;

		// Equal complexity here already

		const double maxScoreA = a->maxScore(),
					maxScoreB = b->maxScore();

		// Descending by max score
		return maxScoreA >= maxScoreB;
	});

	enabledAspectsCount = enabledAspects.size();

#ifdef _DEBUG
	totalIsBetterMatchCalls = 0U;
	skippedAspects.resize(enabledAspectsCount);
	fill(BOUNDS(skippedAspects), 0U);
#endif

	// Adjust max increase factors for every enabled aspect
	invMaxIncreaseFactors.resize(enabledAspectsCount);
	double maxIncreaseFactor = 1.;
	for(int i = (int)enabledAspectsCount - 1; i >= 0; --i)
		invMaxIncreaseFactors[i] = 1. / (maxIncreaseFactor *= enabledAspects[i]->maxScore());
}

bool MatchEngine::findBetterMatch(BestMatch &draftMatch, unsigned fromSymIdx, unsigned upperSymIdx) const {
	assert(!enabledAspects.empty());
	const auto &patch = draftMatch.patch;
	if(!patch.needsApproximation) {
		if(draftMatch.bestVariant.approx.empty()) {
			// update PatchApprox for uniform Patch only during the compare with 1st sym 
			draftMatch.updatePatchApprox(cfg.matchSettings());
			return true;
		}
		return false;
	}

	assert(upperSymIdx <= getSymsCount());

	unsigned fromCluster, firstSymIdxWithinFromCluster, lastCluster, lastSymIdxWithinLastCluster;
	locateIdx(ce.getClusterOffsets(), fromSymIdx, fromCluster, firstSymIdxWithinFromCluster);
	locateIdx(ce.getClusterOffsets(), upperSymIdx-1, lastCluster, lastSymIdxWithinLastCluster);

	const bool previouslyQualified =
		draftMatch.lastSelectedCandidateCluster.is_initialized() &&
		*draftMatch.lastSelectedCandidateCluster == fromCluster;

	// If fromCluster wasn't considered previously worthy enough to investigate,
	// increment fromCluster and reset firstSymIdxWithinFromCluster
	if(firstSymIdxWithinFromCluster > 0U && !previouslyQualified) {
		++fromCluster;
		firstSymIdxWithinFromCluster = 0U;
	}
	
	bool betterMatchFound = false;
	const Mat &toApprox = patch.matrixToApprox();
	MatchParams &mp = draftMatch.bestVariant.params;
	const auto &clusters = ce.getClusters();

	// Multi-element clusters still qualify with slightly inferior scores,
	// as individual symbols within the cluster might deliver a superior score.
	extern const double InvestigateClusterEvenForInferiorScoreFactor;
	valarray<double> scoresToBeatBySyms(draftMatch.score * invMaxIncreaseFactors),
		scoresToBeatByClusters(InvestigateClusterEvenForInferiorScoreFactor * scoresToBeatBySyms);

	for(unsigned clusterIdx = fromCluster; clusterIdx <= lastCluster;
			++clusterIdx, firstSymIdxWithinFromCluster = 0) {
		const auto &cluster = clusters[clusterIdx];

		// 1st cluster might already have been qualified for thorough examination
		if(clusterIdx == fromCluster && previouslyQualified) { // cluster already qualified
			const unsigned upperLimit =
				(clusterIdx < lastCluster) ? cluster.sz : (lastSymIdxWithinLastCluster + 1U);
			if(checkRangeWithinCluster(firstSymIdxWithinFromCluster, upperLimit,
									*this, toApprox, symsSet, invMaxIncreaseFactors,
									draftMatch, mp)) {
				scoresToBeatBySyms = draftMatch.score * invMaxIncreaseFactors;
				scoresToBeatByClusters = InvestigateClusterEvenForInferiorScoreFactor * scoresToBeatBySyms;
				betterMatchFound = true;
			}

			continue;
		}

		// Current cluster attempts qualification - it computes its own score
		mp.reset(); // preserves patch-invariant fields

		double score;
		const bool trivialCluster = (cluster.sz == 1U);
		if(isBetterMatch(toApprox, cluster, mp,
						trivialCluster ? scoresToBeatBySyms : scoresToBeatByClusters, score)) {
			// Single element clusters have same score as their content.
			// So, making sure the score won't be computed twice:
			if(trivialCluster) {
				draftMatch.lastSelectedCandidateCluster = clusterIdx; // cluster is a selected candidate
				const unsigned idx = cluster.idxOfFirstSym;
				const auto &symData = symsSet[idx];
				draftMatch.update(score, symData.code, idx, symData, mp);
				scoresToBeatBySyms = draftMatch.score * invMaxIncreaseFactors;
				scoresToBeatByClusters = InvestigateClusterEvenForInferiorScoreFactor * scoresToBeatBySyms;
				betterMatchFound = true;
				continue;
			}

			// Nontrivial cluster
			draftMatch.lastSelectedCandidateCluster = clusterIdx; // cluster is a selected candidate

			const unsigned upperLimit = (clusterIdx < lastCluster) ? cluster.sz :
											(lastSymIdxWithinLastCluster + 1U);
			if(checkRangeWithinCluster(firstSymIdxWithinFromCluster, upperLimit,
										*this, toApprox, symsSet, invMaxIncreaseFactors,
										draftMatch, mp)) {
				scoresToBeatBySyms = draftMatch.score * invMaxIncreaseFactors;
				scoresToBeatByClusters = InvestigateClusterEvenForInferiorScoreFactor * scoresToBeatBySyms;
				betterMatchFound = true;
			}
		}
	}

	if(betterMatchFound)
		draftMatch.updatePatchApprox(cfg.matchSettings());

	return betterMatchFound;
}

bool MatchEngine::isBetterMatch(const Mat &patch, const SymData &symData, MatchParams &mp,
								const valarray<double> &scoresToBeat, double &score) const {
#ifdef _DEBUG
	++totalIsBetterMatchCalls;
#endif // _DEBUG

	// There is at least one enabled match aspect,
	// since findBetterMatch prevents further calls when there are no enabled aspects.
	assert(enabledAspectsCount > 0U);

	score = enabledAspects[0]->assessMatch(patch, symData, mp);
	unsigned i = 0U, lim = (unsigned)enabledAspectsCount - 1U;
	while(++i <= lim) {
		if(score < scoresToBeat[i]) {
#ifdef _DEBUG
			for(unsigned j = i; j <= lim; ++j)
				++skippedAspects[j];
#endif // _DEBUG
			return false; // skip further aspects checking when score can't beat best match score
		}
		score *= enabledAspects[i]->assessMatch(patch, symData, mp);
	}

	return score > scoresToBeat[lim];
}

MatchEngine& MatchEngine::useSymsMonitor(AbsJobMonitor &symsMonitor_) {
	symsMonitor = &symsMonitor_;
	ce.useSymsMonitor(symsMonitor_);
	return *this;
}

#ifdef _DEBUG

bool MatchEngine::usesUnicode() const {
	return fe.getEncoding().compare("UNICODE") == 0;
}

#else // _DEBUG not defined

bool MatchEngine::usesUnicode() const { return true; }

#endif // _DEBUG
