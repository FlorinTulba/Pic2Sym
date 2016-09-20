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
								 const VSymData &symsSet,
								 BestMatch &draftMatch, MatchParams &mp) {
		bool betterMatchFound = false;
		for(unsigned i = from; i < upperLimit; ++i) {
			mp.reset(); // preserves patch-invariant fields
			const auto &symData = symsSet[i];
			double score = me.assessMatch(toApprox, symData, mp);

			if(score > draftMatch.score) {
				draftMatch.update(score, symData.code, i, symData, mp);
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

void MatchEngine::getReady() {
	updateSymbols();

	cachedData.update(cfg.symSettings().getFontSz(), fe);

	enabledAspects.clear();
	for(auto pAspect : availAspects)
		if(pAspect->enabled())
			enabledAspects.push_back(&*pAspect);
}

bool MatchEngine::findBetterMatch(BestMatch &draftMatch, unsigned fromSymIdx, unsigned upperSymIdx) const {
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

	for(unsigned clusterIdx = fromCluster; clusterIdx <= lastCluster;
			++clusterIdx, firstSymIdxWithinFromCluster = 0) {
		const auto &cluster = clusters[clusterIdx];

		// 1st cluster might already have been qualified for thorough examination
		if(clusterIdx == fromCluster && previouslyQualified) { // cluster already qualified
			const unsigned upperLimit =
				(clusterIdx < lastCluster) ? cluster.sz : (lastSymIdxWithinLastCluster + 1U);
			if(checkRangeWithinCluster(firstSymIdxWithinFromCluster, upperLimit,
									*this, toApprox, symsSet,
									draftMatch, mp))
				betterMatchFound = true;
			continue;
		}

		// Current cluster attempts qualification - it computes its own score
		mp.reset(); // preserves patch-invariant fields
		double score = assessMatch(toApprox, cluster, mp);

		// Single element clusters have same score as their content.
		// So, making sure the score won't be computed twice:
		if(cluster.sz == 1U) {
			if(score > draftMatch.score) {
				draftMatch.lastSelectedCandidateCluster = clusterIdx; // cluster is a selected candidate
				const unsigned idx = cluster.idxOfFirstSym;
				const auto &symData = symsSet[idx];
				draftMatch.update(score, symData.code, idx, symData, mp);
				betterMatchFound = true;
			}
			continue;
		}
		
		// Multi-element clusters still qualify with slightly inferior scores,
		// as individual symbols within the cluster might deliver a superior score.
		extern const double InvestigateClusterEvenForInferiorScoreFactor;
		if(score > draftMatch.score * InvestigateClusterEvenForInferiorScoreFactor) {
			draftMatch.lastSelectedCandidateCluster = clusterIdx; // cluster is a selected candidate
			
			const unsigned upperLimit = (clusterIdx < lastCluster) ? cluster.sz :
				(lastSymIdxWithinLastCluster + 1U);
			if(checkRangeWithinCluster(firstSymIdxWithinFromCluster, upperLimit,
									*this, toApprox, symsSet,
									draftMatch, mp))
				betterMatchFound = true;
		}
	}

	if(betterMatchFound)
		draftMatch.updatePatchApprox(cfg.matchSettings());

	return betterMatchFound;
}

double MatchEngine::assessMatch(const Mat &patch,
								const SymData &symData,
								MatchParams &mp) const {
	double score = 1.;
	for(auto pAspect : enabledAspects)
		score *= pAspect->assessMatch(patch, symData, mp);
	// parallelization would require 'mp' fields to be guarded by locks

	return score;
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
