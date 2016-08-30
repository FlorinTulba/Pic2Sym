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
#include "matchAspectsFactory.h"
#include "matchParams.h"
#include "patch.h"
#include "settings.h"
#include "misc.h"
#include "ompTrace.h"

#include <numeric>

#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

extern const Size BlurWinSize;
extern const double BlurStandardDeviation;
extern const double MaxAvgProjErrForClustering;
extern const double MaxRelMcOffsetForClustering;
extern const double MaxDiffAvgPixelValForClustering;

// constants for foreground / background thresholds
// 1/255 = 0.00392, so 0.004 tolerates pixels with 1 brightness unit less / more than ideal
// STILL_BG was set to 0, as there are font families with extremely similar glyphs.
// When Unit Testing shouldn't identify exactly each glyph, STILL_BG might be > 0.
// But testing on 'BPmonoBold.ttf' does tolerate such larger values (0.025, for instance).
extern const double MatchEngine_updateSymbols_STILL_BG;					// darkest shades

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

	/// Computes most information about a symbol based on glyph and negGlyph parameters
	void computeSymData(const Mat &glyph, const Mat &negGlyph,
						Mat &fgMask, Mat &bgMask, Mat &edgeMask,
						Mat &groundedGlyph, Mat &blurOfGroundedGlyph, Mat &varianceOfGroundedGlyph,
						double &minVal, double &maxVal) {
		static const double STILL_FG = 1. - MatchEngine_updateSymbols_STILL_BG;	// brightest shades

		minMaxIdx(glyph, &minVal, &maxVal);
		groundedGlyph = (minVal==0. ? glyph : (glyph - minVal)); // min val on 0

		fgMask = (glyph >= (minVal + STILL_FG * (maxVal-minVal)));
		bgMask = (glyph <= (minVal + MatchEngine_updateSymbols_STILL_BG * (maxVal-minVal)));

		// Storing a blurred version of the grounded glyph for structural similarity match aspect
		GaussianBlur(groundedGlyph, blurOfGroundedGlyph,
					 BlurWinSize, BlurStandardDeviation, 0.,
					 BORDER_REPLICATE);

		// edgeMask selects all pixels that are not minVal, nor maxVal
		inRange(glyph, minVal+EPS, maxVal-EPS, edgeMask);

		// Storing also the variance of the grounded glyph for structural similarity match aspect
		// Actual varianceOfGroundedGlyph is obtained in the subtraction after the blur
		GaussianBlur(groundedGlyph.mul(groundedGlyph), varianceOfGroundedGlyph,
					 BlurWinSize, BlurStandardDeviation, 0.,
					 BORDER_REPLICATE);

		varianceOfGroundedGlyph -= blurOfGroundedGlyph.mul(blurOfGroundedGlyph);
	}

	/// Clusters symsSet into clusters, while clusterOffsets reports where each cluster starts
	void clusterSyms(VSymData &symsSet,
					 VClusterData &clusters, set<unsigned> &clusterOffsets) {
		static const int SmallSymSzI = 5, SmallSymSzAreaI = SmallSymSzI * SmallSymSzI;
		static const double SmallSymSzD = (double)SmallSymSzI, SmallSymAreaD = (double)SmallSymSzAreaI;
		static const Size SmallSymDim(SmallSymSzI, SmallSymSzI);
		static const unsigned DiagsCountU = 2U * SmallSymSzI - 1U;
		static const double DiagsCountD = (double)DiagsCountU;

		const unsigned symsCount = (unsigned)symsSet.size();

		/// Data used to decide if 2 symbols can be grouped together
		struct InfoForClustering {
			const Point2d mc;	///< original mc divided by font size (0..1 x 0..1 range)
			const double avgPixVal;	///< original pixelSum divided by font area (0..1 range)
			const Mat mat;		///< resized glyph to 5x5
			const Mat hAvgProj, vAvgProj; // horizontal and vertical projection
			const Mat backslashDiagAvgProj, slashDiagAvgProj; // normal and inverse diagonal projections

			InfoForClustering(const Point2d &mc_, double avgPixVal_, const Mat mat_,
							  const Mat hAvgProj_, const Mat vAvgProj_,
							  const Mat backslashDiagAvgProj_, const Mat slashDiagAvgProj_) :
				mc(mc_), avgPixVal(avgPixVal_), mat(mat_), hAvgProj(hAvgProj_), vAvgProj(vAvgProj_),
				backslashDiagAvgProj(backslashDiagAvgProj_), slashDiagAvgProj(slashDiagAvgProj_) {}
		};
		vector<InfoForClustering> smallSyms;
#pragma region initSmallSyms
		smallSyms.reserve(symsCount);
		const double initFontSz = (double)symsSet[0].symAndMasks[SymData::NEG_SYM_IDX].rows,
					initFontArea = initFontSz * initFontSz;
		for(const auto &symData : symsSet) {
			const Mat &gsi = symData.symAndMasks[SymData::GROUNDED_SYM_IDX]; // double values, 0..1 range
			Mat smallSym, flippedSmallSym,
				hAvgProj, vAvgProj,
				backslashDiagAvgProj(1, DiagsCountU, CV_64FC1, 0.),
				slashDiagAvgProj(1, DiagsCountU, CV_64FC1, 0.);
			resize(gsi, smallSym, SmallSymDim, 0., 0., CV_INTER_AREA);

			// computing average projections
			reduce(smallSym, hAvgProj, 0, CV_REDUCE_AVG);
			reduce(smallSym, vAvgProj, 1, CV_REDUCE_AVG);
			flip(smallSym, flippedSmallSym, 1); // flip around vertical axis
			for(int diagIdx = -SmallSymSzI+1, i = 0; diagIdx < SmallSymSzI; ++diagIdx, ++i) {
				const Mat backslashDiag = smallSym.diag(diagIdx);
				backslashDiagAvgProj.at<double>(i) = *mean(backslashDiag).val;

				const Mat slashDiag = flippedSmallSym.diag(-diagIdx);
				slashDiagAvgProj.at<double>(i) = *mean(slashDiag).val;
			}

			smallSym /= SmallSymSzD;
			hAvgProj /= SmallSymSzD;
			vAvgProj /= SmallSymSzD;
			backslashDiagAvgProj /= DiagsCountD;
			slashDiagAvgProj /= DiagsCountD;

			smallSyms.emplace_back(symData.mc/initFontSz, symData.pixelSum/initFontArea, smallSym,
								   hAvgProj, vAvgProj, backslashDiagAvgProj, slashDiagAvgProj);
		}
#pragma endregion initSmallSyms

		// cluster smallSyms
#if defined _DEBUG && !defined UNIT_TESTING
		unsigned countAvgPixDiff = 0U, countMcsOffset = 0U, countDiff = 0U, countHdiff = 0U, countVdiff = 0U, countBslashDiff = 0U, countSlashDiff = 0U;
#endif
		static const double SqMaxRelMcOffsetForClustering = MaxRelMcOffsetForClustering * MaxRelMcOffsetForClustering;
		const double TotMaxProjErrForClustering = MaxAvgProjErrForClustering * SmallSymSzD;

		vector<int> clusterLabels(symsCount, -1);
		const unsigned clustersCount = (unsigned)partition(smallSyms, clusterLabels, [&] (
														   const InfoForClustering &a,
														   const InfoForClustering &b) {
#if !defined _DEBUG || defined UNIT_TESTING
			if(abs(a.avgPixVal - b.avgPixVal) > MaxDiffAvgPixelValForClustering)
				return false;
			const Point2d mcDelta = a.mc - b.mc;
			const double mcDeltaY = abs(mcDelta.y);
			if(mcDeltaY > MaxRelMcOffsetForClustering)
				return false;
			const double mcDeltaX = abs(mcDelta.x);
			if(mcDeltaX > MaxRelMcOffsetForClustering)
				return false;
			if(mcDeltaX*mcDeltaX + mcDeltaY*mcDeltaY > SqMaxRelMcOffsetForClustering)
				return false;

			const double *pDataA = reinterpret_cast<const double*>(a.vAvgProj.datastart),
						*pDataAEnd = reinterpret_cast<const double*>(a.vAvgProj.datalimit),
						*pDataB = reinterpret_cast<const double*>(b.vAvgProj.datastart);
			for(double sumOfAbsDiffs = 0.; pDataA != pDataAEnd;) {
				sumOfAbsDiffs += abs(*pDataA++ - *pDataB++);
				if(sumOfAbsDiffs > MaxAvgProjErrForClustering)
					return false;
			}

			pDataA = reinterpret_cast<const double*>(a.hAvgProj.datastart);
			pDataAEnd = reinterpret_cast<const double*>(a.hAvgProj.datalimit);
			pDataB = reinterpret_cast<const double*>(b.hAvgProj.datastart);
			for(double sumOfAbsDiffs = 0.; pDataA != pDataAEnd;) {
				sumOfAbsDiffs += abs(*pDataA++ - *pDataB++);
				if(sumOfAbsDiffs > MaxAvgProjErrForClustering)
					return false;
			}

			pDataA = reinterpret_cast<const double*>(a.backslashDiagAvgProj.datastart);
			pDataAEnd = reinterpret_cast<const double*>(a.backslashDiagAvgProj.datalimit);
			pDataB = reinterpret_cast<const double*>(b.backslashDiagAvgProj.datastart);
			for(double sumOfAbsDiffs = 0.; pDataA != pDataAEnd;) {
				sumOfAbsDiffs += abs(*pDataA++ - *pDataB++);
				if(sumOfAbsDiffs > MaxAvgProjErrForClustering)
					return false;
			}

			pDataA = reinterpret_cast<const double*>(a.slashDiagAvgProj.datastart);
			pDataAEnd = reinterpret_cast<const double*>(a.slashDiagAvgProj.datalimit);
			pDataB = reinterpret_cast<const double*>(b.slashDiagAvgProj.datastart);
			for(double sumOfAbsDiffs = 0.; pDataA != pDataAEnd;) {
				sumOfAbsDiffs += abs(*pDataA++ - *pDataB++);
				if(sumOfAbsDiffs > MaxAvgProjErrForClustering)
					return false;
			}

			auto itA = a.mat.begin<double>(), itAEnd = a.mat.end<double>(),
				itB = b.mat.begin<double>();
			for(double sumOfAbsDiffs = 0.; itA != itAEnd;) {
				sumOfAbsDiffs += abs(*itA++ - *itB++);
				if(sumOfAbsDiffs > TotMaxProjErrForClustering)
					return false;
			}

			return true;

#else // DEBUG mode
			const double avgPixDiff = abs(a.avgPixVal - b.avgPixVal);
			const bool bAvgPixDiff = avgPixDiff > MaxDiffAvgPixelValForClustering;
			if(bAvgPixDiff) {
 				++countAvgPixDiff;
				return false;
			}

			const double mcsOffset = norm(a.mc - b.mc);
			const bool bMcsOffset = mcsOffset > MaxRelMcOffsetForClustering;
			if(bMcsOffset) {
 				++countMcsOffset;
				return false;
			}

			const double vDiff = norm(a.vAvgProj - b.vAvgProj, NORM_L1);
			const bool bVdiff = vDiff > MaxAvgProjErrForClustering;
			if(bVdiff) {
 				++countVdiff;
				return false;
			}

			const double hDiff = norm(a.hAvgProj - b.hAvgProj, NORM_L1);
			const bool bHdiff = hDiff > MaxAvgProjErrForClustering;
			if(bHdiff) {
 				++countHdiff;
				return false;
			}

			const double backslashDiff = norm(a.backslashDiagAvgProj - b.backslashDiagAvgProj, NORM_L1);
			const bool bBslashDiff = backslashDiff > MaxAvgProjErrForClustering;
			if(bBslashDiff) {
 				++countBslashDiff;
				return false;
			}

			const double slashDiff = norm(a.slashDiagAvgProj - b.slashDiagAvgProj, NORM_L1);
			const bool bSlashDiff = slashDiff > MaxAvgProjErrForClustering;
			if(bSlashDiff) {
 				++countSlashDiff;
				return false;
			}

			const double diff = norm(a.mat - b.mat, NORM_L1);
			const bool bDiff = diff > TotMaxProjErrForClustering;
			if(bDiff) {
				++countDiff;
				return false;
			}

 			return true;
#endif
		});
		cout<<"The "<<symsCount<<" symbols were clustered in "<<clustersCount<<" groups"<<endl;

#if defined _DEBUG && !defined UNIT_TESTING
		PRINTLN(countAvgPixDiff);
		PRINTLN(countMcsOffset);
		PRINTLN(countVdiff);
		PRINTLN(countHdiff);
		PRINTLN(countBslashDiff);
		PRINTLN(countSlashDiff);
		PRINTLN(countDiff);
#endif

		VSymData newSymsSet;
		newSymsSet.reserve(symsCount);
		vector<vector<unsigned>> symsIndicesPerCluster(clustersCount);
		for(unsigned i = 0U; i<symsCount; ++i)
			symsIndicesPerCluster[clusterLabels[i]].push_back(i);
		
		// Add the clusters in descending order of their size:

		// Typically, there are only a few clusters larger than 1 element.
		// This partition separates the actual formed clusters from one-of-a-kind elements
		// leaving less work to perform to the sort executed afterwards
		auto itFirstClusterWithOneItem = stable_partition(BOUNDS(symsIndicesPerCluster),
												   [] (const vector<unsigned> &a) {
			return a.size() > 1U; // place actual clusters at the beginning of the vector
		});
		unsigned clustered = 0U;
		for_each(begin(symsIndicesPerCluster), itFirstClusterWithOneItem,
				 [&clustered] (const vector<unsigned> &a) { clustered += (unsigned)a.size(); });
		cout<<"There are "<<
			distance(begin(symsIndicesPerCluster), itFirstClusterWithOneItem)
			<<" non-trivial clusters that hold a total of "<<clustered<<" symbols."<<endl;
		
		// Stable partition and sort leave the symbols organized in a more pleasant way than using unstable algorithms.
		stable_sort(begin(symsIndicesPerCluster), itFirstClusterWithOneItem,
			 [] (const vector<unsigned> &a, const vector<unsigned> &b) {
			return a.size() > b.size(); // sort in descending order
		});

		unsigned maxClusterSz = 1U;
		for(unsigned i = 0U, offset = 0U; i<clustersCount; ++i) {
			const auto &symsIndices = symsIndicesPerCluster[i];
			const unsigned clusterSz = (unsigned)symsIndices.size();
			if(clusterSz > maxClusterSz)
				maxClusterSz = clusterSz;

			clusterOffsets.insert(offset);
			clusters.emplace_back(symsSet, offset, symsIndices); // needs symsSet[symsIndices] !!
			for(const auto idx : symsIndices)
				newSymsSet.push_back(move(symsSet[idx])); // destroys symsSet[idx] !!

			offset += clusterSz;
		}
		clusterOffsets.insert(symsCount);

		cout<<"Largest cluster contains "<<maxClusterSz<<" symbols"<<endl;

		symsSet = move(newSymsSet);
	}
} // anonymous namespace

MatchEngine::MatchEngine(const Settings &cfg_, FontEngine &fe_) : cfg(cfg_), fe(fe_) {
	for(const auto &aspectName: MatchAspect::aspectNames())
		availAspects.push_back(
			MatchAspectsFactory::create(aspectName, cachedData, cfg_.matchSettings()));
}

void MatchEngine::updateSymbols() {
	const string idForSymsToUse = getIdForSymsToUse(); // throws for invalid cmap/size
	if(symsIdReady.compare(idForSymsToUse) == 0)
		return; // already up to date

	extern const bool PrepareMoreGlyphsAtOnce;

	symsSet.clear(); clusterOffsets.clear(); clusters.clear();
	const auto &rawSyms = fe.symsSet();
	const int symsCount = (int)rawSyms.size();
	symsSet.reserve(symsCount);

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
		computeSymData(glyph, negGlyph,
					   fgMask, bgMask, edgeMask,
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
	}

	// Clustering symsSet (which gets reordered) - clusterOffsets will point where each cluster starts
	clusterSyms(symsSet, clusters, clusterOffsets);

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
	return clusterOffsets;
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
	locateIdx(clusterOffsets, fromSymIdx, fromCluster, firstSymIdxWithinFromCluster);
	locateIdx(clusterOffsets, upperSymIdx-1, lastCluster, lastSymIdxWithinLastCluster);

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

#ifdef _DEBUG

bool MatchEngine::usesUnicode() const {
	return fe.getEncoding().compare("UNICODE") == 0;
}

#else // _DEBUG not defined

bool MatchEngine::usesUnicode() const { return true; }

#endif // _DEBUG

SymData::SymData() : symAndMasks({ { Mat(), Mat(), Mat(), Mat(), Mat(), Mat(), Mat() } }) {}

ClusterData::ClusterData(const VSymData &symsSet, unsigned idxOfFirstSym_,
						 const vector<unsigned> &clusterSymIndices) : SymData(),
		idxOfFirstSym(idxOfFirstSym_), sz((unsigned)clusterSymIndices.size()) {
	assert(!clusterSymIndices.empty() && !symsSet.empty());
	const Mat &firstNegSym = symsSet[0].symAndMasks[NEG_SYM_IDX];
	const int rows = firstNegSym.rows, cols = firstNegSym.cols;
	double pixelSum = 0.;
	Point2d mc;
	Mat synthesizedSym, negSynthesizedSym(rows, cols, CV_64FC1, Scalar(0.));
	for(const auto clusterSymIdx : clusterSymIndices) {
		const SymData &symData = symsSet[clusterSymIdx];
		Mat negSym;
		symData.symAndMasks[NEG_SYM_IDX].convertTo(negSym, CV_64FC1);
		negSynthesizedSym += negSym;
		pixelSum += symData.pixelSum;
		mc += symData.mc;
	}
	const double invClusterSz = 1./sz;
	negSynthesizedSym *= invClusterSz;
	synthesizedSym = 255. - negSynthesizedSym;
	negSynthesizedSym.convertTo(negSynthesizedSym, CV_8UC1);

	Mat fgMask, bgMask, edgeMask, groundedGlyph, blurOfGroundedGlyph, varianceOfGroundedGlyph;
	double minVal, maxVal; // for very small fonts, minVal might be > 0 and maxVal might be < 255
	computeSymData(synthesizedSym, negSynthesizedSym,
				   fgMask, bgMask, edgeMask,
				   groundedGlyph, blurOfGroundedGlyph, varianceOfGroundedGlyph,
				   minVal, maxVal);
	const_cast<double&>(this->minVal) = minVal;
	const_cast<double&>(diffMinMax) = maxVal - minVal;
	const_cast<double&>(this->pixelSum) = pixelSum * invClusterSz;
	const_cast<Point2d&>(this->mc) = mc * invClusterSz;
	const_cast<Mat&>(symAndMasks[FG_MASK_IDX]) = fgMask;
	const_cast<Mat&>(symAndMasks[BG_MASK_IDX]) = bgMask;
	const_cast<Mat&>(symAndMasks[EDGE_MASK_IDX]) = edgeMask;
	const_cast<Mat&>(symAndMasks[NEG_SYM_IDX]) = negSynthesizedSym;
	const_cast<Mat&>(symAndMasks[GROUNDED_SYM_IDX]) = groundedGlyph;
	const_cast<Mat&>(symAndMasks[BLURRED_GR_SYM_IDX]) = blurOfGroundedGlyph;
	const_cast<Mat&>(symAndMasks[VARIANCE_GR_SYM_IDX]) = varianceOfGroundedGlyph;
}