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

#include "ttsasClustering.h"
#include "clusterEngine.h"
#include "clusterSerialization.h"
#include "taskMonitor.h"
#include "tinySym.h"
#include "tinySymsProvider.h"
#include "misc.h"

#include <set>
#include <map>
#include <algorithm>

#define _USE_MATH_DEFINES
#include <math.h>

#include <boost/optional/optional.hpp>

#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace boost;
using namespace cv;

extern const bool FastDistSymToClusterComputation;
extern const bool TTSAS_Accept1stClusterThatQualifiesAsParent;
extern const double TTSAS_Threshold_Member;
extern const double MaxRelMcOffsetForTTSAS_Clustering;
extern const double MaxDiffAvgPixelValForTTSAS_Clustering;

namespace {
	typedef unsigned ClusterIdx; 
	typedef unsigned SymIdx; 
	typedef double Dist;
	typedef vector<ClusterIdx> NearbyClusters;

	/**
	Produces the values of a Basel-like series:
		BaselSeriesLimit - SumOfFirst_N_fromBaselSeries
	or
		pi^2/6 - SumOfFirst_N_for(1/i^2)

	First value from the cache is the limit of the Basel sum: pi^2/6.
	Every new value is the previous minus 1/i^2
	*/
	double threshOutsider(size_t clustSz) {
		static vector<double> vals { M_PI*M_PI/6. };
		static size_t lastSz = 1U;

		if(lastSz <= clustSz) {
			enum {IncrementSz = 50};
			const size_t newSz = clustSz + IncrementSz;
			vals.resize(newSz, 0.);
			for(size_t i = lastSz; i < newSz; ++i)
				vals[i] = max(0., vals[i-1U] - 1./(i*i));
			lastSz = newSz;
		}
		return vals[clustSz];
	}

	/// Representative of a cluster of symbols
	struct Cluster {
		vector<SymIdx> memberIndices;	///< indices of the member symbols
		TinySym centroid;			///< characteristics of the centroid

		Cluster(const TinySym &sym, SymIdx symIdx) :
			memberIndices({ symIdx }),

			/*
			Using normal constructor instead of copy-constructor to clone all matrices.
			Normal construction lets the centroid use the same matrices from
			the cluster root symbol (parameter sym from here),
			so any addMember() call would change the root symbol, as well.
			*/
			centroid(sym.mc, sym.avgPixVal, sym.mat.clone(),
					sym.hAvgProj.clone(), sym.vAvgProj.clone(),
					sym.backslashDiagAvgProj.clone(), sym.slashDiagAvgProj.clone()) {}

		size_t membersCount() const { return memberIndices.size(); }

		/// Last added member to the cluster
		SymIdx idxOfLastMember() const { return memberIndices.back(); }
		
		/// Updates centroid for the expanded cluster that considers also sym as a member
		void addMember(const TinySym &sym, SymIdx symIdx) {
			memberIndices.push_back(symIdx);
			const double invNewMembersCount = 1./memberIndices.size(),
						oneMinusInvNewMembersCount = 1. - invNewMembersCount;

			// All fields suffer the transformation below due to the expansion of the cluster
#define UpdateCentroidField(field) \
			centroid.field = centroid.field * oneMinusInvNewMembersCount + sym.field * invNewMembersCount

			UpdateCentroidField(mc);
			UpdateCentroidField(avgPixVal);
			UpdateCentroidField(hAvgProj);
			UpdateCentroidField(vAvgProj);
			UpdateCentroidField(backslashDiagAvgProj);
			UpdateCentroidField(slashDiagAvgProj);

#undef UpdateCentroidField

			scaleAdd(centroid.mat, oneMinusInvNewMembersCount, sym.mat * invNewMembersCount, centroid.mat);
		}
	};

	/**
	Given a symbol (sym), it tries to find a parent cluster (with index idxOfParentCluster,
	located at a distance of distToParentCluster) among the ones known so far (clusters).
	If all known clusters appear a bit distant from sym, reserves will collect the ones that are
	quite close to sym.

	No matter how large the cluster becomes, its members are never more than
			TTSAS_Threshold_Member * 0.645 (pi^2/6-1) far apart from the centroid, since:
	- the thresholds of the parameters decrease proportional to the cluster size
	  (Original_Threshold / Expanded_Cluster_Size) - see distToCluster
	- accepted new members update the cluster's parameters by moving each of them towards the
	  values of the new members by a decreasing amount, proportional to the cluster size
	  (Value_From_New_Member / Expanded_Cluster_Size) - see Cluster::addMember
	So, if all new members have parameters touching the currently allowed threshold (Original_Threshold / Expanded_Cluster_Size),
	then their individual contributions will be (Original_Threshold / Expanded_Cluster_Size^2)
	
	Thus the total contribution for such a cluster with infinite size and constructed as described above is:
		Original_Threshold * SumFrom2toInfinity(1/i^2)

	The sum from above starts from 2, unlike from 1 - as in the Basel problem (https://en.wikipedia.org/wiki/Basel_problem),
	so it will be equal to: pi^2/6 - 1 =~ 0.645.
	*/
	class ParentClusterFinder {
	protected:
		const TinySym &sym;					///< the symbol whose parent cluster needs to be found
		const vector<Cluster> &clusters;	///< the clusters known so far

		/// clusters located in TTSAS_Threshold_Member*[1/(Cluster_Size+1) .. threshOutsider(Cluster_Size)] range from sym
		NearbyClusters reserves;

		/// first / best match for a parent, depending on TTSAS_Accept1stClusterThatQualifiesAsParent
		optional<ClusterIdx> idxOfParentCluster;
		Dist distToParentCluster = numeric_limits<Dist>::infinity(); ///< distance to considered parent

		/**
		Computing the distance from sym to the cluster with clustIdx index as economic as possible.
		Parameter clusterSz is used to lower the thresholds of all parameters
		for every increase in the size of the cluster.
		In this way, the centroid of each expanding cluster remains close enough to any of its members.
		*/
		Dist distToCluster(const TinySym &centroidCluster, size_t clusterSz) {
			const double thresholdOutsider = threshOutsider(clusterSz);

			if(!FastDistSymToClusterComputation) {
				const double l1Dist = norm(centroidCluster.mat - sym.mat, NORM_L1);
				if(l1Dist > thresholdOutsider * TTSAS_Threshold_Member)
					return numeric_limits<Dist>::infinity();
				return l1Dist;
			}

			// Comparing glyph & cluster densities
			if(abs(sym.avgPixVal - centroidCluster.avgPixVal)  >
						MaxDiffAvgPixelValForTTSAS_Clustering * thresholdOutsider)
				return numeric_limits<Dist>::infinity(); // skip the rest for very different densities

			// Comparing glyph & cluster mass-centers
			const Point2d mcDelta = sym.mc - centroidCluster.mc;
			const double mcDeltaY = abs(mcDelta.y) / thresholdOutsider;
			if(mcDeltaY > MaxRelMcOffsetForTTSAS_Clustering) // vertical mass-centers offset
				return numeric_limits<Dist>::infinity(); // skip the rest for very distant mass-centers
			const double mcDeltaX = abs(mcDelta.x) / thresholdOutsider;
			if(mcDeltaX > MaxRelMcOffsetForTTSAS_Clustering) // horizontal mass-centers offset
				return numeric_limits<Dist>::infinity(); // skip the rest for very distant mass-centers
			static const double SqMaxRelMcOffsetForClustering =
				MaxRelMcOffsetForTTSAS_Clustering * MaxRelMcOffsetForTTSAS_Clustering;
			if(mcDeltaX*mcDeltaX + mcDeltaY*mcDeltaY > SqMaxRelMcOffsetForClustering)
				return numeric_limits<Dist>::infinity(); // skip the rest for very distant mass-centers

			const double *pDataA, *pDataAEnd, *pDataB,
						ThresholdOutsider = thresholdOutsider * TTSAS_Threshold_Member;

#define CheckProjections(ProjectionField) \
			pDataA = reinterpret_cast<const double*>(sym.ProjectionField.datastart); \
			pDataAEnd = reinterpret_cast<const double*>(sym.ProjectionField.datalimit); \
			pDataB = reinterpret_cast<const double*>(centroidCluster.ProjectionField.datastart); \
			for(double sumOfAbsDiffs = 0.; pDataA != pDataAEnd;) { \
				sumOfAbsDiffs += abs(*pDataA++ - *pDataB++); \
				if(sumOfAbsDiffs > ThresholdOutsider) \
					return numeric_limits<Dist>::infinity(); /* stop as soon as projections appear too different */ \
			}

			// Comparing glyph & cluster horizontal, vertical and both diagonal projections
			CheckProjections(vAvgProj);
			CheckProjections(hAvgProj);
			CheckProjections(backslashDiagAvgProj);
			CheckProjections(slashDiagAvgProj);

#undef CheckProjections

			// Comparing glyph & cluster L1 norm
			auto itA = sym.mat.begin<double>(), itAEnd = sym.mat.end<double>(),
				itB = centroidCluster.mat.begin<double>();
			double sumOfAbsDiffs = 0.;
			while(itA != itAEnd) {
				sumOfAbsDiffs += abs(*itA++ - *itB++);
				if(sumOfAbsDiffs > ThresholdOutsider)
					return numeric_limits<Dist>::infinity(); // stop as soon as the increasing sum is beyond threshold
			}

			return sumOfAbsDiffs;
		}

	public:
		ParentClusterFinder(const TinySym &sym_, const vector<Cluster> &clusters_) :
			sym(sym_), clusters(clusters_) {}

		bool found() const { return idxOfParentCluster.is_initialized(); }

		/// Retrieves the best parent found so far, if any
		const optional<ClusterIdx>& result() const { return idxOfParentCluster; }

		/**
		Provides access to the clusters that are still near the symbol, but currently are unlikely
		to be its parent clusters.
		*/
		const NearbyClusters& reserveCandidates() const { return reserves; }

		/**
		Called for each neighbor cluster that didn't change during the previous loop,
		as long as a parent cluster hasn't been found.
		*/
		void rememberReserve(ClusterIdx neighborIdx) { reserves.push_back(neighborIdx); }

		/// Called when enough clusters were examined
		void prepareReport() {
			// There is either a found parent cluster or several reserve candidates
			if(found()) 
				reserves.clear();
		}

		/**
		Checks if cluster clustIdx can be a (better) parent-match for sym.
		If the cluster appear currently too far to be a parent, but still close enough, 
		it gets promoted to a 'reserves' status.
		@return true if cluster clustIdx is the best match so far
		*/
		bool examine(ClusterIdx clustIdx) {
			if(TTSAS_Accept1stClusterThatQualifiesAsParent && found())
				return clustIdx == idxOfParentCluster.get(); // keep current parent
			
			const Cluster &cluster = clusters[clustIdx];
			const TinySym &centroidCluster = cluster.centroid;
			const size_t clusterSz = cluster.membersCount();
			const double expandedClusterSz = double(clusterSz + (size_t)1U);

			// Inf for really distant clusters or if their centroid could become 
			// too distant for previous members when including this symbol
			const Dist dist = distToCluster(centroidCluster, clusterSz);
			if(dist * expandedClusterSz < TTSAS_Threshold_Member) { // qualifies as parent
				if(TTSAS_Accept1stClusterThatQualifiesAsParent || dist < distToParentCluster) {
					idxOfParentCluster = clustIdx;
					distToParentCluster = dist;
					return true; // found first/better parent
				}
				return false; // keep current parent
			}
			
			if(!isinf(dist)) // dist <= threshOutsider(clusterSz)
				reserves.push_back(clustIdx); // qualifies as reserve candidate, but not as parent

			return false; // keep current parent
		}

		/**
		Checks if clusters within first..last range contain a (better) parent-match for sym.
		Any clusters that appear currently too far to be parents, but still close enough,
		get promoted to a 'reserves' status.
		@return true if clusters within first..last range contain the best match so far
		*/
		template<class It>
		bool examine(It first, It last) {
			if(TTSAS_Accept1stClusterThatQualifiesAsParent) {
				if(found())
					return find(first, last, idxOfParentCluster.get()) != last; // keep current parent

				while(first != last)
					if(examine(*first++))
						return true; // found first parent and accepted it as the parent of sym

				return false; // keep current parent

			} else { // TTSAS_Accept1stClusterThatQualifiesAsParent == false - finding best parent
				bool foundBetterMatch = false;
				while(first != last)
					if(examine(*first++))
						foundBetterMatch = true;

				return foundBetterMatch;
			}
		}
	};
} // anonymous namespace

const string TTSAS_Clustering::Name("TTSAS");

unsigned TTSAS_Clustering::formGroups(const VSymData &symsToGroup,
									  vector<vector<unsigned>> &symsIndicesPerCluster,
									  const string &fontType/* = ""*/) {
	static TaskMonitor ttsasClustering("TTSAS clustering", *symsMonitor);

	boost::filesystem::path clusteredSetFile;
	ClusterIO rawClustersIO;
	if(!ClusterEngine::clusteredAlready(fontType, Name, clusteredSetFile)
			|| !rawClustersIO.loadFrom(clusteredSetFile.string())) {
		if(tsp == nullptr)
			THROW_WITH_CONST_MSG(__FUNCTION__ " should be called only after calling setTinySymsProvider()!", logic_error);

		const auto &tinySyms = tsp->getTinySyms();

		const size_t countOfTinySymsToGroup = tinySyms.size();
		ttsasClustering.setTotalSteps(countOfTinySymsToGroup);
		
		rawClustersIO.clusterLabels.resize(countOfTinySymsToGroup);

		size_t clusteredTinySyms = 0U;

		// Symbols that still don't belong to the known clusters together with their known neighbor clusters
		map<SymIdx, NearbyClusters> ambiguousTinySyms;
		for(SymIdx i = 0U, tinySymsCount = (unsigned)countOfTinySymsToGroup; i < tinySymsCount; ++i)
			ambiguousTinySyms.emplace_hint(ambiguousTinySyms.end(), i, NearbyClusters());

		vector<Cluster> rawClusters; // known clusters
		set<ClusterIdx> prevNewClusters,	// indices into clusters to clusters created during the previous loop
			prevUpdatedClusters,// indices into clusters to clusters updated during the previous loop
			newClusters,		// indices into clusters to clusters created during the current loop
			updatedClusters;	// indices into clusters to clusters updated during the current loop

		while(!ambiguousTinySyms.empty()) { // leave when no more ambiguous symbols
			prevNewClusters = std::move(newClusters);
			prevUpdatedClusters = std::move(updatedClusters);

			auto itAmbigSym = ambiguousTinySyms.begin();
			SymIdx ambigSymIdx = itAmbigSym->first; // first index of the remaining ambiguous symbol

			const auto newSymClustered = [&] {
				itAmbigSym = ambiguousTinySyms.erase(itAmbigSym);
				ttsasClustering.taskAdvanced(++clusteredTinySyms);
			};

			const auto createNewCluster = [&] {
				newClusters.emplace_hint(newClusters.end(), (unsigned)rawClusters.size());
				rawClusters.emplace_back(tinySyms[ambigSymIdx], ambigSymIdx);
				newSymClustered();
			};

			/*
			Facts:
			- the last added member of each existing cluster is a removed index from ambiguousTinySyms during a previous loop
			- ambiguous symbols are traversed in ascending order of their indices
			Conclusions:
			- If a cluster was created/updated during the previous loop after ambigSymIdx,
			it is suffixed with an index > ambigSymIdx
			- If a cluster was created/updated during the previous loop before ambigSymIdx,
			it is suffixed with an index < ambigSymIdx
			and can be ignored by all ambiguous symbols with indices >= ambigSymIdx,
			since it was already checked:
			*/
			const auto ignoreAlreadyCheckedClusters = [&] (set<ClusterIdx> &prevClusters) {
				// Remove_if doesn't work on sets, so here's the required code
				for(auto itPrevClust = prevClusters.begin(); itPrevClust != prevClusters.end();) {
					if(rawClusters[*itPrevClust].idxOfLastMember() < ambigSymIdx)
						itPrevClust = prevClusters.erase(itPrevClust); // remove already processed cluster
					else // > ambigSymIdx
						++itPrevClust; // cluster yet to be processed, keeping it
				}
			};

			/*
			If previous loop didn't introduce/affect any cluster,
			then this first ambiguous symbol already has checked the known clusters
			and has detected some neighbor ones which unfortunately didn't expand towards this symbol.
			So this symbol must initiate its own cluster:
			*/
			if(prevNewClusters.empty() && prevUpdatedClusters.empty()) {
				createNewCluster();
				if(ambiguousTinySyms.empty())
					break; // leave when no more ambiguous symbols
			}

			// Traverse the remaining ambiguous symbols and distribute them
			// to new/existing clusters whenever possible
			do {
				ambigSymIdx = itAmbigSym->first; // first index of the remaining ambiguous symbol
				auto &neighborClusters = itAmbigSym->second; // reference to its modifiable set of neighbors

				ParentClusterFinder pcf(tinySyms[ambigSymIdx], rawClusters);
				if(TTSAS_Accept1stClusterThatQualifiesAsParent) {
					// Clusters created/updated later than the previous examination of this symbol must always be checked for paternal match.
					if(!pcf.examine(BOUNDS(newClusters))) {
						ignoreAlreadyCheckedClusters(prevNewClusters); // shrink set before the next check
						if(!pcf.examine(BOUNDS(prevNewClusters))) {
							/*
							The symbol ambigSymIdx was already aware at the start of this loop
							of the clusters updated during previous and current loop.

							It already detected which from them are the nearest and
							it is highly unlikely that some of the rest of the meanwhile updated clusters
							to become neighbors, as well.

							Therefore, among all updated clusters, only those who were already neighbors
							are examined closely.
							*/
							ignoreAlreadyCheckedClusters(prevUpdatedClusters); // shrink set before the checks from next for loop
							for(const ClusterIdx neighborIdx : neighborClusters) {
								if(updatedClusters.find(neighborIdx) != updatedClusters.end()
										|| prevUpdatedClusters.find(neighborIdx) != prevUpdatedClusters.end()) {
									if(pcf.examine(neighborIdx))
										// reachable as demonstrated in Unit Test:
										// CheckMemberPromotingReserves_CarefullyOrderedAndChosenSyms_ReserveBecomesParentCluster
										break;
								} else pcf.rememberReserve(neighborIdx);
							}
						}
					}

				} else { // TTSAS_Accept1stClusterThatQualifiesAsParent == false
					// Clusters created/updated later than the previous examination of this symbol must always be checked for paternal match.
					pcf.examine(BOUNDS(newClusters));
					ignoreAlreadyCheckedClusters(prevNewClusters); // shrink set before the next check
					pcf.examine(BOUNDS(prevNewClusters));

					/*
					The symbol ambigSymIdx was already aware at the start of this loop
					of the clusters updated during previous and current loop.

					It already detected which from them are the nearest and
					it is highly unlikely that some of the rest of the meanwhile updated clusters
					to become neighbors, as well.

					Therefore, among all updated clusters, only those who were already neighbors
					are examined closely.
					*/
					ignoreAlreadyCheckedClusters(prevUpdatedClusters); // shrink set before the checks from next for loop
					for(const ClusterIdx neighborIdx : neighborClusters) {
						if(updatedClusters.find(neighborIdx) != updatedClusters.end()
							   || prevUpdatedClusters.find(neighborIdx) != prevUpdatedClusters.end())
						   pcf.examine(neighborIdx);
						else if(!pcf.found()) // Reserves don't matter after a parent has been found
							pcf.rememberReserve(neighborIdx);
					}
				}

				pcf.prepareReport();

				if(pcf.found()) { // identified a cluster for this symbol
					const ClusterIdx parentClusterIdx = pcf.result().get();
					auto &updatedCluster = rawClusters[parentClusterIdx];
					updatedCluster.addMember(tinySyms[ambigSymIdx], ambigSymIdx);

#ifdef _DEBUG
					// Check that centroid's parameters stay within 0.645*Original_Threshold from each member.
					// (see Doxy comment for ParentClusterFinder for details)
					static const double BaselSum_1 = M_PI * M_PI / 6. - 1., // ~0.645
									maxDiffAvgPixelVal = BaselSum_1 * MaxDiffAvgPixelValForTTSAS_Clustering,
									maxRelMcOffset = BaselSum_1 * MaxRelMcOffsetForTTSAS_Clustering,
									threshold_Member = BaselSum_1 * TTSAS_Threshold_Member;
					const double threshold_Outsider = threshOutsider(updatedCluster.membersCount());
					const auto &centroid = updatedCluster.centroid;
					for(const auto &memberIdx : updatedCluster.memberIndices) {
						const auto &member = tinySyms[memberIdx];
						assert(abs(member.avgPixVal - centroid.avgPixVal) < maxDiffAvgPixelVal);
						assert(norm(member.mc - centroid.mc) < maxRelMcOffset);
						assert(norm(member.hAvgProj - centroid.hAvgProj, NORM_L1) < threshold_Outsider);
						assert(norm(member.vAvgProj - centroid.vAvgProj, NORM_L1) < threshold_Outsider);
						assert(norm(member.backslashDiagAvgProj - centroid.backslashDiagAvgProj, NORM_L1) < threshold_Outsider);
						assert(norm(member.slashDiagAvgProj - centroid.slashDiagAvgProj, NORM_L1) < threshold_Outsider);
						assert(norm(member.mat - centroid.mat, NORM_L1) < threshold_Member);
					}
#endif // _DEBUG

					if(newClusters.end() == newClusters.find(parentClusterIdx))
						updatedClusters.insert(parentClusterIdx); // add parentClusterIdx, unless it already appears in newClusters
					newSymClustered();

				} else if(pcf.reserveCandidates().empty()) { // way too far from all current clusters
					createNewCluster();

				} else { // not too close, neither too far from existing clusters
					neighborClusters = std::move(const_cast<NearbyClusters&>(pcf.reserveCandidates())); // updating the neighbors
					++itAmbigSym;
				}

			} while(itAmbigSym != ambiguousTinySyms.end());
		}

		// Fill in rawClustersIO fields
		rawClustersIO.clustersCount = (unsigned)rawClusters.size();
		for(int i = 0, lim = (int)rawClustersIO.clustersCount; i < lim; ++i) {
			const auto &clustMembers = rawClusters[i].memberIndices;
			for(const auto member : clustMembers)
				rawClustersIO.clusterLabels[member] = i;
		}
		
		cout<<endl<<"All the "<<countOfTinySymsToGroup<<" symbols of the charmap were clustered in "
			<<rawClustersIO.clustersCount<<" groups"<<endl;

		rawClustersIO.saveTo(clusteredSetFile.string());
	}

	// Adapt rawClusters for filtered cmap
	symsIndicesPerCluster.assign(rawClustersIO.clustersCount, vector<unsigned>());
	for(unsigned i = 0U, lim = (unsigned)symsToGroup.size(); i < lim; ++i)
		symsIndicesPerCluster[rawClustersIO.clusterLabels[symsToGroup[i].symIdx]].push_back(i);
	const auto newEndIt = remove_if(BOUNDS(symsIndicesPerCluster),
									[] (const vector<unsigned> &elem) { return elem.empty(); });
	symsIndicesPerCluster.resize(distance(symsIndicesPerCluster.begin(), newEndIt));

	ttsasClustering.taskDone();

	return (unsigned)symsIndicesPerCluster.size();
}