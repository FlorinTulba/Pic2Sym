/******************************************************************************
 The application Pic2Sym approximates images by a
 grid of colored symbols with colored backgrounds.

 Copyrights from the libraries used by the program:
 - (c) 2003 Boost (www.boost.org)
     License: doc/licenses/Boost.lic
     http://www.boost.org/LICENSE_1_0.txt
 - (c) 2015-2016 OpenCV (www.opencv.org)
     License: doc/licenses/OpenCV.lic
     http://opencv.org/license/
 - (c) 1996-2002, 2006 The FreeType Project (www.freetype.org)
     License: doc/licenses/FTL.txt
     http://git.savannah.gnu.org/cgit/freetype/freetype2.git/plain/docs/FTL.TXT
 - (c) 1997-2002 OpenMP Architecture Review Board (www.openmp.org)
   (c) Microsoft Corporation (implementation for OpenMP C/C++ v2.0 March 2002)
     See: https://msdn.microsoft.com/en-us/library/8y6825x5.aspx
 - (c) 1995-2017 zlib software (Jean-loup Gailly and Mark Adler - www.zlib.net)
     License: doc/licenses/zlib.lic
     http://www.zlib.net/zlib_license.html


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
 If not, see: http://www.gnu.org/licenses/agpl-3.0.txt .
 *****************************************************************************/

#include "precompiled.h"

#include "clusterEngine.h"
#include "clusterSerialization.h"
#include "jobMonitorBase.h"
#include "taskMonitor.h"
#include "tinySym.h"
#include "tinySymsProvider.h"
#include "ttsasClustering.h"

#pragma warning(push, 0)

#include <algorithm>
#include <map>
#include <set>

#define _USE_MATH_DEFINES
#include <math.h>

#include <optional>

#include <opencv2/imgproc/imgproc.hpp>

#pragma warning(pop)

using namespace std;
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
double threshOutsider(size_t clustSz) noexcept {
  static vector<double> vals{M_PI * M_PI / 6.};

  if (static size_t lastSz = 1U; lastSz <= clustSz) {
    static constexpr size_t IncrementSz = 50ULL;
    const size_t newSz = clustSz + IncrementSz;
    vals.resize(newSz, 0.);
    for (size_t i = lastSz; i < newSz; ++i)
      vals[i] = max(0., vals[i - 1ULL] - 1. / (i * i));
    lastSz = newSz;
  }
  return vals[clustSz];
}

/// Interface for centroids
class ICentroid /*abstract*/ : public virtual ITinySym {
 public:
  /**
  Computing the distance from this centroid to sym as economic as possible.

  @param sym the symbol used for determining the distance
  @param thresholdOutsider allows leaving the computation earlier for
  `outsiders`

  @return the distance between the centroid and sym or infinity for detected
  `outsiders`
  */
  virtual Dist distTo(const ITinySym& sym, const double thresholdOutsider) const
      noexcept = 0;

  /// Moves the centroid of a cluster towards a new member for a distance
  /// depending on a weight
  virtual void shiftTowards(const ITinySym& sym, double weight) noexcept = 0;
};

/// Representative of a cluster of symbols
class Cluster {
 public:
#pragma warning(disable : WARN_INHERITED_VIA_DOMINANCE)
  /// Centroid for clusters
  class Centroid : public TinySym, public ICentroid {
   public:
    /// When seeding a cluster, the centroid of the new cluster is the same as
    /// the first symbol
    explicit Centroid(const ITinySym& firstSym) noexcept : TinySym() {
      setMc(firstSym.getMc());
      setAvgPixVal(firstSym.getAvgPixVal());

      // Clones all matrices from firstSym to allow independent update of the
      // centroid.
      setHAvgProj(firstSym.getHAvgProj().clone());
      setVAvgProj(firstSym.getVAvgProj().clone());
      setBackslashDiagAvgProj(firstSym.getBackslashDiagAvgProj().clone());
      setSlashDiagAvgProj(firstSym.getSlashDiagAvgProj().clone());
      setMat(firstSym.getMat().clone());
    }

    /**
    Computing the distance from this centroid to sym as economic as possible.

    @param sym the symbol used for determining the distance
    @param thresholdOutsider allows leaving the computation earlier for
    `outsiders`

    @return the distance between the centroid and sym or infinity for detected
    `outsiders`
    */
    Dist distTo(const ITinySym& sym, const double thresholdOutsider) const
        noexcept override {
      if (!FastDistSymToClusterComputation) {
        const double l1Dist = norm(getMat() - sym.getMat(), NORM_L1);
        if (l1Dist > thresholdOutsider * TTSAS_Threshold_Member)
          return numeric_limits<Dist>::infinity();
        return l1Dist;
      }

      // Comparing glyph & cluster densities
      if (abs(sym.getAvgPixVal() - getAvgPixVal()) >
          MaxDiffAvgPixelValForTTSAS_Clustering * thresholdOutsider)
        // Skip the rest for very different densities
        return numeric_limits<Dist>::infinity();

      // Comparing glyph & cluster mass-centers
      const Point2d mcDelta = sym.getMc() - getMc();
      const double mcDeltaY = abs(mcDelta.y) / thresholdOutsider;

      // vertical mass-centers offset
      if (mcDeltaY > MaxRelMcOffsetForTTSAS_Clustering)
        // Skip the rest for very distant mass-centers
        return numeric_limits<Dist>::infinity();
      const double mcDeltaX = abs(mcDelta.x) / thresholdOutsider;

      // horizontal mass-centers offset
      if (mcDeltaX > MaxRelMcOffsetForTTSAS_Clustering)
        // Skip the rest for very distant mass-centers
        return numeric_limits<Dist>::infinity();

      static const double SqMaxRelMcOffsetForClustering =
          MaxRelMcOffsetForTTSAS_Clustering * MaxRelMcOffsetForTTSAS_Clustering;

      if (mcDeltaX * mcDeltaX + mcDeltaY * mcDeltaY >
          SqMaxRelMcOffsetForClustering)
        // Skip the rest for very distant mass-centers
        return numeric_limits<Dist>::infinity();

      const double *pDataA = nullptr, *pDataAEnd = nullptr, *pDataB = nullptr,
                   ThresholdOutsider =
                       thresholdOutsider * TTSAS_Threshold_Member;

#define CheckProjections(ProjectionField)                                     \
  pDataA = reinterpret_cast<const double*>(sym.ProjectionField.datastart);    \
  pDataAEnd = reinterpret_cast<const double*>(sym.ProjectionField.datalimit); \
  pDataB = reinterpret_cast<const double*>(ProjectionField.datastart);        \
  for (double sumOfAbsDiffs = 0.; pDataA != pDataAEnd;) {                     \
    sumOfAbsDiffs += abs(*pDataA++ - *pDataB++);                              \
    if (sumOfAbsDiffs > ThresholdOutsider)                                    \
      /* stop as soon as projections appear too different */                  \
      return numeric_limits<Dist>::infinity();                                \
  }

      // Comparing glyph & cluster horizontal, vertical and both diagonal
      // projections
      CheckProjections(getVAvgProj());
      CheckProjections(getHAvgProj());
      CheckProjections(getBackslashDiagAvgProj());
      CheckProjections(getSlashDiagAvgProj());

#undef CheckProjections

      // Comparing glyph & cluster L1 norm
      MatConstIterator_<double> itA = sym.getMat().begin<double>(),
                                itB = getMat().begin<double>();
      const MatConstIterator_<double> itAEnd = sym.getMat().end<double>();
      double sumOfAbsDiffs = 0.;
      while (itA != itAEnd) {
        sumOfAbsDiffs += abs(*itA++ - *itB++);
        if (sumOfAbsDiffs > ThresholdOutsider)
          // Stop as soon as the increasing sum is beyond threshold
          return numeric_limits<Dist>::infinity();
      }

      return sumOfAbsDiffs;
    }

    /// Moves the centroid of a cluster towards a new member for a distance
    /// depending on a weight
    void shiftTowards(const ITinySym& sym, double weight) noexcept override {
      const double oneMinusWeight = 1. - weight;

      setMc(getMc() * oneMinusWeight + weight * sym.getMc());
      setAvgPixVal(getAvgPixVal() * oneMinusWeight +
                   weight * sym.getAvgPixVal());
      setHAvgProj(getHAvgProj() * oneMinusWeight + weight * sym.getHAvgProj());
      setVAvgProj(getVAvgProj() * oneMinusWeight + weight * sym.getVAvgProj());
      setBackslashDiagAvgProj(getBackslashDiagAvgProj() * oneMinusWeight +
                              weight * sym.getBackslashDiagAvgProj());
      setSlashDiagAvgProj(getSlashDiagAvgProj() * oneMinusWeight +
                          weight * sym.getSlashDiagAvgProj());

      Mat mat_ = getMat();
      scaleAdd(mat_, oneMinusWeight, weight * sym.getMat(), mat_);
      setMat(mat_);
    }
  };
#pragma warning(default : WARN_INHERITED_VIA_DOMINANCE)

  vector<SymIdx> memberIndices;  ///< indices of the member symbols
  Centroid centroid;             ///< characteristics of the centroid

  Cluster(const ITinySym& sym, SymIdx symIdx) noexcept
      : memberIndices({symIdx}), centroid(sym) {}

  size_t membersCount() const noexcept { return memberIndices.size(); }

  /// Last added member to the cluster
  SymIdx idxOfLastMember() const noexcept { return memberIndices.back(); }

  /// Updates centroid for the expanded cluster that considers also sym as a
  /// member
  void addMember(const ITinySym& sym, SymIdx symIdx) noexcept {
    memberIndices.push_back(symIdx);
    centroid.shiftTowards(sym, 1. / memberIndices.size());
  }
};

/**
Given a symbol (sym), it tries to find a parent cluster (with index
idxOfParentCluster, located at a distance of distToParentCluster) among the ones
known so far (clusters). If all known clusters appear a bit distant from sym,
reserves will collect the ones that are quite close to sym.

No matter how large the cluster becomes, its members are never more than
    TTSAS_Threshold_Member * 0.645 (pi^2/6-1) far apart from the centroid,
since:
- the thresholds of the parameters decrease proportional to the cluster size
  (Original_Threshold / Expanded_Cluster_Size) - see ICentroid::distTo()
- accepted new members update the cluster's parameters by moving each of them
towards the values of the new members by a decreasing amount, proportional to
the cluster size (Value_From_New_Member / Expanded_Cluster_Size) - see
Cluster::addMember So, if all new members have parameters touching the currently
allowed threshold (Original_Threshold / Expanded_Cluster_Size), then their
individual contributions will be (Original_Threshold / Expanded_Cluster_Size^2)

Thus the total contribution for such a cluster with infinite size and
constructed as described above is: Original_Threshold *
SumFrom2toInfinity(1/i^2)

The sum from above starts from 2, unlike from 1 - as in the Basel problem
(https://en.wikipedia.org/wiki/Basel_problem), so it will be equal to: pi^2/6 -
1 =~ 0.645.
*/
class ParentClusterFinder final {
 public:
  ParentClusterFinder(const ITinySym& sym_,
                      const vector<Cluster>& clusters_) noexcept
      : sym(sym_), clusters(clusters_) {}
  ~ParentClusterFinder() noexcept = default;

  ParentClusterFinder(const ParentClusterFinder&) noexcept = default;
  ParentClusterFinder(ParentClusterFinder&&) noexcept = default;

  // The reference fields are supposed not to change for copy / move
  void operator=(const ParentClusterFinder&) = delete;
  void operator=(ParentClusterFinder&&) = delete;

  bool found() const noexcept { return idxOfParentCluster.has_value(); }

  /// Retrieves the best parent found so far, if any
  const optional<ClusterIdx>& result() const noexcept {
    return idxOfParentCluster;
  }

  /**
  Are there any clusters still near the symbol, but which are currently
  unlikely to be its parent clusters.
  */
  bool anyReserveCandidates() const noexcept { return reserves.empty(); }

  /**
  Moves to dest the clusters that are still near the symbol, but which are
  currently unlikely to be its parent clusters.
  */
  void moveReserveCandidates(NearbyClusters& dest) noexcept {
    dest = move(reserves);
    assert(reserves.empty());
  }

  /**
  Called for each neighbor cluster that didn't change during the previous loop,
  as long as a parent cluster hasn't been found.
  */
  void rememberReserve(ClusterIdx neighborIdx) noexcept {
    reserves.push_back(neighborIdx);
  }

  /// Called when enough clusters were examined
  void prepareReport() noexcept {
    // There is either a found parent cluster or several reserve candidates
    if (found())
      reserves.clear();
  }

  /**
  Checks if cluster clustIdx can be a (better) parent-match for sym.
  If the cluster appear currently too far to be a parent, but still close
  enough, it gets promoted to a 'reserves' status.
  @return true if cluster clustIdx is the best match so far
  */
  bool examine(ClusterIdx clustIdx) noexcept {
    if (TTSAS_Accept1stClusterThatQualifiesAsParent && found())
      return clustIdx == idxOfParentCluster.value();  // keep current parent

    const Cluster& cluster = clusters[(size_t)clustIdx];
    const ICentroid& centroidCluster = cluster.centroid;
    const size_t clusterSz = cluster.membersCount();
    const double expandedClusterSz = double(clusterSz + (size_t)1U);

    // Inf for really distant clusters or if their centroid could become
    // too distant for previous members when including this symbol
    const Dist dist = centroidCluster.distTo(sym, threshOutsider(clusterSz));
    if (dist * expandedClusterSz <
        TTSAS_Threshold_Member) {  // qualifies as parent
      if (TTSAS_Accept1stClusterThatQualifiesAsParent ||
          dist < distToParentCluster) {
        idxOfParentCluster = clustIdx;
        distToParentCluster = dist;
        return true;  // found first/better parent
      }
      return false;  // keep current parent
    }

    if (!isinf(dist))  // dist <= threshOutsider(clusterSz)
      // Qualifies as reserve candidate, but not as parent
      reserves.push_back(clustIdx);

    return false;  // keep current parent
  }

  /**
  Checks if clusters within first..last range contain a (better) parent-match
  for sym. Any clusters that appear currently too far to be parents, but still
  close enough, get promoted to a 'reserves' status.
  @return true if clusters within first..last range contain the best match so
  far
  */
  template <class It>
  bool examine(It first, It last) noexcept {
    if (TTSAS_Accept1stClusterThatQualifiesAsParent) {
      if (found())
        // Keep current parent
        return find(first, last, idxOfParentCluster.value()) != last;

      while (first != last)
        if (examine(*first++))
          // Found first parent and accepted it as the parent of sym
          return true;

      return false;  // keep current parent

    } else {
      // TTSAS_Accept1stClusterThatQualifiesAsParent == false - finding best
      // parent
      bool foundBetterMatch = false;
      while (first != last)
        if (examine(*first++))
          foundBetterMatch = true;

      return foundBetterMatch;
    }
  }

 private:
  const ITinySym& sym;  ///< the symbol whose parent cluster needs to be found
  const vector<Cluster>& clusters;  ///< the clusters known so far

  /// clusters located in TTSAS_Threshold_Member*[1/(Cluster_Size+1) ..
  /// threshOutsider(Cluster_Size)] range from sym
  NearbyClusters reserves;

  /// first / best match for a parent, depending on
  /// TTSAS_Accept1stClusterThatQualifiesAsParent
  optional<ClusterIdx> idxOfParentCluster;

  /// Distance to considered parents
  Dist distToParentCluster = numeric_limits<Dist>::infinity();
};
}  // anonymous namespace

#pragma warning(disable : WARN_THROWS_ALTHOUGH_NOEXCEPT)
unsigned TTSAS_Clustering::formGroups(
    const VSymData& symsToGroup,
    vector<vector<unsigned>>& symsIndicesPerCluster,
    const string& fontType /* = ""*/) noexcept(!UT) {
  static TaskMonitor ttsasClustering("TTSAS clustering", *symsMonitor);

  std::filesystem::path clusteredSetFile;
  ClusterIO rawClustersIO;
  if (!ClusterEngine::clusteredAlready(fontType, Name, clusteredSetFile) ||
      !rawClustersIO.loadFrom(clusteredSetFile.string())) {
    if (tsp == nullptr)
      THROW_WITH_CONST_MSG(__FUNCTION__ " should be called only after calling "
                           "setTinySymsProvider()!", logic_error);

    const VTinySyms& tinySyms = tsp->getTinySyms();

    const size_t countOfTinySymsToGroup = tinySyms.size();
    ttsasClustering.setTotalSteps(countOfTinySymsToGroup);

    vector<int> clusterLabels(countOfTinySymsToGroup);

    size_t clusteredTinySyms = 0ULL;

    // Symbols that still don't belong to the known clusters together with their
    // known neighbor clusters
    map<SymIdx, NearbyClusters> ambiguousTinySyms;
    for (SymIdx i = 0U, tinySymsCount = (unsigned)countOfTinySymsToGroup;
         i < tinySymsCount; ++i)
      ambiguousTinySyms.emplace_hint(ambiguousTinySyms.end(), i,
                                     NearbyClusters());

    vector<Cluster> rawClusters;      // known clusters
    set<ClusterIdx> prevNewClusters,  // indices into clusters to clusters
                                      // created during the previous loop
        prevUpdatedClusters,  // indices into clusters to clusters updated
                              // during the previous loop
        newClusters,  // indices into clusters to clusters created during the
                      // current loop
        updatedClusters;  // indices into clusters to clusters updated during
                          // the current loop

    // Leave when no more ambiguous symbols
    while (!ambiguousTinySyms.empty()) {
      prevNewClusters = move(newClusters);
      prevUpdatedClusters = move(updatedClusters);

      auto itAmbigSym = ambiguousTinySyms.begin();

      // First index of the remaining ambiguous symbol
      SymIdx ambigSymIdx = itAmbigSym->first;

      const auto newSymClustered = [&]() noexcept {
        itAmbigSym = ambiguousTinySyms.erase(itAmbigSym);
        ttsasClustering.taskAdvanced(++clusteredTinySyms);
      };

      const auto createNewCluster = [&]() noexcept {
        newClusters.emplace_hint(newClusters.end(),
                                 (unsigned)rawClusters.size());
        rawClusters.emplace_back(tinySyms[(size_t)ambigSymIdx], ambigSymIdx);
        newSymClustered();
      };

      /*
      Facts:
      - the last added member of each existing cluster is a removed index from
      ambiguousTinySyms during a previous loop
      - ambiguous symbols are traversed in ascending order of their indices
      Conclusions:
      - If a cluster was created/updated during the previous loop after
      ambigSymIdx, it is suffixed with an index > ambigSymIdx
      - If a cluster was created/updated during the previous loop before
      ambigSymIdx, it is suffixed with an index < ambigSymIdx and can be ignored
      by all ambiguous symbols with indices >= ambigSymIdx, since it was already
      checked:
      */
      const auto ignoreAlreadyCheckedClusters = [&](set<ClusterIdx> &
                                                    prevClusters) noexcept {
        // Remove_if doesn't work on sets, so here's the required code
        for (auto itPrevClust = prevClusters.begin();
             itPrevClust != prevClusters.end();) {
          if (rawClusters[size_t(*itPrevClust)].idxOfLastMember() < ambigSymIdx)
            // Remove already processed cluster
            itPrevClust = prevClusters.erase(itPrevClust);
          else  // > ambigSymIdx
            // cluster yet to be processed, keeping it
            ++itPrevClust;
        }
      };

      /*
      If previous loop didn't introduce/affect any cluster,
      then this first ambiguous symbol already has checked the known clusters
      and has detected some neighbor ones which unfortunately didn't expand
      towards this symbol. So this symbol must initiate its own cluster:
      */
      if (prevNewClusters.empty() && prevUpdatedClusters.empty()) {
        createNewCluster();
        if (ambiguousTinySyms.empty())
          break;  // leave when no more ambiguous symbols
      }

      // Traverse the remaining ambiguous symbols and distribute them
      // to new/existing clusters whenever possible
      do {
        // First index of the remaining ambiguous symbol
        ambigSymIdx = itAmbigSym->first;

        // Reference to its modifiable set of neighbors
        NearbyClusters& neighborClusters = itAmbigSym->second;

        ParentClusterFinder pcf(tinySyms[(size_t)ambigSymIdx], rawClusters);
        if (TTSAS_Accept1stClusterThatQualifiesAsParent) {
          // Clusters created/updated later than the previous examination of
          // this symbol must always be checked for paternal match.
          if (!pcf.examine(BOUNDS(newClusters))) {
            // Shrink set before the next check
            ignoreAlreadyCheckedClusters(prevNewClusters);
            if (!pcf.examine(BOUNDS(prevNewClusters))) {
              /*
              The symbol ambigSymIdx was already aware at the start of this loop
              of the clusters updated during previous and current loop.

              It already detected which from them are the nearest and
              it is highly unlikely that some of the rest of the meanwhile
              updated clusters to become neighbors, as well.

              Therefore, among all updated clusters, only those who were already
              neighbors are examined closely.
              */
              ignoreAlreadyCheckedClusters(prevUpdatedClusters);
              // Shrink set before the checks from next for loop

              for (const ClusterIdx neighborIdx : neighborClusters) {
                if (updatedClusters.find(neighborIdx) !=
                        updatedClusters.end() ||
                    prevUpdatedClusters.find(neighborIdx) !=
                        prevUpdatedClusters.end()) {
                  if (pcf.examine(neighborIdx))
                    // reachable as demonstrated in Unit Test:
                    // CheckMemberPromotingReserves_CarefullyOrderedAndChosenSyms_ReserveBecomesParentCluster
                    break;
                } else
                  pcf.rememberReserve(neighborIdx);
              }
            }
          }

        } else {  // TTSAS_Accept1stClusterThatQualifiesAsParent == false
          // Clusters created/updated later than the previous examination of
          // this symbol must always be checked for paternal match.
          pcf.examine(BOUNDS(newClusters));
          ignoreAlreadyCheckedClusters(prevNewClusters);
          // Shrink set before the next check

          pcf.examine(BOUNDS(prevNewClusters));

          /*
          The symbol ambigSymIdx was already aware at the start of this loop
          of the clusters updated during previous and current loop.

          It already detected which from them are the nearest and
          it is highly unlikely that some of the rest of the meanwhile updated
          clusters to become neighbors, as well.

          Therefore, among all updated clusters, only those who were already
          neighbors are examined closely.
          */
          ignoreAlreadyCheckedClusters(prevUpdatedClusters);
          // Shrink set before the checks from next for loop

          for (const ClusterIdx neighborIdx : neighborClusters) {
            if (updatedClusters.find(neighborIdx) != updatedClusters.end() ||
                prevUpdatedClusters.find(neighborIdx) !=
                    prevUpdatedClusters.end())
              pcf.examine(neighborIdx);
            else if (!pcf.found())
              // Reserves don't matter after a parent has been found
              pcf.rememberReserve(neighborIdx);
          }
        }

        pcf.prepareReport();

        if (pcf.found()) {  // identified a cluster for this symbol
          const ClusterIdx parentClusterIdx = pcf.result().value();
          Cluster& updatedCluster = rawClusters[(size_t)parentClusterIdx];
          updatedCluster.addMember(tinySyms[(size_t)ambigSymIdx], ambigSymIdx);

#ifdef _DEBUG
          // Check that centroid's parameters stay within
          // 0.645*Original_Threshold from each member. (see Doxy comment for
          // ParentClusterFinder for details)
          static constexpr double BaselSum_1 = M_PI * M_PI / 6. - 1.;  // ~0.645
          static const double maxDiffAvgPixelVal =
                                  BaselSum_1 *
                                  MaxDiffAvgPixelValForTTSAS_Clustering,
                              maxRelMcOffset =
                                  BaselSum_1 *
                                  MaxRelMcOffsetForTTSAS_Clustering,
                              threshold_Member =
                                  BaselSum_1 * TTSAS_Threshold_Member;

          const double threshold_Outsider =
              threshOutsider(updatedCluster.membersCount());
          const ICentroid& centroid = updatedCluster.centroid;
          for (const auto& memberIdx : updatedCluster.memberIndices) {
            const ITinySym& member = tinySyms[(size_t)memberIdx];
            assert(abs(member.getAvgPixVal() - centroid.getAvgPixVal()) <
                   maxDiffAvgPixelVal);
            assert(norm(member.getMc() - centroid.getMc()) < maxRelMcOffset);
            assert(norm(member.getHAvgProj() - centroid.getHAvgProj(),
                        NORM_L1) < threshold_Outsider);
            assert(norm(member.getVAvgProj() - centroid.getVAvgProj(),
                        NORM_L1) < threshold_Outsider);
            assert(norm(member.getBackslashDiagAvgProj() -
                            centroid.getBackslashDiagAvgProj(),
                        NORM_L1) < threshold_Outsider);
            assert(norm(member.getSlashDiagAvgProj() -
                            centroid.getSlashDiagAvgProj(),
                        NORM_L1) < threshold_Outsider);
            assert(norm(member.getMat() - centroid.getMat(), NORM_L1) <
                   threshold_Member);
          }
#endif  // _DEBUG

          if (newClusters.end() == newClusters.find(parentClusterIdx))
            // Add parentClusterIdx, unless it already appears in newClusters
            updatedClusters.insert(parentClusterIdx);
          newSymClustered();

        } else if (pcf.anyReserveCandidates()) {
          // Way too far from all current clusters
          createNewCluster();

        } else {  // not too close, neither too far from existing clusters
          // Updating the neighbors
          pcf.moveReserveCandidates(neighborClusters);
          ++itAmbigSym;
        }

      } while (itAmbigSym != ambiguousTinySyms.end());
    }

    for (int i = 0, lim = (int)rawClusters.size(); i < lim; ++i) {
      const vector<SymIdx>& clustMembers = rawClusters[(size_t)i].memberIndices;
      for (const auto member : clustMembers)
        clusterLabels[(size_t)member] = i;
    }

    // Fill in rawClustersIO fields
    rawClustersIO.reset((unsigned)rawClusters.size(), move(clusterLabels));
    cout << "\nAll the " << countOfTinySymsToGroup
         << " symbols of the charmap were clustered in "
         << rawClustersIO.getClustersCount() << " groups" << endl;

    rawClustersIO.saveTo(clusteredSetFile.string());
  }

  // Adapt rawClusters for filtered cmap
  symsIndicesPerCluster.assign(rawClustersIO.getClustersCount(),
                               vector<unsigned>());
  for (unsigned i = 0U, lim = (unsigned)symsToGroup.size(); i < lim; ++i)
    symsIndicesPerCluster[(size_t)rawClustersIO.getClusterLabels()
                              [symsToGroup[(size_t)i]->getSymIdx()]]
        .push_back(i);
  const auto newEndIt = remove_if(BOUNDS(symsIndicesPerCluster), [
  ](const vector<unsigned>& elem) noexcept { return elem.empty(); });

  symsIndicesPerCluster.resize(
      (size_t)distance(symsIndicesPerCluster.begin(), newEndIt));

  ttsasClustering.taskDone();

  return (unsigned)symsIndicesPerCluster.size();
}
#pragma warning(default : WARN_THROWS_ALTHOUGH_NOEXCEPT)
