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
#include "clusterSupportBase.h"
#include "jobMonitorBase.h"
#include "noClustering.h"
#include "partitionClustering.h"
#include "preselectManager.h"
#include "symbolsSupportBase.h"
#include "taskMonitor.h"
#include "ttsasClustering.h"

#ifndef UNIT_TESTING

#include "appStart.h"

#pragma warning(push, 0)

#include <filesystem>

#pragma warning(pop)

#endif  // UNIT_TESTING not defined

using namespace std;
using namespace cv;

extern const string ClusterAlgName;
extern const double MinAverageClusterSize;

namespace {
/// Gets a reference to the clustering algorithm named algName or ignores it for
/// invalid name.
ClusterAlg& algByName(const string& algName) noexcept {
  ClusterAlg* pAlg = nullptr;
  if (algName == TTSAS_Clustering::Name) {
    static TTSAS_Clustering alg;

    pAlg = &alg;
  } else if (algName == PartitionClustering::Name) {
    static PartitionClustering alg;

    pAlg = &alg;
  } else {
    if (algName == NoClustering::Name) {
      cerr << "Unaware of clustering algorithm '" << algName
           << "'! Therefore no clustering will be used!" << endl;
    }
    static NoClustering alg;

    pAlg = &alg;
  }
  return *pAlg;
}

/// Reports various details about the identified clusters
void reportClustersInfo(const vector<vector<unsigned>>& symsIndicesPerCluster,
                        unsigned clustersCount,
                        const VSymData& symsSet) noexcept {
  const auto maxClusterSz = max_element(CBOUNDS(symsIndicesPerCluster),
                                        [](const vector<unsigned>& a,
                                           const vector<unsigned>& b) noexcept {
                                          return a.size() < b.size();
                                        })
                                ->size();
  const auto nonTrivialClusters = (unsigned)count_if(
      CBOUNDS(symsIndicesPerCluster),
      [](const vector<unsigned>& a) noexcept { return a.size() > 1ULL; });
  const auto clusteredSyms =
      (unsigned)symsSet.size() - (clustersCount - nonTrivialClusters);

  cout << "There are " << nonTrivialClusters
       << " non-trivial clusters that hold a total of " << clusteredSyms
       << " symbols.\n"
       << "Largest cluster contains " << maxClusterSz << " symbols." << endl;
}
}  // anonymous namespace

#pragma warning(disable : WARN_BASE_INIT_USING_THIS)
ClusterEngine::ClusterEngine(ITinySymsProvider& tsp_,
                             VSymData& symsSet_) noexcept
    : clusterSupport(IPreselManager::concrete().createClusterSupport(tsp_,
                                                                     *this,
                                                                     symsSet_)),
      clustAlg(algByName(ClusterAlgName).setTinySymsProvider(tsp_)) {
  assert(clusterSupport);  // with preselection or not, but not nullptr
}
#pragma warning(default : WARN_BASE_INIT_USING_THIS)

#pragma warning(disable : WARN_THROWS_ALTHOUGH_NOEXCEPT)
void ClusterEngine::process(VSymData& symsSet,
                            const string& fontType /* = ""*/) noexcept(!UT) {
  if (symsSet.empty())
    return;

  clusters.clear();
  clusterOffsets.clear();
  symsIndicesPerCluster.clear();

  clustersCount = clustAlg.formGroups(symsSet, symsIndicesPerCluster, fontType);

  const double averageClusterSize = (double)symsSet.size() / clustersCount;
  cout << "Average cluster size is " << averageClusterSize << endl;

  static bool checked_symsMonitor = false;
  if (!checked_symsMonitor) {
    if (nullptr == symsMonitor)
      THROW_WITH_CONST_MSG(__FUNCTION__ " should be called only after "
                           "a call to useSymsMonitor()!", logic_error);
    checked_symsMonitor = true;
  }
  static TaskMonitor reorderClusters("reorders clusters", *symsMonitor);

  // Sort symsIndicesPerCluster in ascending order of avgPixVal taken from the
  // first symbol from each cluster
  sort(BOUNDS(symsIndicesPerCluster), [&](const vector<unsigned>& a,
                                          const vector<unsigned>& b) noexcept {
    return symsSet[(size_t)a.front()]->getAvgPixVal() <
           symsSet[(size_t)b.front()]->getAvgPixVal();
  });

  if (averageClusterSize > MinAverageClusterSize) {
    worthy = true;
    reportClustersInfo(symsIndicesPerCluster, clustersCount, symsSet);

    // Trivial clusters are already in ascending order of avgPixVal (see the
    // sort from above)

    clusterSupport->delimitGroups(symsIndicesPerCluster, clusters,
                                  clusterOffsets);
  } else {
    worthy = false;
    cout << "Ignoring clustering due to the low average cluster size." << endl;
  }

  reorderClusters.taskDone();  // mark it as already finished
}
#pragma warning(default : WARN_THROWS_ALTHOUGH_NOEXCEPT)

const VClusterData& ClusterEngine::getClusters() const noexcept {
  assert(worthy);  // adviced
  // When worthGrouping returns false, one shouldn't need clusters
  return clusters;
}

const set<unsigned>& ClusterEngine::getClusterOffsets() const noexcept {
  assert(worthy);  // adviced
  // When worthGrouping returns false, one shouldn't need clusters
  return clusterOffsets;
}

ClusterEngine& ClusterEngine::useSymsMonitor(
    AbsJobMonitor& symsMonitor_) noexcept {
  symsMonitor = &symsMonitor_;
  clustAlg.useSymsMonitor(symsMonitor_);
  return *this;
}

IClustersSupport& ClusterEngine::support() noexcept {
  return *clusterSupport;
}

const IClustersSupport& ClusterEngine::support() const noexcept {
  return *clusterSupport;
}

#ifndef UNIT_TESTING

using namespace std::filesystem;

bool ClusterEngine::clusteredAlready(const string& fontType,
                                     const string& algName,
                                     path& clusteredSetFile) noexcept {
  if (fontType.empty())
    return false;

  clusteredSetFile = AppStart::dir();
  if (!exists(clusteredSetFile.append("ClusteredSets"))) {
    create_directory(clusteredSetFile);
    return false;
  }

  clusteredSetFile.append(fontType).concat("_").concat(algName).concat(
      ".clf");  // CLustered Fonts => clf

  return exists(clusteredSetFile);
}

#endif  // UNIT_TESTING not defined
