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

#ifndef H_CLUSTER_ENGINE
#define H_CLUSTER_ENGINE

#include "clusterAlg.h"
#include "clusterEngineBase.h"

#pragma warning(push, 0)

#include <filesystem>

#pragma warning(pop)

class ITinySymsProvider;  // forward declaration

/// Clusters a set of symbols
class ClusterEngine : public IClusterEngine {
 public:
  /// Creates the cluster algorithm prescribed in varConfig.txt
  ClusterEngine(ITinySymsProvider& tsp_, VSymData& symsSet_) noexcept;

  /**
  Determines if fontType was already clustered using algName clustering
  algorithm. The path to the file supposed to contain the clustering results is
  returned in the clusteredSetFile parameter.
  */
  static bool clusteredAlready(
      const std::string& fontType,
      const std::string& algName,
      std::filesystem::path& clusteredSetFile) noexcept;

  /**
  Clusters symsSet & tinySymsSet into clusters, while clusterOffsets reports
  where each cluster starts. When using the tiny symbols preselection, the
  clusters will contain tiny symbols.
  @param symsSet original symbols to be clustered
  @param fontType allows checking for previously conducted clustering of current
  font type; empty for various unit tests
  @throw logic_error if called before useSymsMonitor()

  Exception to be only reported, not handled
  */
  void process(VSymData& symsSet,
               const std::string& fontType = "") noexcept(!UT) override;

  unsigned getClustersCount() const noexcept final { return clustersCount; }

  /// @return true if clustering should increase transformation performance
  const bool& worthGrouping() const noexcept final { return worthy; }

  /// @return for each cluster a vector of the symbols belonging to it
  const std::vector<std::vector<unsigned>>& getSymsIndicesPerCluster() const
      noexcept final {
    return symsIndicesPerCluster;
  }

  /**
  The clustered symbols. When using the tiny symbols preselection, the clusters
  will contain tiny symbols. Use it only if worthGrouping() returns true.
  @return clusters

  Advice: Use it only if worthGrouping() returns true.
  */
  const VClusterData& getClusters() const noexcept override;

  /**
  @return clusterOffsets

  Advice: Use it only if worthGrouping() returns true.
  */
  const std::set<unsigned>& getClusterOffsets() const noexcept override;

  /// Setting the symbols monitor
  ClusterEngine& useSymsMonitor(AbsJobMonitor& symsMonitor_) noexcept override;

  /// Access to clusterSupport
  IClustersSupport& support() noexcept final;

  /// Access to clusterSupport
  const IClustersSupport& support() const noexcept final;

  PRIVATE :

      /// observer of the symbols' loading, filtering and clustering, who
      /// reports their progress
      AbsJobMonitor* symsMonitor = nullptr;

  /// Provided support from the preselection manager
  const std::unique_ptr<IClustersSupport> clusterSupport;

  ClusterAlg& clustAlg;  ///< algorithm used for clustering

  /// The clustered symbols. When using the tiny symbols preselection, the
  /// clusters will contain tiny symbols.
  VClusterData clusters;

  /// Start indices in symsSet where each cluster starts
  std::set<unsigned> clusterOffsets;

  /// Indices of the member symbols from each cluster
  std::vector<std::vector<unsigned>> symsIndicesPerCluster;

  unsigned clustersCount = 0U;  ///< number of clusters

  /// Grouping symbols is worth-doing only above a threshold average cluster
  /// size
  bool worthy = false;
};

#endif  // H_CLUSTER_ENGINE
