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

#ifndef H_CLUSTER_ENGINE
#define H_CLUSTER_ENGINE

#include "clusterDataBase.h"
#include "clusterAlg.h"

#pragma warning ( push, 0 )

#include "std_memory.h"
#include <set>

#include "boost_filesystem_path.h"

#pragma warning ( pop )

// Forward declarations
class AbsJobMonitor;
struct ITinySymsProvider;
struct IClustersSupport;

/// The cluster processing interface of the ClusterEngine used within ClusterSupport*
struct IClusterProcessing /*abstract*/ {
	/**
	Clusters symsSet & tinySymsSet into clusters, while clusterOffsets reports where each cluster starts.
	When using the tiny symbols preselection, the clusters will contain tiny symbols.
	@param symsSet original symbols to be clustered
	@param fontType allows checking for previously conducted clustering of current font type; empty for various unit tests
	*/
	virtual void process(VSymData &symsSet, const std::stringType &fontType = "") = 0;

	virtual unsigned getClustersCount() const = 0;

	virtual ~IClusterProcessing() = 0 {}
};

/// Interface for the ClusterEngine
struct IClusterEngine /*abstract*/ : IClusterProcessing {
	virtual const bool& worthGrouping() const = 0;
	virtual const std::vector<std::vector<unsigned>>& getSymsIndicesPerCluster() const = 0;

	/**
	The clustered symbols. When using the tiny symbols preselection, the clusters will contain tiny symbols.
	Use it only if worthGrouping() returns true.
	@return clusters
	*/
	virtual const VClusterData& getClusters() const = 0;
	virtual const std::set<unsigned>& getClusterOffsets() const = 0; ///< returns clusterOffsets ; use it only if worthGrouping() returns true

	virtual IClusterEngine& useSymsMonitor(AbsJobMonitor &symsMonitor_) = 0; ///< setting the symbols monitor

	virtual IClustersSupport& support() = 0; ///< access to clusterSupport
	virtual const IClustersSupport& support() const = 0; ///< access to clusterSupport

	virtual ~IClusterEngine() = 0 {}
};

/// Clusters a set of symbols
class ClusterEngine : public IClusterEngine {
protected:
	/// observer of the symbols' loading, filtering and clustering, who reports their progress
	AbsJobMonitor *symsMonitor = nullptr;
	std::uniquePtr<IClustersSupport> clusterSupport;	///< provided support from the preselection manager

	ClusterAlg &clustAlg;	///< algorithm used for clustering

	/// The clustered symbols. When using the tiny symbols preselection, the clusters will contain tiny symbols.
	VClusterData clusters;
	std::set<unsigned> clusterOffsets;	///< start indices in symsSet where each cluster starts
	std::vector<std::vector<unsigned>> symsIndicesPerCluster; ///< indices of the member symbols from each cluster
	unsigned clustersCount = 0U;		///< number of clusters
	bool worthy = false;				///< grouping symbols is worth-doing only above a threshold average cluster size

public:
	ClusterEngine(ITinySymsProvider &tsp_, VSymData &symsSet_); ///< Creates the cluster algorithm prescribed in varConfig.txt
	ClusterEngine(const ClusterEngine&) = delete;
	void operator=(const ClusterEngine&) = delete;

	/**
	Determines if fontType was already clustered using algName clustering algorithm.
	The path to the file supposed to contain the clustering results is returned in the clusteredSetFile parameter.
	*/
	static bool clusteredAlready(const std::stringType &fontType, const std::stringType &algName,
								 boost::filesystem::path &clusteredSetFile);
	
	/**
	Clusters symsSet & tinySymsSet into clusters, while clusterOffsets reports where each cluster starts.
	When using the tiny symbols preselection, the clusters will contain tiny symbols.
	@param symsSet original symbols to be clustered
	@param fontType allows checking for previously conducted clustering of current font type; empty for various unit tests
	*/
	void process(VSymData &symsSet, const std::stringType &fontType = "") override;
	unsigned getClustersCount() const override final { return clustersCount; }

	const bool& worthGrouping() const override final { return worthy; }
	const std::vector<std::vector<unsigned>>& getSymsIndicesPerCluster() const override final {
		return symsIndicesPerCluster; 
	}

	/**
	The clustered symbols. When using the tiny symbols preselection, the clusters will contain tiny symbols.
	Use it only if worthGrouping() returns true.
	@return clusters
	*/
	const VClusterData& getClusters() const override;
	const std::set<unsigned>& getClusterOffsets() const override; ///< returns clusterOffsets ; use it only if worthGrouping() returns true

	ClusterEngine& useSymsMonitor(AbsJobMonitor &symsMonitor_) override; ///< setting the symbols monitor

	IClustersSupport& support() override final;				///< access to clusterSupport
	const IClustersSupport& support() const override final; ///< access to clusterSupport
};

#endif // H_CLUSTER_ENGINE
