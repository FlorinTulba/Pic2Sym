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

#ifndef H_CLUSTER_ENGINE_BASE
#define H_CLUSTER_ENGINE_BASE

#include "clusterProcessingBase.h"
#include "clusterDataBase.h"

#pragma warning ( push, 0 )

#include <set>

#pragma warning ( pop )

// Forward declarations
class AbsJobMonitor;
struct IClustersSupport;

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

#endif // H_CLUSTER_ENGINE_BASE
