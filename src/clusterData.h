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

#ifndef H_CLUSTER_DATA
#define H_CLUSTER_DATA

#include "symData.h"

struct SymsSupport; // Forward declaration

/**
Synthesized symbol as the representative of several symbols that were clustered together.
Inherits from SymData to qualify in passing as SymData& parameter to assessMatch method.

The specific symbol indices that form the cluster aren't needed since
the clustering regroups the symbols by clusters, so only the index of 1st cluster member and
the cluster size appear as fields.
*/
struct ClusterData : SymData {
	unsigned idxOfFirstSym;	///< index of the first symbol from symsSet that belongs to this cluster
	unsigned sz;			///< size of the cluster - how many symbols form the cluster

	/**
	Constructs a cluster representative for the selected symbols before they get reordered.

	@param symsSet the set of all normal / tiny symbols (before clustering)
	@param idxOfFirstSym_ index of the first symbol from symsSet that belongs to this cluster
	@param clusterSymIndices the indices towards the symbols from symsSet which belong to this cluster
	@param symsSupport ensures that the generated clusters are formed from tiny symbols for preselection mode
	*/
	ClusterData(const VSymData &symsSet, unsigned idxOfFirstSym_,
				const std::vector<unsigned> &clusterSymIndices,
				SymsSupport &symsSupport);

	ClusterData(ClusterData &&other);

	ClusterData(const ClusterData&) = delete;
	void operator=(const ClusterData&) = delete;
	void operator=(ClusterData&&) = delete;
};

/// VClusterData - vector with most information about each cluster
typedef std::vector<const ClusterData> VClusterData;

#endif // H_CLUSTER_DATA