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

#ifndef H_CLUSTER_ALG
#define H_CLUSTER_ALG

#include "symData.h"

#pragma warning ( push, 0 )

#include <string>
#include <vector>

#pragma warning ( pop )

// Forward declarations
class AbsJobMonitor;
struct ITinySymsProvider;

/// Abstract class for clustering algorithms
struct ClusterAlg /*abstract*/ {
	/// Gets a reference to the clustering algorithm named algName or ignores it for invalid name.
	static ClusterAlg& algByName(const std::string &algName);

	virtual ~ClusterAlg() = 0 {}

	/**
	Performs clustering of a set of symbols.
	
	@param symsToGroup symbols to be grouped by similarity
	@param symsIndicesPerCluster returned vector of clusters, each cluster with the indices towards member symbols
	@param fontType font family, style and encoding (not the size); empty for various unit tests

	@return number of clusters obtained
	*/
	virtual unsigned formGroups(const VSymData &symsToGroup,
								std::vector<std::vector<unsigned>> &symsIndicesPerCluster,
								const std::string &fontType = "") = 0;

	/// Assigns to tsp the provider of tiny symbols
	ClusterAlg& setTinySymsProvider(ITinySymsProvider &tsp_);
	
	ClusterAlg& useSymsMonitor(AbsJobMonitor &symsMonitor_); ///< setting the symbols monitor

protected:
	ITinySymsProvider *tsp = nullptr;	///< offers support for tiny symbols

	/// observer of the symbols' loading, filtering and clustering, who reports their progress
	AbsJobMonitor *symsMonitor = nullptr;
};

#endif