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

#ifndef H_TTSAS_CLUSTERING
#define H_TTSAS_CLUSTERING

#include "clusterAlg.h"

/**
Implementation of the "Two-Threshold Sequential Algorithmic Scheme" clustering algorithm.
(see 3.1.2 and 5.2 from https://www.niksula.hut.fi/~jkainula/pdfs/clustering.pdf for details)

Although theoretically its performance is O(N^2), it runs as fast as the 2 set thresholds allow and
it is less influenced by the initial order of the symbols to be grouped.

It was customized to:
- maintain compact clusters (centroids remain close to each member, no matter the cluster size)
- look for the parent cluster within a minimum set of candidates
  (inspecting only new clusters plus the known neighbors who got updated)
- allow choosing either the first suitable parent cluster, or considering only the best paternal match
*/
struct TTSAS_Clustering : ClusterAlg {
	static const std::stringType Name;	///< name of TTSAS algorithm from varConfig.txt

	/**
	Performs clustering of a set of symbols.

	@param symsToGroup symbols to be grouped by similarity
	@param symsIndicesPerCluster returned vector of clusters, each cluster with the indices towards member symbols
	@param fontType font family, style and encoding (not the size); empty for various unit tests

	@return number of clusters obtained
	*/
	unsigned formGroups(const VSymData &symsToGroup,
						std::vector<std::vector<unsigned>> &symsIndicesPerCluster,
						const std::stringType &fontType = "") override;
};

#endif // H_TTSAS_CLUSTERING
