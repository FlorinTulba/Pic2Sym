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

#include "clusterSupport.h"
#include "clusterEngine.h"
#include "symData.h"
#include "misc.h"

#pragma warning ( push, 0 )

#include <numeric>

#pragma warning ( pop )

using namespace std;

ClustersSupport::ClustersSupport(ClusterEngine &ce_, SymsSupport &ss_, VSymData &symsSet_) :
	ce(ce_), ss(ss_), symsSet(symsSet_) {}

void ClustersSupport::groupSyms(const string &fontType/* = ""*/) {
	// Clustering on symsSet
	ce.process(symsSet, fontType);
}

void ClustersSupport::delimitGroups(vector<vector<unsigned>> &symsIndicesPerCluster,
									VClusterData &clusters, set<unsigned> &clusterOffsets) {
	const auto symsCount = symsSet.size();
	VSymData newSymsSet;
	newSymsSet.reserve(symsCount);

	for(unsigned i = 0U, offset = 0U, lim = ce.getClustersCount(); i<lim; ++i) {
		auto &symsIndices = symsIndicesPerCluster[i];
		const unsigned clusterSz = (unsigned)symsIndices.size();
		clusterOffsets.emplace_hint(end(clusterOffsets), offset);
		clusters.emplace_back(symsSet, offset, symsIndices, ss); // needs symsSet[symsIndices] !!

		for(const auto idx : symsIndices) {
			// Don't use move for symsSet[idx], as the symbols need to remain in symsSet for later examination
			newSymsSet.push_back(symsSet[idx]);
		}

		iota(BOUNDS(symsIndices), offset); // new pointers will be consecutive

		offset += clusterSz;
	}
	clusterOffsets.emplace_hint(end(clusterOffsets), (unsigned)symsCount); // delimit last cluster

	symsSet = move(newSymsSet);
}

const VSymData& ClustersSupport::clusteredSyms() const {
	return symsSet;
}