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

#include "clusterEngine.h"
#include "clusterSupport.h"
#include "symbolsSupport.h"
#include "jobMonitorBase.h"
#include "taskMonitor.h"
#include "misc.h"

#ifndef UNIT_TESTING

#include "appStart.h"

#pragma warning ( push, 0 )

#include <boost/filesystem/operations.hpp>

#pragma warning ( pop )

#endif // UNIT_TESTING not defined

#pragma warning ( push, 0 )

#include <set>
#include <iostream>
#include <numeric>

#include <opencv2/imgproc/imgproc.hpp>

#pragma warning ( pop )

using namespace std;
using namespace cv;

extern const string ClusterAlgName;
extern const double MinAverageClusterSize;

/// Reports various details about the identified clusters
static void reportClustersInfo(const vector<vector<unsigned>> &symsIndicesPerCluster,
							   unsigned clustersCount, const VSymData &symsSet) {
	const auto maxClusterSz = max_element(CBOUNDS(symsIndicesPerCluster),
										  [] (const vector<unsigned> &a, const vector<unsigned> &b) {
		return a.size() < b.size();
	})->size();
	const auto nonTrivialClusters = (unsigned)count_if(CBOUNDS(symsIndicesPerCluster),
													   [] (const vector<unsigned> &a) {
		return a.size() > 1ULL;
	});
	const auto clusteredSyms = (unsigned)symsSet.size() - (clustersCount - nonTrivialClusters);

	cout<<"There are "<<nonTrivialClusters
		<<" non-trivial clusters that hold a total of "<<clusteredSyms<<" symbols."<<endl;
	cout<<"Largest cluster contains "<<maxClusterSz<<" symbols."<<endl;
}

ClusterData::ClusterData(const VSymData &symsSet, unsigned idxOfFirstSym_,
						 const vector<unsigned> &clusterSymIndices,
						 SymsSupport &symsSupport) : SymData(),
		idxOfFirstSym(idxOfFirstSym_), sz((unsigned)clusterSymIndices.size()) {
	assert(!clusterSymIndices.empty() && !symsSet.empty());
	const double invClusterSz = 1./sz;
	const Mat &firstNegSym = symsSet[0].negSym;
	const int symSz = firstNegSym.rows;
	double avgPixVal_ = 0.;
	Point2d mc_;
	vector<const SymData*> clusterSyms; clusterSyms.reserve((size_t)sz);

	for(const auto clusterSymIdx : clusterSymIndices) {
		const SymData &symData = symsSet[clusterSymIdx];
		clusterSyms.push_back(&symData);

		// avgPixVal and mc are taken from the normal-size symbol (guaranteed to be non-blank)
		avgPixVal_ += symData.avgPixVal;
		mc_ += symData.mc;
	}
	avgPixVal = avgPixVal_ * invClusterSz;
	mc = mc_ * invClusterSz;

	Mat synthesizedSym;
	symsSupport.computeClusterRepresentative(clusterSyms, symSz, invClusterSz, synthesizedSym, negSym);

	computeFields(synthesizedSym, masks[FG_MASK_IDX], masks[BG_MASK_IDX], masks[EDGE_MASK_IDX],
				  masks[GROUNDED_SYM_IDX], masks[BLURRED_GR_SYM_IDX], masks[VARIANCE_GR_SYM_IDX],
				  minVal, diffMinMax, symsSupport.usingTinySymbols());
}

ClusterData::ClusterData(ClusterData &&other) : SymData(move(other)),
	idxOfFirstSym(other.idxOfFirstSym), sz(other.sz) {}

ClusterEngine::ClusterEngine(ITinySymsProvider &tsp_) :
	clustAlg(ClusterAlg::algByName(ClusterAlgName).setTinySymsProvider(tsp_)) {}

void ClusterEngine::process(VSymData &symsSet, const string &fontType/* = ""*/) {
	if(symsSet.empty())
		return;

	clusters.clear(); clusterOffsets.clear(); symsIndicesPerCluster.clear();

	clustersCount = clustAlg.formGroups(symsSet, symsIndicesPerCluster, fontType);

	const double averageClusterSize = (double)symsSet.size() / clustersCount;
	cout<<"Average cluster size is "<<averageClusterSize<<endl;

#pragma warning ( disable : WARN_THREAD_UNSAFE )
	static TaskMonitor reorderClusters("reorders clusters", *symsMonitor);
#pragma warning ( default : WARN_THREAD_UNSAFE )

	// Sort symsIndicesPerCluster in ascending order of avgPixVal taken from the first symbol from each cluster
	sort(BOUNDS(symsIndicesPerCluster),
		 [&] (const vector<unsigned> &a, const vector<unsigned> &b) {
		return symsSet[(size_t)a.front()].avgPixVal < symsSet[(size_t)b.front()].avgPixVal;
	});

	if(averageClusterSize > MinAverageClusterSize) {
		worthy = true;
		reportClustersInfo(symsIndicesPerCluster, clustersCount, symsSet);

		// Trivial clusters are already in ascending order of avgPixVal (see the sort from above)

		support->delimitGroups(symsIndicesPerCluster, clusters, clusterOffsets);

	} else {
		worthy = false;
		cout<<"Ignoring clustering due to the low average cluster size."<<endl;
	}

	reorderClusters.taskDone(); // mark it as already finished
}

const VClusterData& ClusterEngine::getClusters() const {
	assert(worthy); // when worthGrouping returns false, one shouldn't need clusters, nor rely on their value
	return clusters;
}

const set<unsigned>& ClusterEngine::getClusterOffsets() const {
	assert(worthy); // when worthGrouping returns false, one shouldn't need clusterOffsets, nor rely on their value
	return clusterOffsets;
}

ClusterEngine& ClusterEngine::useSymsMonitor(AbsJobMonitor &symsMonitor_) {
	symsMonitor = &symsMonitor_;
	clustAlg.useSymsMonitor(symsMonitor_);
	return *this;
}

ClusterEngine& ClusterEngine::supportedBy(ClustersSupport &support_) {
	support = &support_;
	return *this;
}

#ifndef UNIT_TESTING

using namespace boost::filesystem;

bool ClusterEngine::clusteredAlready(const string &fontType, const string &algName, path &clusteredSetFile) {
	if(fontType.empty())
		return false;

	clusteredSetFile = AppStart::dir();
	if(!exists(clusteredSetFile.append("ClusteredSets")))
		create_directory(clusteredSetFile);

	clusteredSetFile.append(fontType).concat("_").concat(algName).
		concat(".clf"); // CLustered Fonts => clf

	return exists(clusteredSetFile);
}

#endif // UNIT_TESTING not defined
