/****************************************************************************************
 The application Pic2Sym approximates images by a
 grid of colored symbols with colored backgrounds.

 This file belongs to the Pic2Sym project.

 Copyrights from the libraries used by the program:
 - (c) 2015 Boost (www.boost.org)
   License: <http://www.boost.org/LICENSE_1_0.txt>
            or doc/licenses/Boost.lic
 - (c) 2015 The FreeType Project (www.freetype.org)
   License: <http://git.savannah.gnu.org/cgit/freetype/freetype2.git/plain/docs/FTL.TXT>
	        or doc/licenses/FTL.txt
 - (c) 2015 OpenCV (www.opencv.org)
   License: <http://opencv.org/license.html>
            or doc/licenses/OpenCV.lic
 - (c) 1997-2002 OpenMP Architecture Review Board (www.openmp.org)
   (c) Microsoft Corporation (Visual C++ implementation for OpenMP C/C++ Version 2.0 March 2002)
   See: <https://msdn.microsoft.com/en-us/library/8y6825x5(v=vs.140).aspx>
 
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
 ****************************************************************************************/

#include "clusterEngine.h"
#include "jobMonitorBase.h"
#include "taskMonitor.h"
#include "misc.h"

#ifndef UNIT_TESTING
#include "appStart.h"
#endif // UNIT_TESTING

#include <set>
#include <iostream>
#include <numeric>

#include <opencv2/imgproc/imgproc.hpp>
#include <boost/filesystem/operations.hpp>

using namespace std;
using namespace cv;
using namespace boost::filesystem;

extern const string ClusterAlgName;
extern unsigned TinySymsSz();

namespace {
	const double INV255 = 1./255.;

	/**
	Reorders the clusters by their size - largest ones first and collects the offsets of each cluster.
	
	When tinySymsSet isn't empty (when using tiny symbols preselection),
	the actual returned clusters will contain the tiny symbols
	*/
	void computeClusterOffsets(vector<vector<unsigned>> &symsIndicesPerCluster, unsigned clustersCount,
							   VSymData &symsSet, VSymData &tinySymsSet,
							   VClusterData &clusters, set<unsigned> &clusterOffsets) {
		const bool usingTinySyms = !tinySymsSet.empty();

		// Add the clusters in descending order of their size:

		// Typically, there are only a few clusters larger than 1 element.
		// This partition separates the actual formed clusters from one-of-a-kind elements
		// leaving less work to perform to the sort executed afterwards
		auto itFirstClusterWithOneItem = partition(BOUNDS(symsIndicesPerCluster),
														  [] (const vector<unsigned> &a) {
			return a.size() > 1U; // place actual clusters at the beginning of the vector
		});

		// Sort non-trivial clusters in descending order of their size
		// and then in ascending order of avgPixVal (taken from last cluster member)
		sort(begin(symsIndicesPerCluster), itFirstClusterWithOneItem,
					[&] (const vector<unsigned> &a, const vector<unsigned> &b) {
			const size_t szA = a.size(), szB = b.size();
			return (szA > szB) || ((szA == szB) && (symsSet[a.back()].avgPixVal < symsSet[b.back()].avgPixVal));
		});

		// Sort trivial clusters in ascending order of avgPixVal
		sort(itFirstClusterWithOneItem, end(symsIndicesPerCluster),
			 [&] (const vector<unsigned> &a, const vector<unsigned> &b) {
			return symsSet[a.back()].avgPixVal < symsSet[b.back()].avgPixVal;
		});

		const unsigned nonTrivialClusters = (unsigned)distance(begin(symsIndicesPerCluster), itFirstClusterWithOneItem),
					symsCount = (unsigned)symsSet.size(),
					clusteredSyms = symsCount - ((unsigned)symsIndicesPerCluster.size() - nonTrivialClusters);
		cout<<"There are "<<nonTrivialClusters
			<<" non-trivial clusters that hold a total of "<<clusteredSyms<<" symbols."<<endl;
		cout<<"Largest cluster contains "<<symsIndicesPerCluster[0].size()<<" symbols"<<endl;

		VSymData newSymsSet, newTinySymsSet;
		newSymsSet.reserve(symsCount);
		if(usingTinySyms)
			newTinySymsSet.reserve(symsCount);

		VSymData &sourceSet = (usingTinySyms ? tinySymsSet : symsSet);
		for(unsigned i = 0U, offset = 0U; i<clustersCount; ++i) {
			const auto &symsIndices = symsIndicesPerCluster[i];
			const unsigned clusterSz = (unsigned)symsIndices.size();
			clusterOffsets.insert(offset);
			clusters.emplace_back(sourceSet, offset, symsIndices, usingTinySyms); // needs sourceSet[symsIndices] !!

			for(const auto idx : symsIndices) {
				// Don't use move for symsSet[idx], as the symbols need to remain in symsSet for later examination
				newSymsSet.push_back(symsSet[idx]);

				if(usingTinySyms)
					// Don't use move for tinySymsSet[idx], as the symbols need to remain in tinySymsSet for later examination
					newTinySymsSet.push_back(tinySymsSet[idx]);
			}

			offset += clusterSz;
		}
		clusterOffsets.insert(symsCount); // delimit last cluster

		symsSet = move(newSymsSet);
		if(usingTinySyms)
			tinySymsSet = move(newTinySymsSet);
	}
} // anonymous namespace

ClusterData::ClusterData(const VSymData &symsSet, unsigned idxOfFirstSym_,
						 const vector<unsigned> &clusterSymIndices,
						 bool forTinySyms) : SymData(),
		idxOfFirstSym(idxOfFirstSym_), sz((unsigned)clusterSymIndices.size()) {
	assert(!clusterSymIndices.empty() && !symsSet.empty());
	const Mat &firstNegSym = symsSet[0].negSym;
	const int symSz = firstNegSym.rows;
	double avgPixVal_ = 0.;
	Point2d mc_;
	Mat synthesizedSym, negSynthesizedSym(symSz, symSz, CV_64FC1, Scalar(0.));

	if(forTinySyms) { // tiny symbols have negSym of type double already
		for(const auto clusterSymIdx : clusterSymIndices) {
			const SymData &symData = symsSet[clusterSymIdx];
			negSynthesizedSym += symData.negSym;
			avgPixVal_ += symData.avgPixVal;
			mc_ += symData.mc;
		}

	} else { // normal symbols need to convert their negSym from byte to double when averaging
		for(const auto clusterSymIdx : clusterSymIndices) {
			const SymData &symData = symsSet[clusterSymIdx];
			Mat negSymD;
			symData.negSym.convertTo(negSymD, CV_64FC1);
			negSynthesizedSym += negSymD;
			avgPixVal_ += symData.avgPixVal;
			mc_ += symData.mc;
		}
	}
	const double invClusterSz = 1./sz;
	negSynthesizedSym *= invClusterSz;
	if(forTinySyms) // cluster representatives for tiny symbols have negSym of type double
		negSym = negSynthesizedSym;
	else // cluster representatives for normal symbols have negSym of type byte
		negSynthesizedSym.convertTo(negSym, CV_8UC1);

	synthesizedSym = 1. - negSynthesizedSym * INV255; // providing a symbol in 0..1 range
	computeFields(synthesizedSym, masks[FG_MASK_IDX], masks[BG_MASK_IDX], masks[EDGE_MASK_IDX],
				  masks[GROUNDED_SYM_IDX], masks[BLURRED_GR_SYM_IDX], masks[VARIANCE_GR_SYM_IDX],
				  minVal, diffMinMax, forTinySyms);

	avgPixVal	= avgPixVal_ * invClusterSz;
	mc			= mc_ * invClusterSz;
}

ClusterData::ClusterData(ClusterData &&other) : SymData(move(other)),
	idxOfFirstSym(other.idxOfFirstSym), sz(other.sz) {}

ClusterEngine::ClusterEngine(ITinySymsProvider &tsp_) :
	clustAlg(ClusterAlg::algByName(ClusterAlgName).setTinySymsProvider(tsp_)) {}

void ClusterEngine::process(VSymData &symsSet, VSymData &tinySymsSet, const string &fontType/* = ""*/) {
	if(symsSet.empty())
		return;

	vector<vector<unsigned>> symsIndicesPerCluster;
	const unsigned clustersCount = clustAlg.formGroups(symsSet, symsIndicesPerCluster, fontType);

	static TaskMonitor reorderClusters("reorders clusters", *symsMonitor);
	clusters.clear(); clusterOffsets.clear();
	computeClusterOffsets(symsIndicesPerCluster, clustersCount, symsSet, tinySymsSet, clusters, clusterOffsets);
	reorderClusters.taskDone(); // mark it as already finished
}

ClusterEngine& ClusterEngine::useSymsMonitor(AbsJobMonitor &symsMonitor_) {
	symsMonitor = &symsMonitor_;
	clustAlg.useSymsMonitor(symsMonitor_);
	return *this;
}

#ifndef UNIT_TESTING

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

#endif // UNIT_TESTING
