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

#include "partitionClustering.h"
#include "clusterEngine.h"
#include "clusterSerialization.h"
#include "taskMonitor.h"
#include "tinySym.h"
#include "tinySymsProvider.h"
#include "misc.h"

#pragma warning ( push, 0 )

#include <iostream>

#include <opencv2/imgproc/imgproc.hpp>

#pragma warning ( pop )

using namespace std;
using namespace cv;

extern const bool FastDistSymToClusterComputation;
extern const double MaxAvgProjErrForPartitionClustering;
extern const double MaxRelMcOffsetForPartitionClustering;
extern const double MaxDiffAvgPixelValForPartitionClustering;

const string PartitionClustering::Name("Partition");

unsigned PartitionClustering::formGroups(const VSymData &symsToGroup,
										 vector<vector<unsigned>> &symsIndicesPerCluster,
										 const string &fontType/* = ""*/) {
#pragma warning ( disable : WARN_THREAD_UNSAFE )
	static TaskMonitor partitionClustering("partition clustering", *symsMonitor);
#pragma warning ( default : WARN_THREAD_UNSAFE )

	boost::filesystem::path clusteredSetFile;
	ClusterIO rawClusters;
	if(!ClusterEngine::clusteredAlready(fontType, Name, clusteredSetFile)
		   || !rawClusters.loadFrom(clusteredSetFile.string())) {
		if(tsp == nullptr)
			THROW_WITH_CONST_MSG(__FUNCTION__ " should be called only after calling setTinySymsProvider()!", logic_error);

		const auto &tinySyms = tsp->getTinySyms();
#pragma warning ( disable : WARN_THREAD_UNSAFE )
		static const double SqMaxRelMcOffsetForClustering =
						MaxRelMcOffsetForPartitionClustering * MaxRelMcOffsetForPartitionClustering;
#pragma warning ( default : WARN_THREAD_UNSAFE )

#if defined _DEBUG && !defined UNIT_TESTING
		unsigned countAvgPixDiff = 0U, countMcsOffset = 0U, countDiff = 0U,
				countHdiff = 0U, countVdiff = 0U, countBslashDiff = 0U, countSlashDiff = 0U;
#endif

		const unsigned tinySymsCount = (unsigned)tinySyms.size();
		rawClusters.clusterLabels.resize(tinySymsCount, -1);
		rawClusters.clustersCount = (unsigned)partition(tinySyms, rawClusters.clusterLabels,
											[&] (const TinySym &a, const TinySym &b) {
#if !defined _DEBUG || defined UNIT_TESTING
			if(!FastDistSymToClusterComputation) {
				const double l1Dist = norm(a.mat - b.mat, NORM_L1);
				return l1Dist <= MaxAvgProjErrForPartitionClustering;
			}

			if(abs(a.avgPixVal - b.avgPixVal) > MaxDiffAvgPixelValForPartitionClustering)
				return false;
			const Point2d mcDelta = a.mc - b.mc;
			const double mcDeltaY = abs(mcDelta.y);
			if(mcDeltaY > MaxRelMcOffsetForPartitionClustering)
				return false;
			const double mcDeltaX = abs(mcDelta.x);
			if(mcDeltaX > MaxRelMcOffsetForPartitionClustering)
				return false;
			if(mcDeltaX*mcDeltaX + mcDeltaY*mcDeltaY > SqMaxRelMcOffsetForClustering)
				return false;

			const double *pDataA, *pDataAEnd, *pDataB;

#define CheckProjections(ProjectionField) \
			pDataA = reinterpret_cast<const double*>(a.ProjectionField.datastart); \
			pDataAEnd = reinterpret_cast<const double*>(a.ProjectionField.datalimit); \
			pDataB = reinterpret_cast<const double*>(b.ProjectionField.datastart); \
			for(double sumOfAbsDiffs = 0.; pDataA != pDataAEnd;) { \
				sumOfAbsDiffs += abs(*pDataA++ - *pDataB++); \
				if(sumOfAbsDiffs > MaxAvgProjErrForPartitionClustering) \
					return false; \
			}

			CheckProjections(vAvgProj);
			CheckProjections(hAvgProj);
			CheckProjections(backslashDiagAvgProj);
			CheckProjections(slashDiagAvgProj);

#undef CheckProjections

			auto itA = a.mat.begin<double>(), itAEnd = a.mat.end<double>(),
				itB = b.mat.begin<double>();
			for(double sumOfAbsDiffs = 0.; itA != itAEnd;) {
				sumOfAbsDiffs += abs(*itA++ - *itB++);
				if(sumOfAbsDiffs > MaxAvgProjErrForPartitionClustering)
					return false;
			}

			return true;

#else // DEBUG mode and UNIT_TESTING is not defined
			if(!FastDistSymToClusterComputation) {
				const double l1Dist = norm(a.mat - b.mat, NORM_L1);
				return l1Dist <= MaxAvgProjErrForPartitionClustering;
			}

			const double avgPixDiff = abs(a.avgPixVal - b.avgPixVal);
			const bool bAvgPixDiff = avgPixDiff > MaxDiffAvgPixelValForPartitionClustering;
			if(bAvgPixDiff) {
				++countAvgPixDiff;
				return false;
			}

			const double mcsOffset = norm(a.mc - b.mc);
			const bool bMcsOffset = mcsOffset > MaxRelMcOffsetForPartitionClustering;
			if(bMcsOffset) {
				++countMcsOffset;
				return false;
			}

#define CheckDifferences(Field, WorkloadReductionQuota) \
			{ \
				const double l1Norm = norm(a.Field - b.Field, NORM_L1); \
				const bool contributesToWorkloadReduction = l1Norm > MaxAvgProjErrForPartitionClustering; \
				if(contributesToWorkloadReduction) { \
					++WorkloadReductionQuota; \
					return false; \
				} \
			}

			CheckDifferences(vAvgProj, countVdiff);
			CheckDifferences(hAvgProj, countHdiff);
			CheckDifferences(backslashDiagAvgProj, countBslashDiff);
			CheckDifferences(slashDiagAvgProj, countSlashDiff);
			CheckDifferences(mat, countDiff);

#undef CheckDifferences

			return true;
#endif // DEBUG, UNIT_TESTING
		});
		cout<<endl<<"All the "<<tinySymsCount<<" symbols of the charmap were clustered in "
			<<rawClusters.clustersCount<<" groups"<<endl;

#if defined _DEBUG && !defined UNIT_TESTING
		PRINTLN(countAvgPixDiff);
		PRINTLN(countMcsOffset);
		PRINTLN(countVdiff);
		PRINTLN(countHdiff);
		PRINTLN(countBslashDiff);
		PRINTLN(countSlashDiff);
		PRINTLN(countDiff);
#endif // DEBUG, UNIT_TESTING
		
		rawClusters.saveTo(clusteredSetFile.string());
	}

	// Adapt clusters for filtered cmap
	symsIndicesPerCluster.assign(rawClusters.clustersCount, vector<unsigned>());
	for(unsigned i = 0U, lim = (unsigned)symsToGroup.size(); i < lim; ++i)
		symsIndicesPerCluster[(size_t)rawClusters.clusterLabels[symsToGroup[(size_t)i].symIdx]].push_back(i);
	const auto newEndIt = remove_if(BOUNDS(symsIndicesPerCluster),
							   [] (const vector<unsigned> &elem) { return elem.empty(); });
	symsIndicesPerCluster.resize((size_t)distance(symsIndicesPerCluster.begin(), newEndIt));

	partitionClustering.taskDone();

	return (unsigned)symsIndicesPerCluster.size();
}