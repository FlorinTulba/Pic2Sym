/****************************************************************************************
 The application Pic2Sym approximates images by a
 grid of colored symbols with colored backgrounds.

 This file belongs to the UnitTesting project.

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

#include "testMain.h"
#include "misc.h"
#include "clusterEngine.h"
#include "noClustering.h"
#include "partitionClustering.h"
#include "ttsasClustering.h"
#include "jobMonitor.h"

using namespace std;
using namespace cv;

extern const string ClusterAlgName;
extern const bool FastDistSymToClusterComputation;
extern const double MaxAvgProjErrForPartitionClustering;
extern const double MaxRelMcOffsetForPartitionClustering;
extern const double MaxDiffAvgPixelValForPartitionClustering;
extern const bool TTSAS_Accept1stClusterThatQualifiesAsParent;
extern const double TTSAS_Threshold_Member;
extern const double MaxRelMcOffsetForTTSAS_Clustering;
extern const double MaxDiffAvgPixelValForTTSAS_Clustering;

namespace ut {
	/// Provides a way to use specific clustering settings during each test and revert them afterwards
	class ClusteringSettingsFixt : public Fixt {
	protected:
		JobMonitor jm;

#define DefineCopyAndRef(Setting, Type) \
		Type orig##Setting = Setting, &ref##Setting = const_cast<Type&>(Setting)

		DefineCopyAndRef(ClusterAlgName, string);
		DefineCopyAndRef(FastDistSymToClusterComputation, bool);
		DefineCopyAndRef(TTSAS_Accept1stClusterThatQualifiesAsParent, bool);
		DefineCopyAndRef(MaxAvgProjErrForPartitionClustering, double);
		DefineCopyAndRef(MaxRelMcOffsetForPartitionClustering, double);
		DefineCopyAndRef(MaxDiffAvgPixelValForPartitionClustering, double);
		DefineCopyAndRef(TTSAS_Threshold_Member, double);
		DefineCopyAndRef(MaxRelMcOffsetForTTSAS_Clustering, double);
		DefineCopyAndRef(MaxDiffAvgPixelValForTTSAS_Clustering, double);

#undef DefineCopyAndRef

	public:
		~ClusteringSettingsFixt() {
#define RevertIfChanged(Setting) \
			if(Setting != orig##Setting) \
				ref##Setting = orig##Setting

			RevertIfChanged(ClusterAlgName);
			RevertIfChanged(FastDistSymToClusterComputation);
			RevertIfChanged(TTSAS_Accept1stClusterThatQualifiesAsParent);
			RevertIfChanged(MaxAvgProjErrForPartitionClustering);
			RevertIfChanged(MaxRelMcOffsetForPartitionClustering);
			RevertIfChanged(MaxDiffAvgPixelValForPartitionClustering);
			RevertIfChanged(MaxDiffAvgPixelValForPartitionClustering);
			RevertIfChanged(TTSAS_Threshold_Member);
			RevertIfChanged(MaxRelMcOffsetForTTSAS_Clustering);
			RevertIfChanged(MaxDiffAvgPixelValForTTSAS_Clustering);

#undef RevertIfChanged
		}
	};

	/// Hack to be able to refer protected clustAlg field from ClusterEngine
	struct TestClusterEngine : public ClusterEngine {
		//using ClusterEngine::clustAlg; // checkAlgType from below does the job without changing field visibility

		template<class AlgType>
		bool checkAlgType() const {
			return string(typeid(AlgType).name()).
				compare(typeid(clustAlg).name()) == 0;
		}
	};

	const double TinySymArea = double(TinySymData::TinySymSz*TinySymData::TinySymSz);
	const unsigned TinySymMidSide = (unsigned)TinySymData::TinySymSz>>1,
				TinySymDiagsCount = ((unsigned)TinySymData::TinySymSz<<1) - 1U;
	const Point2d TinySymCenter(TinySymMidSide, TinySymMidSide);

	const TinySymData EmptyTinySym(TinySymCenter, 0., Mat(TinySymData::TinySymSz, TinySymData::TinySymSz, CV_64FC1, Scalar(0.)),
								   Mat(1, TinySymData::TinySymSz, CV_64FC1, Scalar(0.)), Mat(1, TinySymData::TinySymSz, CV_64FC1, Scalar(0.)),
								   Mat(1, TinySymDiagsCount, CV_64FC1, Scalar(0.)), Mat(1, TinySymDiagsCount, CV_64FC1, Scalar(0.)));

	const SymData EmptySymData5x5(0UL, 0., 0., 0., TinySymCenter,
		{ { SymData::NEG_SYM_IDX, Mat(TinySymData::TinySymSz, TinySymData::TinySymSz, CV_8UC1, Scalar(255U)) },
		{ SymData::GROUNDED_SYM_IDX, Mat(TinySymData::TinySymSz, TinySymData::TinySymSz, CV_64FC1, Scalar(0.)) } }),
				MainDiagSymData5x5(1UL, 0., 1., (double)TinySymData::TinySymSz, TinySymCenter,
		{ { SymData::NEG_SYM_IDX, (255U - Mat::eye(TinySymData::TinySymSz, TinySymData::TinySymSz, CV_8UC1) * 255U) },
		{ SymData::GROUNDED_SYM_IDX, Mat::eye(TinySymData::TinySymSz, TinySymData::TinySymSz, CV_64FC1) } });

	void updateCentralPixel(TinySymData &sym, double pixelVal) {
		const double oldPixelVal = sym.mat.at<double>(TinySymMidSide, TinySymMidSide),
					oldPixSum = sym.avgPixVal * TinySymArea,
					diff = pixelVal - oldPixelVal,
					avgDiff = diff / TinySymData::TinySymSz;

		sym.mc = (oldPixSum*sym.mc + diff*TinySymCenter) / (oldPixSum + diff);
		
		sym.avgPixVal += diff / TinySymArea;
		
		// Clone all matrices from sym, to ensure that the central pixel
		// won't be changed in the template provided as the sym parameter
		sym.mat = sym.mat.clone();
		sym.mat.at<double>(TinySymMidSide, TinySymMidSide) = pixelVal;

#define UpdateProjectionAt(Field, At) \
		sym.Field = sym.Field.clone(); \
		sym.Field.at<double>(At) += avgDiff

		UpdateProjectionAt(hAvgProj, TinySymMidSide);
		UpdateProjectionAt(vAvgProj, TinySymMidSide);
		UpdateProjectionAt(backslashDiagAvgProj, TinySymData::TinySymSz-1);
		UpdateProjectionAt(slashDiagAvgProj, TinySymData::TinySymSz-1);

#undef UpdateProjectionAt
	};
}

using namespace ut;

BOOST_FIXTURE_TEST_SUITE(ClusterEngineCreation_Tests, ClusteringSettingsFixt)
	BOOST_AUTO_TEST_CASE(ClusterEngineCreation_InvalidAlgName_UsingNoClustering) {
		BOOST_TEST_MESSAGE("Running ClusterEngineCreation_InvalidAlgName_UsingNoClustering");
		refClusterAlgName = "Invalid_Algorithm_Name!!!!";
		TestClusterEngine ce;
		BOOST_REQUIRE(ce.checkAlgType<NoClustering>());
	}

	BOOST_AUTO_TEST_CASE(ClusterEngineCreation_NoClusteringRequest_UsingNoClustering) {
		BOOST_TEST_MESSAGE("Running ClusterEngineCreation_NoClusteringRequest_UsingNoClustering");
		refClusterAlgName = "None";
		TestClusterEngine ce;
		BOOST_REQUIRE(ce.checkAlgType<NoClustering>());
	}

	BOOST_AUTO_TEST_CASE(ClusterEngineCreation_PartitionClusteringRequest_UsingPartitionClustering) {
		BOOST_TEST_MESSAGE("Running ClusterEngineCreation_PartitionClusteringRequest_UsingPartitionClustering");
		refClusterAlgName = "Partition";
		TestClusterEngine ce;
		BOOST_REQUIRE(ce.checkAlgType<PartitionClustering>());
	}

	BOOST_AUTO_TEST_CASE(ClusterEngineCreation_TTSASClusteringRequest_UsingTTSASClustering) {
		BOOST_TEST_MESSAGE("Running ClusterEngineCreation_TTSASClusteringRequest_UsingTTSASClustering");
		refClusterAlgName = "TTSAS";
		TestClusterEngine ce;
		BOOST_REQUIRE(ce.checkAlgType<TTSAS_Clustering>());
	}

BOOST_AUTO_TEST_SUITE_END() // ClusterEngineCreation

BOOST_FIXTURE_TEST_SUITE(BasicClustering_Tests, ClusteringSettingsFixt)
	BOOST_AUTO_TEST_CASE(UsingNoClustering_6identicalSymbols_0nonTrivialClusters) {
		BOOST_TEST_MESSAGE("Running UsingNoClustering_6identicalSymbols_0nonTrivialClusters");
		refClusterAlgName = "None";
		ClusterEngine ce;
		ce.useSymsMonitor(jm);
		size_t symsCount = 6U;
		VSymData symsSet(symsCount, EmptySymData5x5);
		ce.process(symsSet);
		BOOST_REQUIRE(ce.getClusters().size() == symsCount);
	}

	BOOST_AUTO_TEST_CASE(UsingPartitionClustering_6identicalSymbols_noTrivialClusters) {
		BOOST_TEST_MESSAGE("Running UsingPartitionClustering_6identicalSymbols_noTrivialClusters");
		refClusterAlgName = "Partition";
		ClusterEngine ce;
		ce.useSymsMonitor(jm);
		const size_t symsCount = 6U;
		VSymData symsSet(symsCount, EmptySymData5x5);
		ce.process(symsSet);
		BOOST_REQUIRE(ce.getClusters().size() == 1U);
	}

	BOOST_AUTO_TEST_CASE(UsingTTSASclustering_6identicalSymbols_noTrivialClusters) {
		BOOST_TEST_MESSAGE("Running UsingTTSASclustering_6identicalSymbols_noTrivialClusters");
		refClusterAlgName = "TTSAS";
		ClusterEngine ce;
		ce.useSymsMonitor(jm);
		const size_t symsCount = 6U;
		VSymData symsSet(symsCount, EmptySymData5x5);
		ce.process(symsSet);
		BOOST_REQUIRE(ce.getClusters().size() == 1U);
	}

	BOOST_AUTO_TEST_CASE(UsingPartitionClustering_x0x0xSequence_2Clusters) {
		BOOST_TEST_MESSAGE("Running UsingPartitionClustering_x0x0xSequence_2Clusters");
		refClusterAlgName = "Partition";
		ClusterEngine ce;
		ce.useSymsMonitor(jm);
		VSymData symsSet { MainDiagSymData5x5, EmptySymData5x5, MainDiagSymData5x5, EmptySymData5x5, MainDiagSymData5x5 };
		const size_t symsCount = symsSet.size();
		ce.process(symsSet);
		const auto &clusterOffsets = ce.getClusterOffsets();
		const auto &clusters = ce.getClusters();
		BOOST_REQUIRE(clusters.size() == 2U);
		BOOST_REQUIRE(clusters[0].idxOfFirstSym == 0 && clusters[0].sz == 3); // largest cluster starts at 0 (3 items)
		BOOST_REQUIRE(clusters[1].idxOfFirstSym == 3 && clusters[1].sz == 2); // smallest cluster starts at 3 (2 items)
		BOOST_REQUIRE(clusterOffsets.size() == 3U);
		BOOST_REQUIRE(clusterOffsets.find(0U) != clusterOffsets.end()); // largest cluster starts at 0 (3 items)
		BOOST_REQUIRE(clusterOffsets.find(3U) != clusterOffsets.end()); // smallest cluster starts at 3 (2 items)
		BOOST_REQUIRE(clusterOffsets.find((unsigned)symsCount) != clusterOffsets.end());
	}

	BOOST_AUTO_TEST_CASE(UsingTTSASclustering_x0x0xSequence_2Clusters) {
		BOOST_TEST_MESSAGE("Running UsingTTSASclustering_x0x0xSequence_2Clusters");
		refClusterAlgName = "TTSAS";
		ClusterEngine ce;
		ce.useSymsMonitor(jm);
		VSymData symsSet { MainDiagSymData5x5, EmptySymData5x5, MainDiagSymData5x5, EmptySymData5x5, MainDiagSymData5x5 };
		const size_t symsCount = symsSet.size();
		ce.process(symsSet);
		const auto &clusterOffsets = ce.getClusterOffsets();
		const auto &clusters = ce.getClusters();
		BOOST_REQUIRE(clusters.size() == 2U);
		BOOST_REQUIRE(clusters[0].idxOfFirstSym == 0 && clusters[0].sz == 3); // largest cluster starts at 0 (3 items)
		BOOST_REQUIRE(clusters[1].idxOfFirstSym == 3 && clusters[1].sz == 2); // smallest cluster starts at 3 (2 items)
		BOOST_REQUIRE(clusterOffsets.size() == 3U);
		BOOST_REQUIRE(clusterOffsets.find(0U) != clusterOffsets.end()); // largest cluster starts at 0 (3 items)
		BOOST_REQUIRE(clusterOffsets.find(3U) != clusterOffsets.end()); // smallest cluster starts at 3 (2 items)
		BOOST_REQUIRE(clusterOffsets.find((unsigned)symsCount) != clusterOffsets.end());
	}

BOOST_AUTO_TEST_SUITE_END() // BasicClustering_Tests

BOOST_FIXTURE_TEST_SUITE(TTSAS_Clustering_Tests, ClusteringSettingsFixt)
	BOOST_AUTO_TEST_CASE(CheckMemberThresholdTTSAS_BelowOrAboveThreshold_MemberOrNot) {
		BOOST_TEST_MESSAGE("Running CheckMemberThresholdTTSAS_BelowOrAboveThreshold_MemberOrNot");
		TTSAS_Clustering tc;
		tc.useSymsMonitor(jm);
		vector<const TinySymData> smallSyms;
		vector<vector<unsigned>> symsIndicesPerCluster;

		for(unsigned n = 1U; n < 30U; ++n) {
			// Use n identical symbols => a single cluster
			BOOST_TEST_MESSAGE("Checking member thresholds with a set of " + to_string(n) + " identical symbols");

			smallSyms.assign(n, EmptyTinySym);
			BOOST_REQUIRE(1U == tc.formGroups(smallSyms, symsIndicesPerCluster));

			const double ThresholdForNplus1 = TTSAS_Threshold_Member / (n+1U);

			// Append a symbol beyond the threshold => 2 clusters
			TinySymData newSym = EmptyTinySym; // changing only central pixel => same mass-center
			updateCentralPixel(newSym, ThresholdForNplus1 + EPS);
			smallSyms.push_back(newSym);
			BOOST_TEST(2U == tc.formGroups(smallSyms, symsIndicesPerCluster));

			// Just check that adding now a symbol below the threshold results in a single cluster
			updateCentralPixel(newSym = EmptyTinySym, ThresholdForNplus1 - EPS);
			smallSyms.push_back(newSym);
			BOOST_TEST(1U == tc.formGroups(smallSyms, symsIndicesPerCluster));
			smallSyms.pop_back(); // Remove last added element

			// Now replace the symbol beyond the threshold with the one below => a single cluster
			const_cast<TinySymData&>(smallSyms.back()) = newSym;
			BOOST_TEST(1U == tc.formGroups(smallSyms, symsIndicesPerCluster));
			
			// By now, the central pixel of the centroid has the value: TTSAS_Threshold_Member / (n+1)^2
			// The new threshold is TTSAS_Threshold_Member / (n+2)
			const double CentralPixelOfCentroidNplus1 = TTSAS_Threshold_Member / ((n+1U)*(n+1U)),
						ThresholdForNplus2 = TTSAS_Threshold_Member / (n+2U),
						BorderMemberNplus2 = CentralPixelOfCentroidNplus1 + ThresholdForNplus2;

			// Append a symbol beyond the threshold => 2 clusters
			updateCentralPixel(newSym = EmptyTinySym, BorderMemberNplus2 + EPS);
			smallSyms.push_back(newSym);
			BOOST_TEST(2U == tc.formGroups(smallSyms, symsIndicesPerCluster));

			// Bring that last symbol below the threshold => a single cluster
			updateCentralPixel(newSym = EmptyTinySym, BorderMemberNplus2 - EPS);
			const_cast<TinySymData&>(smallSyms.back()) = newSym;
			BOOST_TEST(1U == tc.formGroups(smallSyms, symsIndicesPerCluster));

			// Swap the last 2 symbols, to check that the order isn't a problem in this case.
			swap(smallSyms[n], smallSyms.back());
			BOOST_TEST(1U == tc.formGroups(smallSyms, symsIndicesPerCluster));
		}
	}

	BOOST_AUTO_TEST_CASE(CheckMemberPromotingReserves_CarefullyOrderedAndChosenSyms_ReserveBecomesParentCluster) {
		BOOST_TEST_MESSAGE("Running CheckMemberPromotingReserves_CarefullyOrderedAndChosenSyms_ReserveBecomesParentCluster");
		TTSAS_Clustering tc;
		tc.useSymsMonitor(jm);
		vector<vector<unsigned>> symsIndicesPerCluster;
		vector<const TinySymData> smallSyms;
		smallSyms.reserve(4U);
		
		/*
		Scenario with 4 symbols:

		First loop:
		1) add 1st symbol S1 which becomes right away the root of the 1st (and only) cluster C1
		2) add 2nd symbol S2 rather different from C1, but still C1 becomes reserve candidate for S2
		3) add 3rd symbol S3 just a bit more similar to C1 than S2,
			so C1 becomes reserve candidate for S3, as well
		4) add 4th symbol S4 way more similar to C1, so that C1 becomes parent of S1 and S4

		At the end of 1st loop, S1 and S4 form cluster C1, while S2 and S3 see previous C1 just as a candidate.
		Now S2 perceives the new centroid of C1 still as reserve candidate, and not as parent cluster,
		while C1 can already be promoted to parent for S3, as its centroid is more similar to S3.
		
		However, let's see how these facts are observed and acted upon during the second loop:
		5) S2 rechecks the updated C1 from previous new clusters and decides C1 is still just a candidate
		6) S3 rechecks the updated C1 from previous new clusters and decides C1 to become its parent

		At the end of 2nd loop, S1, S3 and S4 form cluster C1, while S2 sees previous C1 just as a candidate.
		But by this time, C1's centroid is already similar-enough to S2 to be promoted to parent.

		Third loop:
		7) S2 rechecks the updated C1 from updated clusters and accepts the long-waited promotion of C1
		*/

		TinySymData sym = EmptyTinySym; // changing only central pixel => same mass-center
		const double Border4 = TTSAS_Threshold_Member / 2., // Border for 4th symbol

					// Border for 2nd and 3rd symbols (1/(2^2) + 1/3) * TTSAS_Threshold_Member
					Border23 = 7. * TTSAS_Threshold_Member / 12.; 

		// 1st symbol - the root of the sole cluster
		smallSyms.push_back(sym); 

		// 2nd symbol, which promotes its updated reserve cluster to parent cluster in 3rd loop
		updateCentralPixel(sym = EmptyTinySym, Border23 + EPS);
		smallSyms.push_back(sym);

		// 3rd symbol, which accepts the parent at the end of 2nd loop from previous new clusters
		// Normally, this is also a promotion of updated reserve candidate,
		// but the update was kept in newClusters instead of updatedClusters,
		// as the cluster was just created during that loop
		updateCentralPixel(sym = EmptyTinySym, Border23 - EPS);
		smallSyms.push_back(sym);

		// 4th symbol, which accepts the parent at the end of 1st loop from new clusters
		updateCentralPixel(sym = EmptyTinySym, Border4 - EPS);
		smallSyms.push_back(sym);

		BOOST_TEST(1U == tc.formGroups(smallSyms, symsIndicesPerCluster));
	}

BOOST_AUTO_TEST_SUITE_END() // TTSAS_Clustering_Tests