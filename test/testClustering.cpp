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
#include "clusterEngine.h"
#include "noClustering.h"
#include "partitionClustering.h"
#include "ttsasClustering.h"

using namespace std;
using namespace cv;

extern const string ClusterAlgName;
extern const double MaxAvgProjErrForPartitionClustering;
extern const double MaxRelMcOffsetForPartitionClustering;
extern const double MaxDiffAvgPixelValForPartitionClustering;
extern const bool TTSAS_Accept1stClusterThatQualifiesAsParent;
extern const double TTSAS_Threshold_Member;
extern const double TTSAS_Threshold_Outsider;
extern const double MaxRelMcOffsetForTTSAS_Clustering;
extern const double MaxDiffAvgPixelValForTTSAS_Clustering;

namespace ut {
	/// Provides a way to use specific clustering settings during each test and revert them afterwards
	class ClusteringSettingsFixt : public Fixt {
	protected:
#define DEFINE_COPY_AND_REF(Setting, Type) \
		Type orig##Setting = Setting, &ref##Setting = const_cast<Type&>(Setting)

		DEFINE_COPY_AND_REF(ClusterAlgName, string);
		DEFINE_COPY_AND_REF(TTSAS_Accept1stClusterThatQualifiesAsParent, bool);
		DEFINE_COPY_AND_REF(MaxAvgProjErrForPartitionClustering, double);
		DEFINE_COPY_AND_REF(MaxRelMcOffsetForPartitionClustering, double);
		DEFINE_COPY_AND_REF(MaxDiffAvgPixelValForPartitionClustering, double);
		DEFINE_COPY_AND_REF(TTSAS_Threshold_Member, double);
		DEFINE_COPY_AND_REF(TTSAS_Threshold_Outsider, double);
		DEFINE_COPY_AND_REF(MaxRelMcOffsetForTTSAS_Clustering, double);
		DEFINE_COPY_AND_REF(MaxDiffAvgPixelValForTTSAS_Clustering, double);

#undef DEFINE_COPY_AND_REF

	public:
		~ClusteringSettingsFixt() {
#define REVERT_IF_CHANGED(Setting) \
			if(Setting != orig##Setting) \
				ref##Setting = orig##Setting

			REVERT_IF_CHANGED(ClusterAlgName);
			REVERT_IF_CHANGED(TTSAS_Accept1stClusterThatQualifiesAsParent);
			REVERT_IF_CHANGED(MaxAvgProjErrForPartitionClustering);
			REVERT_IF_CHANGED(MaxRelMcOffsetForPartitionClustering);
			REVERT_IF_CHANGED(MaxDiffAvgPixelValForPartitionClustering);
			REVERT_IF_CHANGED(MaxDiffAvgPixelValForPartitionClustering);
			REVERT_IF_CHANGED(TTSAS_Threshold_Member);
			REVERT_IF_CHANGED(TTSAS_Threshold_Outsider);
			REVERT_IF_CHANGED(MaxRelMcOffsetForTTSAS_Clustering);
			REVERT_IF_CHANGED(MaxDiffAvgPixelValForTTSAS_Clustering);

#undef REVERT_IF_CHANGED
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

	const SymData EmptySym5x5(0UL, 0., 0., 0., Point2d(2., 2.),
		{ { SymData::NEG_SYM_IDX, Mat(5, 5, CV_8UC1, Scalar(255U)) },
		{ SymData::GROUNDED_SYM_IDX, Mat(5, 5, CV_64FC1, Scalar(0.)) } }),
				MainDiag5x5(1UL, 0., 1., 5., Point2d(2., 2.),
		{ { SymData::NEG_SYM_IDX, (255U - Mat::eye(5, 5, CV_8UC1) * 255U) }, 
		{ SymData::GROUNDED_SYM_IDX, Mat::eye(5, 5, CV_64FC1) } });
}

BOOST_FIXTURE_TEST_SUITE(ClusterEngineCreation_Tests, ut::ClusteringSettingsFixt)
	BOOST_AUTO_TEST_CASE(ClusterEngineCreation_InvalidAlgName_UsingNoClustering) {
		BOOST_TEST_MESSAGE("Running ClusterEngineCreation_InvalidAlgName_UsingNoClustering");
		refClusterAlgName = "Invalid_Algorithm_Name!!!!";
		ut::TestClusterEngine ce;
		BOOST_REQUIRE(ce.checkAlgType<NoClustering>());
	}

	BOOST_AUTO_TEST_CASE(ClusterEngineCreation_NoClusteringRequest_UsingNoClustering) {
		BOOST_TEST_MESSAGE("Running ClusterEngineCreation_NoClusteringRequest_UsingNoClustering");
		refClusterAlgName = "None";
		ut::TestClusterEngine ce;
		BOOST_REQUIRE(ce.checkAlgType<NoClustering>());
	}

	BOOST_AUTO_TEST_CASE(ClusterEngineCreation_PartitionClusteringRequest_UsingPartitionClustering) {
		BOOST_TEST_MESSAGE("Running ClusterEngineCreation_NoClusteringRequest_UsingNoClustering");
		refClusterAlgName = "Partition";
		ut::TestClusterEngine ce;
		BOOST_REQUIRE(ce.checkAlgType<PartitionClustering>());
	}

	BOOST_AUTO_TEST_CASE(ClusterEngineCreation_TTSASClusteringRequest_UsingTTSASClustering) {
		BOOST_TEST_MESSAGE("Running ClusterEngineCreation_NoClusteringRequest_UsingNoClustering");
		refClusterAlgName = "TTSAS";
		ut::TestClusterEngine ce;
		BOOST_REQUIRE(ce.checkAlgType<TTSAS_Clustering>());
	}

BOOST_AUTO_TEST_SUITE_END() // ClusterEngineCreation

BOOST_FIXTURE_TEST_SUITE(BasicClustering_Tests, ut::ClusteringSettingsFixt)
	BOOST_AUTO_TEST_CASE(UsingNoClustering_6identicalSymbols_0nonTrivialClusters) {
		BOOST_TEST_MESSAGE("Running UsingNoClustering_6identicalSymbols_0nonTrivialClusters");
		refClusterAlgName = "None";
		ClusterEngine ce;
		size_t symsCount = 6U;
		VSymData symsSet(symsCount, ut::EmptySym5x5);
		ce.process(symsSet);
		BOOST_REQUIRE(ce.getClusters().size() == symsCount);
	}

	BOOST_AUTO_TEST_CASE(UsingPartitionClustering_6identicalSymbols_noTrivialClusters) {
		BOOST_TEST_MESSAGE("Running UsingPartitionClustering_6identicalSymbols_noTrivialClusters");
		refClusterAlgName = "Partition";
		ClusterEngine ce;
		const size_t symsCount = 6U;
		VSymData symsSet(symsCount, ut::EmptySym5x5);
		ce.process(symsSet);
		BOOST_REQUIRE(ce.getClusters().size() == 1U);
	}

	BOOST_AUTO_TEST_CASE(UsingTTSASclustering_6identicalSymbols_noTrivialClusters) {
		BOOST_TEST_MESSAGE("Running UsingTTSASclustering_6identicalSymbols_noTrivialClusters");
		refClusterAlgName = "TTSAS";
		ClusterEngine ce;
		const size_t symsCount = 6U;
		VSymData symsSet(symsCount, ut::EmptySym5x5);
		ce.process(symsSet);
		BOOST_REQUIRE(ce.getClusters().size() == 1U);
	}

	BOOST_AUTO_TEST_CASE(UsingPartitionClustering_x0x0xSequence_2Clusters) {
		BOOST_TEST_MESSAGE("Running UsingPartitionClustering_x0x0xSequence_2Clusters");
		refClusterAlgName = "Partition";
		ClusterEngine ce;
		VSymData symsSet { ut::MainDiag5x5, ut::EmptySym5x5, ut::MainDiag5x5, ut::EmptySym5x5, ut::MainDiag5x5 };
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
		VSymData symsSet { ut::MainDiag5x5, ut::EmptySym5x5, ut::MainDiag5x5, ut::EmptySym5x5, ut::MainDiag5x5 };
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

// Generate several random centroids, so that they are distant enough.
// For each centroid, generate a few random cluster members.
// Shuffle the resulted symbols.
// Check afterwards if the algorithms correctly groups the symbols.
// Accept tests if most characteristics match.
// Display desired and obtained clusters when grouping appears different than expected.
// Test suite checking FontEngine constraints
