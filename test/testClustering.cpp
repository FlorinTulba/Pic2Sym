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
 
 (c) 2016-2019 Florin Tulba <florintulba@yahoo.com>

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

/*
Iterating this file twice, for both values of the boolean setting PreselectionByTinySyms.
It's simpler than duplicating each test or using the BOOST_DATA_TEST_CASE approach.
*/
#if !BOOST_PP_IS_ITERATING
#pragma warning ( push, 0 )

#	include <boost/preprocessor/iteration/iterate.hpp>

#pragma warning ( pop )

// Common part until #else (included just once)
#include "testMain.h"
#include "selectBranch.h"
#include "tinySymsProvider.h"
#include "preselectManager.h"
#include "preselectionHelper.h"
#include "symbolsSupportWithPreselection.h"
#include "clusterSupportWithPreselection.h"
#include "fileIterationHelper.h"
#include "clusterEngine.h"
#include "clusterSupport.h"
#include "noClustering.h"
#include "partitionClustering.h"
#include "ttsasClustering.h"
#include "tinySym.h"
#include "jobMonitor.h"
#include "controller.h"
#include "misc.h"

using namespace std;
using namespace cv;

extern const string ClusterAlgName;
extern const double TTSAS_Threshold_Member;
extern unsigned TinySymsSz();

namespace ut {
	/// Handy provider of tiny symbols
	struct TinySymsProvider : ITinySymsProvider {
		vector<const TinySym> tinySyms;

		const vector<const TinySym>& getTinySyms() override {
			return tinySyms;
		}

		void disposeTinySyms() override {
			tinySyms.clear();
		}
	};

	/// Provides a way to use specific clustering settings during each test and revert them afterwards
	template<bool PreselMode>
	class ClusteringSettingsFixt : public PreselFixt<PreselMode> {
	protected:
		JobMonitor jm;			///< a job monitor instance
		TinySymsProvider tsp;	///< a provider of tiny symbols

#define DefineCopyAndRef(Setting, Type) \
		Type orig##Setting = Setting, &ref##Setting = const_cast<Type&>(Setting)

		DefineCopyAndRef(ClusterAlgName, string);
		DefineCopyAndRef(TTSAS_Threshold_Member, double);

#undef DefineCopyAndRef

	public:
		~ClusteringSettingsFixt() {
#define RevertIfChanged(Setting) \
			if(Setting != orig##Setting) \
				ref##Setting = orig##Setting

			RevertIfChanged(ClusterAlgName);
			RevertIfChanged(TTSAS_Threshold_Member);

#undef RevertIfChanged
		}
	};

	/// Hack to be able to refer protected clustAlg field from ClusterEngine
	struct TestClusterEngine : public ClusterEngine {
		//using ClusterEngine::clustAlg; // checkAlgType from below does the job without changing field visibility

		ITinySymsProvider *tsp;	///< provider of tiny symbols

		TestClusterEngine() : ClusterEngine(*(tsp = new TinySymsProvider), VSymData()) {}
		~TestClusterEngine() { delete tsp; }

		template<class AlgType>
		bool checkAlgType() const {
			return string(typeid(AlgType).name()).
				compare(typeid(clustAlg).name()) == 0;
		}
	};

	const unsigned TinySymsSize = TinySymsSz();
	const double TinySymArea = double(TinySymsSize*TinySymsSize);
	const unsigned TinySymMidSide = TinySymsSize>>1, // TinySymsSize is odd, so it's ok to keep center as unsigned value
				TinySymDiagsCount = (TinySymsSize<<1) - 1U;
	const Point2d TinySymCenter(TinySymMidSide, TinySymMidSide),
				UnitSquareCenter(.5, .5);

	const SymData EmptySymData5x5(0UL, 0U, 0., 0., 0., 0., TinySymCenter,
					{ { ISymData::GROUNDED_SYM_IDX, Mat(TinySymsSize, TinySymsSize, CV_64FC1, Scalar(0.)) } },
					Mat(TinySymsSize, TinySymsSize, CV_8UC1, Scalar(255U)));

	/**
	Initializes the diagonal projection of a diagonal matrix of size TinySymsSize
	All elements 0 except the central one, which is TinySymsSize * 1 = TinySymsSize

	Needed to provide a parameter for constructing MainDiagTinySym() object from below.
	*/
	const Mat slashProjectionOfMatWithMainDiagSet() {
		static Mat result(1, TinySymDiagsCount, CV_64FC1, 0.);
		static bool initialized = false;
		if(!initialized) {
			result.at<double>(TinySymsSize - 1U) = (double)TinySymsSize;
			initialized = true;
		}
		return result.clone();
	}

	const TinySym& EmptyTinySym() {
		static const TinySym ts(Mat(TinySymsSize, TinySymsSize, CV_64FC1, 255.));
		return ts;
	}
	const TinySym& MainDiagTinySym() {
		static const TinySym ts(255.-255.*Mat::eye(TinySymsSize, TinySymsSize, CV_64FC1), Point2d(.5, .5), .2);
		return ts;
	}

	/// Updates the central pixel of the tiny symbol sym and all the other involved parameters
	void updateCentralPixel(TinySym &sym, double pixelVal) {
		const double oldPixelVal = sym.mat.at<double>(TinySymMidSide, TinySymMidSide),
					oldPixSum = sym.getAvgPixVal() * TinySymArea,
					diff = pixelVal - oldPixelVal,
					avgDiff = diff / TinySymsSize;

		sym.mc = (oldPixSum*sym.getMc() + diff*UnitSquareCenter) / (oldPixSum + diff);
		
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
		UpdateProjectionAt(backslashDiagAvgProj, TinySymsSize-1);
		UpdateProjectionAt(slashDiagAvgProj, TinySymsSize-1);

#undef UpdateProjectionAt
	};

	/// symIdx from the symsSet become consecutive values
	void setConsecIndices(const SymData &sym, VSymData &symsSet) {
		size_t idx = 0U;
		for(auto &sd : symsSet)
			sd = move(sym.clone(idx++));
	}
}

using namespace ut;

/*
Iterating this file 2 times, with counter values from 0 to 1.
	0 will be used for PreselectionByTinySyms set on false
	1 will be used for PreselectionByTinySyms set on true
*/
#	define BOOST_PP_ITERATION_LIMITS		(0, 1)
#	define BOOST_PP_FILENAME_1				"testClustering.cpp" /* __FILE__ didn't work! */
#	include BOOST_PP_ITERATE()

#else // BOOST_PP_IS_ITERATING is 1 (true) -- The rest of the file is iterated twice

#	if BOOST_PP_ITERATION() == 0
#		define SuiteFixture	ClusteringSettingsFixt<false>
#		define SuiteSuffix	_noPreselection

#	elif BOOST_PP_ITERATION() == 1
#		undef SuiteFixture
#		define SuiteFixture	ClusteringSettingsFixt<true>
#		undef SuiteSuffix
#		define SuiteSuffix	_withPreselection

#	else // BOOST_PP_ITERATION() >= 2
#		undef SuiteFixture
#		undef SuiteSuffix
#	endif // BOOST_PP_ITERATION()

FixtureTestSuiteSuffix(SuiteFixture, ClusterEngineCreation_Tests, SuiteSuffix)
	AutoTestCase1(ClusterEngineCreation_InvalidAlgName_UsingNoClustering, SuiteSuffix);
		refClusterAlgName = "Invalid_Algorithm_Name!!!!";
		TestClusterEngine ce;
		BOOST_REQUIRE(ce.checkAlgType<NoClustering>());
	}

	AutoTestCase1(ClusterEngineCreation_NoClusteringRequest_UsingNoClustering, SuiteSuffix);
		refClusterAlgName = "None";
		TestClusterEngine ce;
		BOOST_REQUIRE(ce.checkAlgType<NoClustering>());
	}

	AutoTestCase1(ClusterEngineCreation_PartitionClusteringRequest_UsingPartitionClustering, SuiteSuffix);
		refClusterAlgName = "Partition";
		TestClusterEngine ce;
		BOOST_REQUIRE(ce.checkAlgType<PartitionClustering>());
	}

	AutoTestCase1(ClusterEngineCreation_TTSASClusteringRequest_UsingTTSASClustering, SuiteSuffix);
		refClusterAlgName = "TTSAS";
		TestClusterEngine ce;
		BOOST_REQUIRE(ce.checkAlgType<TTSAS_Clustering>());
	}
BOOST_AUTO_TEST_SUITE_END() // ClusterEngineCreation

FixtureTestSuiteSuffix(SuiteFixture, BasicClustering_Tests, SuiteSuffix)
	AutoTestCase1(UsingNoClustering_6identicalSymbols_0nonTrivialClusters, SuiteSuffix);
		refClusterAlgName = "None";
		size_t symsCount = 6U;
		tsp.tinySyms.assign(symsCount, EmptyTinySym());
		VSymData symsSet(symsCount); setConsecIndices(EmptySymData5x5, symsSet);
		ClusterEngine ce(tsp, symsSet);
		ce.useSymsMonitor(jm);
		ce.support().groupSyms();
		BOOST_REQUIRE(!ce.worthGrouping());
		BOOST_REQUIRE(ce.getClustersCount() == symsCount);
	}

	AutoTestCase1(UsingPartitionClustering_6identicalSymbols_noTrivialClusters, SuiteSuffix);
		refClusterAlgName = "Partition";
		const size_t symsCount = 6U;
		tsp.tinySyms.assign(symsCount, EmptyTinySym());
		VSymData symsSet(symsCount); setConsecIndices(EmptySymData5x5, symsSet);
		ClusterEngine ce(tsp, symsSet);
		ce.useSymsMonitor(jm);
		ce.support().groupSyms();
		BOOST_REQUIRE(ce.getClustersCount() == 1U);
	}

	AutoTestCase1(UsingTTSASclustering_6identicalSymbols_noTrivialClusters, SuiteSuffix);
		refClusterAlgName = "TTSAS";
		const size_t symsCount = 6U;
		tsp.tinySyms.assign(symsCount, EmptyTinySym());
		VSymData symsSet(symsCount); setConsecIndices(EmptySymData5x5, symsSet);
		ClusterEngine ce(tsp, symsSet);
		ce.useSymsMonitor(jm);
		ce.support().groupSyms();
		BOOST_REQUIRE(ce.getClustersCount() == 1U);
	}

	AutoTestCase1(UsingPartitionClustering_x0x0xSequence_2Clusters, SuiteSuffix);
		refClusterAlgName = "Partition";
		tsp.tinySyms = vector<const TinySym>{ MainDiagTinySym(), EmptyTinySym(), MainDiagTinySym(), EmptyTinySym(), MainDiagTinySym() };
		const size_t symsCount = tsp.tinySyms.size();
		VSymData symsSet(symsCount); setConsecIndices(EmptySymData5x5, symsSet);
		ClusterEngine ce(tsp, symsSet);
		ce.useSymsMonitor(jm);
		ce.support().groupSyms();
		BOOST_REQUIRE(ce.worthGrouping());
		const auto &clusterOffsets = ce.getClusterOffsets();
		const auto &clusters = ce.getClusters();
		BOOST_REQUIRE(clusters.size() == 2U);
		BOOST_REQUIRE(clusters[0]->getIdxOfFirstSym() == 0 && clusters[0]->getSz() == 3); // largest cluster starts at 0 (3 items)
		BOOST_REQUIRE(clusters[1]->getIdxOfFirstSym() == 3 && clusters[1]->getSz() == 2); // smallest cluster starts at 3 (2 items)
		BOOST_REQUIRE(clusterOffsets.size() == 3U);
		BOOST_REQUIRE(clusterOffsets.find(0U) != clusterOffsets.end()); // largest cluster starts at 0 (3 items)
		BOOST_REQUIRE(clusterOffsets.find(3U) != clusterOffsets.end()); // smallest cluster starts at 3 (2 items)
		BOOST_REQUIRE(clusterOffsets.find((unsigned)symsCount) != clusterOffsets.end());
	}

	AutoTestCase1(UsingTTSASclustering_x0x0xSequence_2Clusters, SuiteSuffix);
		refClusterAlgName = "TTSAS";
		tsp.tinySyms = vector<const TinySym>{ MainDiagTinySym(), EmptyTinySym(), MainDiagTinySym(), EmptyTinySym(), MainDiagTinySym() };
		const size_t symsCount = tsp.tinySyms.size();
		VSymData symsSet(symsCount); setConsecIndices(EmptySymData5x5, symsSet);
		ClusterEngine ce(tsp, symsSet);
		ce.useSymsMonitor(jm);
		ce.support().groupSyms();
		BOOST_REQUIRE(ce.worthGrouping());
		const auto &clusterOffsets = ce.getClusterOffsets();
		const auto &clusters = ce.getClusters();
		BOOST_REQUIRE(clusters.size() == 2U);
		BOOST_REQUIRE(clusters[0]->getIdxOfFirstSym() == 0 && clusters[0]->getSz() == 3); // largest cluster starts at 0 (3 items)
		BOOST_REQUIRE(clusters[1]->getIdxOfFirstSym() == 3 && clusters[1]->getSz() == 2); // smallest cluster starts at 3 (2 items)
		BOOST_REQUIRE(clusterOffsets.size() == 3U);
		BOOST_REQUIRE(clusterOffsets.find(0U) != clusterOffsets.end()); // largest cluster starts at 0 (3 items)
		BOOST_REQUIRE(clusterOffsets.find(3U) != clusterOffsets.end()); // smallest cluster starts at 3 (2 items)
		BOOST_REQUIRE(clusterOffsets.find((unsigned)symsCount) != clusterOffsets.end());
	}
BOOST_AUTO_TEST_SUITE_END() // BasicClustering_Tests

FixtureTestSuiteSuffix(SuiteFixture, TTSAS_Clustering_Tests, SuiteSuffix)
	AutoTestCase1(CheckMemberThresholdTTSAS_BelowOrAboveThreshold_MemberOrNot, SuiteSuffix);
		TTSAS_Clustering tc;
		tc.useSymsMonitor(jm).setTinySymsProvider(tsp);
		vector<vector<unsigned>> symsIndicesPerCluster;
		VSymData symsSet;

		for(unsigned n = 1U; n < 30U; ++n) {
			// Use n identical symbols => a single cluster
			BOOST_TEST_MESSAGE("Checking member thresholds with a set of " << n << " identical symbols");

			tsp.tinySyms.assign(n, EmptyTinySym()); symsSet.resize(n); setConsecIndices(EmptySymData5x5, symsSet);
			BOOST_REQUIRE(1U == tc.formGroups(symsSet, symsIndicesPerCluster));

			const double ThresholdForNplus1 = TTSAS_Threshold_Member / (n+1U);

			// Append a symbol beyond the threshold => 2 clusters
			TinySym newSym = EmptyTinySym(); // changing only central pixel => same mass-center
			updateCentralPixel(newSym, ThresholdForNplus1 + EPS);
			tsp.tinySyms.push_back(newSym); symsSet.push_back(nullptr); setConsecIndices(EmptySymData5x5, symsSet);
			BOOST_TEST(2U == tc.formGroups(symsSet, symsIndicesPerCluster));

			// Just check that adding now a symbol below the threshold results in a single cluster
			updateCentralPixel(newSym = EmptyTinySym(), ThresholdForNplus1 - EPS);
			tsp.tinySyms.push_back(newSym); symsSet.push_back(nullptr); setConsecIndices(EmptySymData5x5, symsSet);
			BOOST_TEST(1U == tc.formGroups(symsSet, symsIndicesPerCluster));
			tsp.tinySyms.pop_back(); symsSet.pop_back(); // Remove last added element

			// Now replace the symbol beyond the threshold with the one below => a single cluster
			const_cast<TinySym&>(tsp.tinySyms.back()) = newSym;
			BOOST_TEST(1U == tc.formGroups(symsSet, symsIndicesPerCluster));
			
			// By now, the central pixel of the centroid has the value: TTSAS_Threshold_Member / (n+1)^2
			// The new threshold is TTSAS_Threshold_Member / (n+2)
			const double CentralPixelOfCentroidNplus1 = TTSAS_Threshold_Member / ((n+1U)*(n+1U)),
						ThresholdForNplus2 = TTSAS_Threshold_Member / (n+2U),
						BorderMemberNplus2 = CentralPixelOfCentroidNplus1 + ThresholdForNplus2;

			// Append a symbol beyond the threshold => 2 clusters
			updateCentralPixel(newSym = EmptyTinySym(), BorderMemberNplus2 + EPS);
			tsp.tinySyms.push_back(newSym); symsSet.push_back(nullptr); setConsecIndices(EmptySymData5x5, symsSet);
			BOOST_TEST(2U == tc.formGroups(symsSet, symsIndicesPerCluster));

			// Bring that last symbol below the threshold => a single cluster
			updateCentralPixel(newSym = EmptyTinySym(), BorderMemberNplus2 - EPS);
			const_cast<TinySym&>(tsp.tinySyms.back()) = newSym;
			BOOST_TEST(1U == tc.formGroups(symsSet, symsIndicesPerCluster));

			// Swap the last 2 symbols, to check that the order isn't a problem in this case.
			swap(tsp.tinySyms[n], tsp.tinySyms.back());
			BOOST_TEST(1U == tc.formGroups(symsSet, symsIndicesPerCluster));
		}
	}

	AutoTestCase1(CheckMemberPromotingReserves_CarefullyOrderedAndChosenSyms_ReserveBecomesParentCluster, SuiteSuffix);
		TTSAS_Clustering tc;
		tc.useSymsMonitor(jm).setTinySymsProvider(tsp);
		vector<vector<unsigned>> symsIndicesPerCluster;
		VSymData symsSet;
		tsp.tinySyms.reserve(4U);
		
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

		TinySym sym = EmptyTinySym(); // changing only central pixel => same mass-center
		const double Border4 = TTSAS_Threshold_Member / 2., // Border for 4th symbol

					// Border for 2nd and 3rd symbols (1/(2^2) + 1/3) * TTSAS_Threshold_Member
					Border23 = 7. * TTSAS_Threshold_Member / 12.; 

		// 1st symbol - the root of the sole cluster
		tsp.tinySyms.push_back(sym); symsSet.push_back(nullptr); setConsecIndices(EmptySymData5x5, symsSet);

		// 2nd symbol, which promotes its updated reserve cluster to parent cluster in 3rd loop
		updateCentralPixel(sym = EmptyTinySym(), Border23 + EPS);
		tsp.tinySyms.push_back(sym); symsSet.push_back(nullptr); setConsecIndices(EmptySymData5x5, symsSet);

		// 3rd symbol, which accepts the parent at the end of 2nd loop from previous new clusters
		// Normally, this is also a promotion of updated reserve candidate,
		// but the update was kept in newClusters instead of updatedClusters,
		// as the cluster was just created during that loop
		updateCentralPixel(sym = EmptyTinySym(), Border23 - EPS);
		tsp.tinySyms.push_back(sym); symsSet.push_back(nullptr); setConsecIndices(EmptySymData5x5, symsSet);

		// 4th symbol, which accepts the parent at the end of 1st loop from new clusters
		updateCentralPixel(sym = EmptyTinySym(), Border4 - EPS);
		tsp.tinySyms.push_back(sym); symsSet.push_back(nullptr); setConsecIndices(EmptySymData5x5, symsSet);

		BOOST_TEST(1U == tc.formGroups(symsSet, symsIndicesPerCluster));
	}
BOOST_AUTO_TEST_SUITE_END() // TTSAS_Clustering_Tests

#endif // BOOST_PP_IS_ITERATING
