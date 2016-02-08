/**********************************************************
 Project:     UnitTesting
 File:        testMatch.cpp

 Author:      Florin Tulba
 Created on:  2016-2-8
 
 Copyright (c) 2016 Florin Tulba
 **********************************************************/

#include "testMain.h"
#include "controller.h"

#include <random>
#include <algorithm>
#include <numeric>
#include <iterator>

#include <boost/optional/optional.hpp>

using namespace cv;
using namespace std;
using namespace boost;

BOOST_FIXTURE_TEST_SUITE(MatchEngine_Tests, ut::Fixt)
	BOOST_AUTO_TEST_CASE(MatchEngine_CheckParams) {
		BOOST_TEST_MESSAGE("Running MatchEngine_CheckParams ...");

		// Patches have pixels with double values 0..255.
		// Glyphs have pixels with double values 0..1.
		// Masks have pixels with byte values 0..255.

		random_device rd;
		mt19937 gen(rd());
		uniform_int_distribution<unsigned> uid;

		const unsigned sz = 50U; // patches of 50x50
		const unsigned char valRand = (unsigned char)uid(gen)&0xFFU;
		const Mat emptyUc(sz, sz, CV_8UC1, Scalar(0U)), emptyD(sz, sz, CV_64FC1, Scalar(0.)),
			wholeUc(sz, sz, CV_8UC1, Scalar(255U)),
			wholeD255(sz, sz, CV_64FC1, Scalar(255.)), wholeD1(sz, sz, CV_64FC1, Scalar(1.));
		vector<unsigned char> randV;
		generate_n(back_inserter(randV), sz*sz, [&gen, &uid] {
			return (unsigned char)uid(gen)&0xFFU;
		});
		Mat randUc(sz, sz, CV_8UC1, (void*)randV.data()), randD1, randD255;
		randUc.convertTo(randD1, CV_64FC1, 1./255);
		randUc.convertTo(randD255, CV_64FC1);

		MatchParams mp;
		Mat consec(1, sz, CV_64FC1);
		iota(consec.begin<double>(), consec.end<double>(), (double)0.); // 0..sz-1
		CachedData cd;
		cd.sz = sz; cd.sz_1 = sz-1U, cd.sz2 = (double)sz*sz;
		cd.smallGlyphsCoverage = .1; cd.preferredMaxMcDist = 3./8*sz;
		cd.mcDistMax = (sz-1U)*sqrt(2);
		cd.patchCenter = Point2d((sz-1U)/2., (sz-1U)/2.);
		cd.consec = consec;

		optional<double> miu, sdev, miu1, sdev1;

		// Check that computeSdev performs the same when preceded or not by computeMean
		MatchParams::computeMean(randD255, wholeUc, miu);
		MatchParams::computeSdev(randD255, wholeUc, miu, sdev); // miu already computed
		MatchParams::computeSdev(randD255, wholeUc, miu1, sdev1); // miu1 not computed yet
		BOOST_REQUIRE(miu && sdev && miu1 && sdev1);
		BOOST_TEST(*miu == *miu1, test_tools::tolerance(1e-4));
		BOOST_TEST(*sdev == *sdev1, test_tools::tolerance(1e-4));
		miu = sdev = miu1 = sdev1 = none;

		// Random data, empty mask => mean = sdev = 0
		MatchParams::computeSdev(randD255, emptyUc, miu, sdev);
		BOOST_REQUIRE(miu && sdev);
		BOOST_TEST(*miu == 0., test_tools::tolerance(1e-4));
		BOOST_TEST(*sdev == 0., test_tools::tolerance(1e-4));
		miu = sdev = none;

		// Uniform data, random mask => mean = valRand, sdev = 0
		MatchParams::computeSdev(Mat(sz, sz, CV_64FC1, Scalar(valRand)), randUc!=0U, miu, sdev);
		BOOST_REQUIRE(miu && sdev);
		BOOST_TEST(*miu == valRand, test_tools::tolerance(1e-4));
		BOOST_TEST(*sdev == 0., test_tools::tolerance(1e-4));
		miu = sdev = none;

		// Half 0, half 255, no mask =>
		//		mean = sdev = 127.5 (sdevMax)
		//		mass-center = ( (sz-1)/2 ,  255*((sz/2-1)*(sz/2)/2)/(255*sz/2) = (sz/2-1)/2 )
		Mat halfH = emptyD.clone(); halfH.rowRange(0, sz/2) = 255.;
		MatchParams::computeSdev(halfH, wholeUc, miu, sdev);
		BOOST_REQUIRE(miu && sdev);
		BOOST_TEST(*miu == 127.5, test_tools::tolerance(1e-4));
		BOOST_TEST(*sdev == CachedData::sdevMax, test_tools::tolerance(1e-4));
		miu = sdev = none;
		mp.computeMcPatch(halfH, cd);
		BOOST_REQUIRE(mp.mcPatch);
		BOOST_TEST(mp.mcPatch->x == (sz-1U)/2., test_tools::tolerance(1e-4));
		BOOST_TEST(mp.mcPatch->y == (sz/2.-1U)/2., test_tools::tolerance(1e-4));
		mp.mcPatch = none;

		//SymData sd();
	}

	BOOST_AUTO_TEST_CASE(MatchEngine_Check) {
		BOOST_TEST_MESSAGE("Running MatchEngine_Check ...");
		Config cfg(10U, 1., 1., 1.);
		Controller c(cfg);
		FontEngine &fe = c.getFontEngine();
		MatchEngine &me = c.getMatchEngine(cfg);

		BOOST_REQUIRE_NO_THROW(c.newFontFamily("res\\BPmonoBold.ttf"));
		BOOST_REQUIRE_NO_THROW(c.newFontEncoding("APPLE_ROMAN"));

		BOOST_REQUIRE(!c.performTransformation()); // no image yet

		BOOST_REQUIRE(!c.newImage(Mat())); // wrong image

		Mat testPatch(c.getFontSize(), c.getFontSize(), CV_8UC1, Scalar(127)),
			testColorPatch(c.getFontSize(), c.getFontSize(), CV_8UC3, Scalar::all(127));

		BOOST_REQUIRE(c.newImage(testPatch)); // image ok
		BOOST_REQUIRE(c.performTransformation()); // ok

		BOOST_REQUIRE(c.newImage(testColorPatch)); // image ok
		BOOST_REQUIRE(c.performTransformation()); // ok
	}
BOOST_AUTO_TEST_SUITE_END() // MatchEngine_Tests
