/**********************************************************
 Project:     UnitTesting
 File:        testMatch.cpp

 Author:      Florin Tulba
 Created on:  2016-2-8
 
 Copyright (c) 2016 Florin Tulba
 **********************************************************/

#include "testMain.h"
#include "misc.h"
#include "controller.h"

#include <random>
#include <algorithm>
#include <numeric>
#include <iterator>
#include <set>

#include <boost/optional/optional.hpp>

using namespace cv;
using namespace std;
using namespace boost;

namespace ut {
	const double NOT_RELEVANT_D = numeric_limits<double>::infinity();
	const unsigned long NOT_RELEVANT_UL = ULONG_MAX;
	const Point2d NOT_RELEVANT_POINT;
	const Mat NOT_RELEVANT_MAT;

	unsigned randUnifUint() {
		static random_device rd;
		static mt19937 gen(rd());
		static uniform_int_distribution<unsigned> uid;
		return uid(gen);
	}

	void alterFgBg(Mat &patchUc, double minVal01, double diffMinMax01) {
		const unsigned char newFg = (unsigned char)randUnifUint()&0xFFU;
		unsigned char newBg;
		int newDiff;
		do {
			newBg = (unsigned char)(randUnifUint() & 0xFFU);
			newDiff = (int)newFg - (int)newBg;
		} while(abs(newDiff) < 30); // keep fg & bg at least 30 brightness units apart
		
		patchUc = (patchUc - (minVal01*255)) * (newDiff/(255*diffMinMax01)) + newBg;
	}

	void addWhiteNoise(Mat &patchUc, double affectedPercentage01, unsigned char maxAmplitude0255) {
		const int side = patchUc.rows, area = side*side, affectedCount = (int)(affectedPercentage01 * area);

		int noise;
		const unsigned twiceMaxAmplitude = ((unsigned)maxAmplitude0255)<<1;
		int prevVal, below, above;
		set<unsigned> affected;
		unsigned linearized;
		div_t pos; // coordinate inside the Mat, expressed as quotient and remainder of linearized
		for(int i = 0; i<affectedCount; ++i) {
			do {
				linearized = randUnifUint() % (unsigned)area;
			} while(affected.find(linearized) != affected.cend());
			affected.insert(linearized);
			pos = div((int)linearized, side);

			prevVal = (int)round(patchUc.at<double>(pos.quot, pos.rem));
			below = max(0, min((int)maxAmplitude0255, prevVal));
			above = max(0, min((int)maxAmplitude0255, 255 - prevVal));
			do {
				noise = (int)(randUnifUint() % (above+below+1)) - below;
			} while(noise == 0);

			patchUc.at<double>(pos.quot, pos.rem) = prevVal + noise;
		}
	}
}

using namespace ut;

BOOST_FIXTURE_TEST_SUITE(MatchEngine_Tests, Fixt)
	BOOST_AUTO_TEST_CASE(MatchEngine_CheckParams) {
		BOOST_TEST_MESSAGE("Running MatchEngine_CheckParams ...");

		// Patches have pixels with double values 0..255.
		// Glyphs have pixels with double values 0..1.
		// Masks have pixels with byte values 0..255.

		const unsigned sz = 50U; // Select an even sz, as the tests need to set exactly half pixels
		const unsigned char valRand = (unsigned char)(randUnifUint() & 0xFFU);
		const Mat
			// completely empty
			emptyUc(sz, sz, CV_8UC1, Scalar(0U)), emptyD(sz, sz, CV_64FC1, Scalar(0.)),
			
			// completely full
			wholeUc(sz, sz, CV_8UC1, Scalar(255U)),
			wholeD255(sz, sz, CV_64FC1, Scalar(255.)), wholeD1(sz, sz, CV_64FC1, Scalar(1.));

		// 2 rows mid height
		Mat horBeltUc = emptyUc.clone(); horBeltUc.rowRange(sz/2-1, sz/2+1) = 255U;

		// 2 columns mid width
		Mat verBeltUc = emptyUc.clone(); verBeltUc.colRange(sz/2-1, sz/2+1) = 255U;

		// 1st horizontal half full
		Mat halfUc = emptyUc.clone(); halfUc.rowRange(0, sz/2) = 255U;
		Mat halfD255 = emptyD.clone(); halfD255.rowRange(0, sz/2) = 255.;
		Mat halfD1 = emptyD.clone(); halfD1.rowRange(0, sz/2) = 1.;

		// 2nd horizontal half full
		Mat invHalfUc = 255U - halfUc;
		Mat invHalfD255 = 255. - halfD255;

		vector<unsigned char> randV;
		generate_n(back_inserter(randV), sz*sz, [] {
			return (unsigned char)(randUnifUint() & 0xFFU);
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
		cd.complPrefMaxMcDist = (sz-1U)*sqrt(2) - cd.preferredMaxMcDist;
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

		// Random patch, empty mask => mean = sdev = 0
		MatchParams::computeSdev(randD255, emptyUc, miu, sdev);
		BOOST_REQUIRE(miu && sdev);
		BOOST_TEST(*miu == 0., test_tools::tolerance(1e-4));
		BOOST_TEST(*sdev == 0., test_tools::tolerance(1e-4));
		miu = sdev = none;

		// Uniform patch, random mask => mean = valRand, sdev = 0
		MatchParams::computeSdev(Mat(sz, sz, CV_64FC1, Scalar(valRand)), randUc!=0U, miu, sdev);
		BOOST_REQUIRE(miu && sdev);
		BOOST_TEST(*miu == (double)valRand, test_tools::tolerance(1e-4));
		BOOST_TEST(*sdev == 0., test_tools::tolerance(1e-4));
		miu = sdev = none;

		// Patch half 0, half 255, no mask =>
		//		mean = sdev = 127.5 (sdevMax)
		//		mass-center = ( (sz-1)/2 ,  255*((sz/2-1)*(sz/2)/2)/(255*sz/2) = (sz/2-1)/2 )
		MatchParams::computeSdev(halfD255, wholeUc, miu, sdev);
		BOOST_REQUIRE(miu && sdev);
		BOOST_TEST(*miu == 127.5, test_tools::tolerance(1e-4));
		BOOST_TEST(*sdev == CachedData::sdevMax, test_tools::tolerance(1e-4));
		mp.computeMcPatch(halfD255, cd);
		BOOST_REQUIRE(mp.mcPatch);
		BOOST_TEST(mp.mcPatch->x == cd.patchCenter.x, test_tools::tolerance(1e-4));
		BOOST_TEST(mp.mcPatch->y == (sz/2.-1U)/2., test_tools::tolerance(1e-4));
		
		// Testing a glyph half 0, half 255
		SymData sdHorizEdgeMask(NOT_RELEVANT_UL,	// glyph code (not relevant here)
					0., // min glyph value (0..1 range)
					1.,	// difference between min and max glyph (0..1 range)
				   sz*sz/2.,	// pixelSum = 255*(sz^2/2)/255 = sz^2/2
				   Point2d(cd.patchCenter.x, (sz/2.-1U)/2.), // glyph's mass center
				   SymData::MatArray { {
						   halfUc,			// fg byte mask (0 or 255)
						   invHalfUc,		// bg byte mask (0 or 255)
						   horBeltUc,		// edge byte mask (0 or 255)
						   invHalfUc,		// glyph inverse in 0..255 byte range
						   halfD1 } });		// grounded glyph is same as glyph (min is already 0)
		SymData sdVertEdgeMask(sdHorizEdgeMask);
		*(const_cast<Mat*>(&sdVertEdgeMask.symAndMasks[SymData::EDGE_MASK_IDX])) = verBeltUc;

		// Checking glyph density for the given glyph
		mp.reset();
		mp.computeRhoApproxSym(sdHorizEdgeMask, cd);
		BOOST_REQUIRE(mp.glyphWeight);
		BOOST_TEST(*mp.glyphWeight == 0.5, test_tools::tolerance(1e-4));
		
		// Testing the mentioned glyph on an uniform patch (all pixels are 'valRand') =>
		// Adapting glyph's fg & bg to match the patch => the glyph becomes all 'valRand' =>
		// Its mass-center will be ( (sz-1)/2 , (sz-1)/2 )
		Mat unifPatchD255(sz, sz, CV_64FC1, Scalar(valRand));
		mp.reset();
		mp.computeMcApproxSym(unifPatchD255, sdHorizEdgeMask, cd);
		BOOST_REQUIRE(mp.mcGlyph);
		BOOST_TEST(mp.mcGlyph->x == cd.patchCenter.x, test_tools::tolerance(1e-4));
		BOOST_TEST(mp.mcGlyph->y == cd.patchCenter.y, test_tools::tolerance(1e-4));
		mp.computeSdevEdge(unifPatchD255, sdHorizEdgeMask);
		BOOST_REQUIRE(mp.sdevEdge);
		BOOST_TEST(*mp.sdevEdge == 0., test_tools::tolerance(1e-4));
		mp.reset();
		mp.computeSdevEdge(unifPatchD255, sdVertEdgeMask);
		BOOST_REQUIRE(mp.sdevEdge);
		BOOST_TEST(*mp.sdevEdge == 0., test_tools::tolerance(1e-4));

		// Testing the mentioned glyph on a patch equal to glyph's inverse =>
		// Adapting glyph's fg & bg to match the patch => the glyph inverses =>
		// Its mass-center will be ( (sz-1)/2 ,  (3*sz/2-1)/2 )
		mp.reset();
		mp.computeMcApproxSym(invHalfD255, sdHorizEdgeMask, cd);
		BOOST_REQUIRE(mp.mcGlyph);
		BOOST_TEST(mp.mcGlyph->x == cd.patchCenter.x, test_tools::tolerance(1e-4));
		BOOST_TEST(mp.mcGlyph->y == (3*sz/2.-1U)/2., test_tools::tolerance(1e-4));
		mp.computeSdevEdge(invHalfD255, sdHorizEdgeMask);
		BOOST_REQUIRE(mp.sdevEdge);
		BOOST_TEST(*mp.sdevEdge == 0., test_tools::tolerance(1e-4));
		mp.reset();
		mp.computeSdevEdge(invHalfD255, sdVertEdgeMask);
		BOOST_REQUIRE(mp.sdevEdge);
		BOOST_TEST(*mp.sdevEdge == 0., test_tools::tolerance(1e-4));

		// Testing the mentioned glyph on a patch half 85, half 170=2*85 =>
		// Adapting glyph's fg & bg to match the patch => the glyph looses contrast =>
		// Its mass-center will be ( (sz-1)/2 ,  (5*sz-6)/12 )
		mp.reset();
		Mat twoBandsD255 = emptyD.clone();
		twoBandsD255.rowRange(0, sz/2) = 170.; twoBandsD255.rowRange(sz/2, sz) = 85.;
		mp.computeMcApproxSym(twoBandsD255, sdHorizEdgeMask, cd);
		BOOST_REQUIRE(mp.mcGlyph);
		BOOST_TEST(mp.mcGlyph->x == cd.patchCenter.x, test_tools::tolerance(1e-4));
		BOOST_TEST(mp.mcGlyph->y == (5*sz-6)/12., test_tools::tolerance(1e-4));
		mp.computeSdevEdge(twoBandsD255, sdHorizEdgeMask);
		BOOST_REQUIRE(mp.sdevEdge);
		BOOST_TEST(*mp.sdevEdge == 0., test_tools::tolerance(1e-4));
		mp.reset();
		mp.computeSdevEdge(twoBandsD255, sdVertEdgeMask);
		BOOST_REQUIRE(mp.sdevEdge);
		BOOST_TEST(*mp.sdevEdge == 0., test_tools::tolerance(1e-4));

		// Testing on a patch with uniform rows of values gradually growing from 0 to sz-1
		Mat szBandsD255 = emptyD.clone();
		for(unsigned i = 0U; i<sz; ++i)
			szBandsD255.row(i) = i;
		double expectedFgAndSdevHorEdge = (sz-2)/4., expectedSdev = 0.;
		for(unsigned i = 0U; i<sz/2; ++i) {
			double diff = i - expectedFgAndSdevHorEdge;
			expectedSdev += diff*diff;
		}
		expectedSdev = sqrt(expectedSdev / (sz/2));
		mp.reset(false);
		mp.computeMcPatch(szBandsD255, cd);
		BOOST_REQUIRE(mp.mcPatch);
		BOOST_TEST(mp.mcPatch->x == cd.patchCenter.x, test_tools::tolerance(1e-4));
		BOOST_TEST(mp.mcPatch->y == (2*sz-1)/3., test_tools::tolerance(1e-4));
		mp.computeFg(szBandsD255, sdHorizEdgeMask);
		BOOST_REQUIRE(mp.fg);
		BOOST_TEST(*mp.fg == expectedFgAndSdevHorEdge, test_tools::tolerance(1e-4));
		mp.computeBg(szBandsD255, sdHorizEdgeMask);
		BOOST_REQUIRE(mp.bg);
		BOOST_TEST(*mp.bg == (3*sz-2)/4., test_tools::tolerance(1e-4));
		mp.computeSdevFg(szBandsD255, sdHorizEdgeMask);
		BOOST_REQUIRE(mp.sdevFg);
		BOOST_TEST(*mp.sdevFg == expectedSdev, test_tools::tolerance(1e-4));
		mp.computeSdevBg(szBandsD255, sdHorizEdgeMask);
		BOOST_REQUIRE(mp.sdevBg);
		BOOST_TEST(*mp.sdevBg == expectedSdev, test_tools::tolerance(1e-4));
		mp.computeMcApproxSym(szBandsD255, sdHorizEdgeMask, cd);
		BOOST_REQUIRE(mp.mcGlyph);
		BOOST_TEST(mp.mcGlyph->x == cd.patchCenter.x, test_tools::tolerance(1e-4));
		BOOST_TEST(mp.mcGlyph->y == (*mp.fg * *mp.fg + *mp.bg * *mp.bg)/(sz-1.), test_tools::tolerance(1e-4));
		mp.computeSdevEdge(szBandsD255, sdHorizEdgeMask);
		BOOST_REQUIRE(mp.sdevEdge);
		BOOST_TEST(*mp.sdevEdge == expectedFgAndSdevHorEdge, test_tools::tolerance(1e-4));
		mp.reset();
		mp.computeSdevEdge(szBandsD255, sdVertEdgeMask);
		BOOST_REQUIRE(mp.sdevEdge);
		BOOST_TEST(*mp.sdevEdge == expectedSdev, test_tools::tolerance(1e-4));
	}

	BOOST_AUTO_TEST_CASE(MatchEngine_CheckFgAspect) {
		BOOST_TEST_MESSAGE("Running MatchEngine_CheckFgAspect ...");

		const Config cfg(50U, // symbol size (use even values, as some tests set half of the patch)
						 1.); // fg correctness coefficient set to 1
		const unsigned sz = cfg.getFontSz();
		const CachedData cd;
		const FgMatch fm(cd, cfg);
		const unsigned char valRand = (unsigned char)(randUnifUint() & 0xFF);
		MatchParams mp;

		// Using a symbol with a diagonal foreground mask
		const Mat diagFgMask = Mat::diag(Mat(1, sz, CV_8UC1, Scalar(255U)));

		SymData sd(NOT_RELEVANT_UL, // symbol code (not relevant here)
				   NOT_RELEVANT_D, NOT_RELEVANT_D, // min and diff between min..max, each in range 0..1 (not relevant here)
				   NOT_RELEVANT_D, // pixelSum (not relevant here)
				   NOT_RELEVANT_POINT, // mc sym for original fg & bg (not relevant here)
				   SymData::MatArray { { // symAndMasks
						   diagFgMask, // FG_MASK_IDX
						   NOT_RELEVANT_MAT, // BG_MASK_IDX (not relevant here)
						   NOT_RELEVANT_MAT, // EDGE_MASK_IDX (not relevant here)
						   NOT_RELEVANT_MAT, // NEG_GLYPH_IDX (not relevant here)
						   NOT_RELEVANT_MAT // GROUNDED_GLYPH_IDX (not relevant here)
					   } });

		// Testing on a uniform patch
		Mat patchD255(sz, sz, CV_64FC1, Scalar((double)valRand));
		double res = fm.assessMatch(patchD255, sd, mp);
		BOOST_REQUIRE(mp.fg && mp.sdevFg);
		BOOST_TEST(*mp.fg == (double)valRand, test_tools::tolerance(1e-4));
		BOOST_TEST(*mp.sdevFg == 0., test_tools::tolerance(1e-4));
		BOOST_TEST(res == 1., test_tools::tolerance(1e-4));

		// Testing on a patch with upper half empty and an uniform lower half (valRand)
		mp.reset();
		patchD255.rowRange(0, sz/2) = 0.;
		res = fm.assessMatch(patchD255, sd, mp);
		BOOST_REQUIRE(mp.fg && mp.sdevFg);
		BOOST_TEST(*mp.fg == valRand/2., test_tools::tolerance(1e-4));
		BOOST_TEST(*mp.sdevFg == valRand/2., test_tools::tolerance(1e-4));
		BOOST_TEST(res == 1.-valRand/255., test_tools::tolerance(1e-4));

		// Testing on a patch with uniform rows, but gradually brighter, from top to bottom
		mp.reset();
		double expectedMiu = (sz-1U)/2., expectedSdev = 0., aux;
		for(unsigned i = 0U; i<sz; ++i) {
			patchD255.row(i) = (double)i;
			aux = i-expectedMiu;
			expectedSdev += aux*aux;
		}
		expectedSdev = sqrt(expectedSdev/sz);
		res = fm.assessMatch(patchD255, sd, mp);
		BOOST_REQUIRE(mp.fg && mp.sdevFg);
		BOOST_TEST(*mp.fg == expectedMiu, test_tools::tolerance(1e-4));
		BOOST_TEST(*mp.sdevFg == expectedSdev, test_tools::tolerance(1e-4));
		BOOST_TEST(res == 1.-expectedSdev/127.5, test_tools::tolerance(1e-4));
	}

	BOOST_AUTO_TEST_CASE(MatchEngine_CheckEdgeAspect) {
		BOOST_TEST_MESSAGE("Running MatchEngine_CheckEdgeAspect ...");

		const Config cfg(50U, // symbol size (use even >=4 values, as some tests below need that)
						 0, // fg correctness coefficient (not relevant here)
						 1); // edge correctness coefficient set to 1
		const unsigned sz = cfg.getFontSz(), area = sz*sz;
		const CachedData cd;
		const EdgeMatch em(cd, cfg);
		const unsigned char valRand = (unsigned char)(randUnifUint() & 0xFF);
		MatchParams mp;

		// Using a symbol with a diagonal foreground mask
		const Mat diagFgMask = Mat::diag(Mat(1, sz, CV_8UC1, Scalar(255U)));

		// Using a symbol with an edge mask formed by 2 neighbor diagonals of the main diagonal
		Mat sideDiagsEdgeMask = Mat::zeros(sz, sz, CV_8UC1);
		sideDiagsEdgeMask.diag(1) = 255U; // 2nd diagonal lower half
		sideDiagsEdgeMask.diag(-1) = 255U; // 2nd diagonal upper half
		const unsigned cnzEdge = 2U*(sz - 1U);
		BOOST_REQUIRE(countNonZero(sideDiagsEdgeMask) == cnzEdge);
		
		// Using a symbol with a background mask full except the main 3 diagonals
		Mat allBut3DiagsBgMask = Mat(sz, sz, CV_8UC1, Scalar(255U));
		allBut3DiagsBgMask.diag() = 0U;
		allBut3DiagsBgMask.diag(1) = 0U; // 2nd diagonal lower half
		allBut3DiagsBgMask.diag(-1) = 0U; // 2nd diagonal upper half
		const unsigned cnzBg = area - 3U*sz + 2U;
		BOOST_REQUIRE(countNonZero(allBut3DiagsBgMask) == cnzBg);
		
		Mat groundedGlyph = Mat::zeros(sz, sz, CV_8UC1); // bg will stay 0
		const unsigned char edgeLevel = 125U,
							maxGlyph = edgeLevel<<1; // 250
		add(groundedGlyph, maxGlyph, groundedGlyph, diagFgMask); // set fg
		add(groundedGlyph, edgeLevel, groundedGlyph, sideDiagsEdgeMask); // set edges
		groundedGlyph.convertTo(groundedGlyph, CV_64FC1, 1./255);

		SymData sd(NOT_RELEVANT_UL, // symbol code (not relevant here)
				   NOT_RELEVANT_D, // min brightness value 0..1 range (not relevant here)
				   maxGlyph/255., // diff between min..max, each in range 0..1
				   NOT_RELEVANT_D, // pixelSum (not relevant here)
				   NOT_RELEVANT_POINT, // mc sym for original fg & bg (not relevant here)
				   SymData::MatArray { { // symAndMasks
						   diagFgMask, // FG_MASK_IDX
						   allBut3DiagsBgMask, // BG_MASK_IDX
						   sideDiagsEdgeMask, // EDGE_MASK_IDX
						   NOT_RELEVANT_MAT, // NEG_GLYPH_IDX (not relevant here)
						   groundedGlyph // GROUNDED_GLYPH_IDX
					   } });

		// Testing on a uniform patch
		Mat patchD255(sz, sz, CV_64FC1, Scalar((double)valRand));
		double res = em.assessMatch(patchD255, sd, mp);
		BOOST_REQUIRE(mp.fg && mp.bg && mp.sdevEdge);
		BOOST_TEST(*mp.fg == (double)valRand, test_tools::tolerance(1e-4));
		BOOST_TEST(*mp.bg == (double)valRand, test_tools::tolerance(1e-4));
		BOOST_TEST(*mp.sdevEdge == 0., test_tools::tolerance(1e-4));
		BOOST_TEST(res == 1., test_tools::tolerance(1e-4));

		// Testing on a patch with upper half empty and an uniform lower half (valRand)
		mp.reset();
		patchD255.rowRange(0, sz/2) = 0.;
		res = em.assessMatch(patchD255, sd, mp);
		BOOST_REQUIRE(mp.fg && mp.bg && mp.sdevEdge);
		BOOST_TEST(*mp.fg == valRand/2., test_tools::tolerance(1e-4));
		BOOST_TEST(*mp.bg == valRand/2., test_tools::tolerance(1e-4));
		BOOST_TEST(*mp.sdevEdge == valRand/2., test_tools::tolerance(1e-4));
		BOOST_TEST(res == 1.-valRand/255., test_tools::tolerance(1e-4));

		// Testing on a patch with uniform rows, but gradually brighter, from top to bottom
		mp.reset();
		double expectedMiu = (sz-1U)/2.,
			expectedSdev = 0., aux;
		for(unsigned i = 0U; i<sz; ++i) {
			patchD255.row(i) = (double)i;
			aux = i-expectedMiu;
			expectedSdev += aux*aux;
		}
		expectedSdev *= 2.;
		expectedSdev -= 2 * expectedMiu*expectedMiu;
		expectedSdev = sqrt(expectedSdev / cnzEdge);
		res = em.assessMatch(patchD255, sd, mp);
		BOOST_REQUIRE(mp.fg && mp.bg && mp.sdevEdge);
		BOOST_TEST(*mp.fg == expectedMiu, test_tools::tolerance(1e-4));
		BOOST_TEST(*mp.bg == expectedMiu, test_tools::tolerance(1e-4));
		BOOST_TEST(*mp.sdevEdge == expectedSdev, test_tools::tolerance(1e-4));
		BOOST_TEST(res == 1.-expectedSdev/127.5, test_tools::tolerance(1e-4));

		// Testing on an uniform lower triangular patch
		mp.reset();
		patchD255 = 0.;
		double expectedFg = valRand, expectedBg = valRand/2.,
			expectedSdevEdge = valRand * sqrt(5) / 4.;
		for(unsigned i = 0U; i<sz; ++i)
			patchD255.diag(-((int)i)) = valRand; // i-th lower diagonal set on valRand
		BOOST_REQUIRE(countNonZero(patchD255) == (area + sz)/2);
		res = em.assessMatch(patchD255, sd, mp);
		BOOST_REQUIRE(mp.fg && mp.bg && mp.sdevEdge);
		BOOST_TEST(*mp.fg == expectedFg, test_tools::tolerance(1e-4));
		BOOST_TEST(*mp.bg == expectedBg, test_tools::tolerance(1e-4));
		BOOST_TEST(*mp.sdevEdge == expectedSdevEdge, test_tools::tolerance(1e-4));
		BOOST_TEST(res == 1.-expectedSdevEdge/127.5, test_tools::tolerance(1e-4));
	}

	BOOST_AUTO_TEST_CASE(MatchEngine_CheckBgAspect) {
		BOOST_TEST_MESSAGE("Running MatchEngine_CheckBgAspect ...");

		const Config cfg(50U, // symbol size (use even >=4 values, as some tests below need that)
						 0, // fg correctness coefficient (not relevant here)
						 0, // edge correctness coefficient (not relevant here)
						 1.); // bg correctness coefficient set to 1
		const unsigned sz = cfg.getFontSz(), area = sz*sz;
		const CachedData cd;
		const BgMatch bm(cd, cfg);
		const unsigned char valRand = (unsigned char)(randUnifUint() & 0xFF);
		MatchParams mp;

		// Using a symbol with a background mask full except the main 3 diagonals
		Mat allBut3DiagsBgMask = Mat(sz, sz, CV_8UC1, Scalar(255U));
		allBut3DiagsBgMask.diag() = 0U;
		allBut3DiagsBgMask.diag(1) = 0U; // 2nd diagonal lower half
		allBut3DiagsBgMask.diag(-1) = 0U; // 2nd diagonal upper half
		const unsigned cnz = area - 3U*sz + 2U;
		BOOST_REQUIRE(countNonZero(allBut3DiagsBgMask) == cnz);
		SymData sd(NOT_RELEVANT_UL, // symbol code (not relevant here)
				   NOT_RELEVANT_D, NOT_RELEVANT_D, // min and diff between min..max, each in range 0..1 (not relevant here)
				   NOT_RELEVANT_D, // pixelSum (not relevant here)
				   NOT_RELEVANT_POINT, // mc sym for original fg & bg (not relevant here)
				   SymData::MatArray { { // symAndMasks
						   NOT_RELEVANT_MAT, // FG_MASK_IDX (not relevant here)
						   allBut3DiagsBgMask, // BG_MASK_IDX
						   NOT_RELEVANT_MAT, // EDGE_MASK_IDX (not relevant here)
						   NOT_RELEVANT_MAT, // NEG_GLYPH_IDX (not relevant here)
						   NOT_RELEVANT_MAT // GROUNDED_GLYPH_IDX (not relevant here)
					   } });

		// Testing on a uniform patch
		Mat patchD255(sz, sz, CV_64FC1, Scalar((double)valRand));
		double res = bm.assessMatch(patchD255, sd, mp);
		BOOST_REQUIRE(mp.bg && mp.sdevBg);
		BOOST_TEST(*mp.bg == (double)valRand, test_tools::tolerance(1e-4));
		BOOST_TEST(*mp.sdevBg == 0., test_tools::tolerance(1e-4));
		BOOST_TEST(res == 1., test_tools::tolerance(1e-4));

		// Testing on a patch with upper half empty and an uniform lower half (valRand)
		mp.reset();
		patchD255.rowRange(0, sz/2) = 0.;
		res = bm.assessMatch(patchD255, sd, mp);
		BOOST_REQUIRE(mp.bg && mp.sdevBg);
		BOOST_TEST(*mp.bg == valRand/2., test_tools::tolerance(1e-4));
		BOOST_TEST(*mp.sdevBg == valRand/2., test_tools::tolerance(1e-4));
		BOOST_TEST(res == 1.-valRand/255., test_tools::tolerance(1e-4));

		// Testing on a patch with uniform rows, but gradually brighter, from top to bottom
		mp.reset();
		double expectedMiu = (sz-1U)/2.,
			expectedSdev = 0., aux;
		for(unsigned i = 0U; i<sz; ++i) {
			patchD255.row(i) = (double)i;
			aux = i-expectedMiu;
			expectedSdev += aux*aux;
		}
		expectedSdev *= sz - 3.;
		expectedSdev += 2 * expectedMiu*expectedMiu;
		expectedSdev = sqrt(expectedSdev / cnz);
		res = bm.assessMatch(patchD255, sd, mp);
		BOOST_REQUIRE(mp.bg && mp.sdevBg);
		BOOST_TEST(*mp.bg == expectedMiu, test_tools::tolerance(1e-4));
		BOOST_TEST(*mp.sdevBg == expectedSdev, test_tools::tolerance(1e-4));
		BOOST_TEST(res == 1.-expectedSdev/127.5, test_tools::tolerance(1e-4));
	}

	BOOST_AUTO_TEST_CASE(MatchEngine_CheckContrastAspect) {
		BOOST_TEST_MESSAGE("Running MatchEngine_CheckContrastAspect ...");

		const Config cfg(50U, // symbol size (use even >=4 values, as some tests below need that)
						 0, // fg correctness coefficient (not relevant here)
						 0, // edge correctness coefficient (not relevant here)
						 0, // bg correctness coefficient (not relevant here)
						 1.); // contrast coefficient set to 1
		const unsigned sz = cfg.getFontSz(), area = sz*sz;
		const CachedData cd;
		const BetterContrast bc(cd, cfg);
		const unsigned char valRand = (unsigned char)(randUnifUint() & 0xFF);
		MatchParams mp;

		// Using a symbol with a diagonal foreground mask
		const Mat diagFgMask = Mat::diag(Mat(1, sz, CV_8UC1, Scalar(255U)));

		// Using a symbol with a background mask full except the main 3 diagonals
		Mat allBut3DiagsBgMask = Mat(sz, sz, CV_8UC1, Scalar(255U));
		allBut3DiagsBgMask.diag() = 0U;
		allBut3DiagsBgMask.diag(1) = 0U; // 2nd diagonal lower half
		allBut3DiagsBgMask.diag(-1) = 0U; // 2nd diagonal upper half
		const unsigned cnz = area - 3U*sz + 2U;
		BOOST_REQUIRE(countNonZero(allBut3DiagsBgMask) == cnz);
		SymData sd(NOT_RELEVANT_UL, // symbol code (not relevant here)
				   NOT_RELEVANT_D, NOT_RELEVANT_D, // min and diff between min..max, each in range 0..1 (not relevant here)
				   NOT_RELEVANT_D, // pixelSum (not relevant here)
				   NOT_RELEVANT_POINT, // mc sym for original fg & bg (not relevant here)
				   SymData::MatArray { { // symAndMasks
						   diagFgMask, // FG_MASK_IDX
						   allBut3DiagsBgMask, // BG_MASK_IDX
						   NOT_RELEVANT_MAT, // EDGE_MASK_IDX (not relevant here)
						   NOT_RELEVANT_MAT, // NEG_GLYPH_IDX (not relevant here)
						   NOT_RELEVANT_MAT // GROUNDED_GLYPH_IDX (not relevant here)
					   } });

		// Testing on a uniform patch
		Mat patchD255(sz, sz, CV_64FC1, Scalar((double)valRand));
		double res = bc.assessMatch(patchD255, sd, mp);
		BOOST_REQUIRE(mp.fg && mp.bg);
		BOOST_TEST(*mp.fg == (double)valRand, test_tools::tolerance(1e-4));
		BOOST_TEST(*mp.bg == (double)valRand, test_tools::tolerance(1e-4));
		BOOST_TEST(res == 0., test_tools::tolerance(1e-4));

		// Testing on a diagonal patch with max contrast
		mp.reset();
		patchD255 = Mat::diag(Mat(1, sz, CV_64FC1, Scalar(255.)));
		res = bc.assessMatch(patchD255, sd, mp);
		BOOST_REQUIRE(mp.fg && mp.bg);
		BOOST_TEST(*mp.fg == 255., test_tools::tolerance(1e-4));
		BOOST_TEST(*mp.bg == 0., test_tools::tolerance(1e-4));
		BOOST_TEST(res == 1., test_tools::tolerance(1e-4));

		// Testing on a diagonal patch with half contrast
		mp.reset();
		patchD255 = Mat::diag(Mat(1, sz, CV_64FC1, Scalar(127.5)));
		res = bc.assessMatch(patchD255, sd, mp);
		BOOST_REQUIRE(mp.fg && mp.bg);
		BOOST_TEST(*mp.fg == 127.5, test_tools::tolerance(1e-4));
		BOOST_TEST(*mp.bg == 0., test_tools::tolerance(1e-4));
		BOOST_TEST(res == .5, test_tools::tolerance(1e-4));
	}

	BOOST_AUTO_TEST_CASE(MatchEngine_CheckGravitationalSmoothnessAspect) {
		BOOST_TEST_MESSAGE("Running MatchEngine_CheckGravitationalSmoothnessAspect ...");

		Config cfg(50U, // symbol size
						 0, // fg correctness coefficient (not relevant here)
						 0, // edge correctness coefficient (not relevant here)
						 0, // bg correctness coefficient (not relevant here)
						 0, // contrast coefficient (not relevant here)
						 1.); // gravitational smoothness coefficient set to 1
		const unsigned sz = cfg.getFontSz(), area = sz*sz, sz_1 = sz-1U;
		CachedData cd; cd.useNewSymSize(sz);
		const GravitationalSmoothness gs(cd, cfg);
		MatchParams mp;

		// Checking a symbol that has a single 255 pixel in bottom right corner
		double pixelSum = 1.; // a single pixel set to max
		Point2d origMcSym(sz_1, sz_1);
		Mat fgMask = Mat::zeros(sz, sz, CV_8UC1), bgMask(sz, sz, CV_8UC1, Scalar(255U));
		fgMask.at<unsigned char>(sz_1, sz_1) = 255U;
		bgMask.at<unsigned char>(sz_1, sz_1) = 0U;

		SymData sd(NOT_RELEVANT_UL, // symbol code (not relevant here)
				   NOT_RELEVANT_D, NOT_RELEVANT_D, // min and diff between min..max, each in range 0..1 (not relevant here)
				   pixelSum,
				   origMcSym,
				   SymData::MatArray { { // symAndMasks
						   fgMask, // FG_MASK_IDX
						   bgMask, // BG_MASK_IDX
						   NOT_RELEVANT_MAT, // EDGE_MASK_IDX (not relevant here)
						   NOT_RELEVANT_MAT, // NEG_GLYPH_IDX (not relevant here)
						   NOT_RELEVANT_MAT // GROUNDED_GLYPH_IDX (not relevant here)
					   } });

		// Using a patch with a single 255 pixel in top left corner
		Mat patchD255 = Mat::zeros(sz, sz, CV_64FC1);
		patchD255.at<double>(0, 0) = 255.;
		double res = gs.assessMatch(patchD255, sd, mp);
		BOOST_REQUIRE(mp.fg && mp.bg && mp.glyphWeight && mp.mcGlyph && mp.mcPatch);
		BOOST_TEST(*mp.fg == 0., test_tools::tolerance(1e-4));
		BOOST_TEST(*mp.bg == 255./(area-1), test_tools::tolerance(1e-4));
		BOOST_TEST(*mp.glyphWeight == 1./area, test_tools::tolerance(1e-8));

		// Symbol's mc migrated diagonally from bottom-right corner to above the center of patch.
		// Migration occurred due to the new fg & bg of the symbol
		BOOST_TEST(mp.mcGlyph->x == cd.patchCenter.x - .5/(sz+1), test_tools::tolerance(1e-8));
		BOOST_TEST(mp.mcGlyph->y == cd.patchCenter.y - .5/(sz+1), test_tools::tolerance(1e-8));
		BOOST_TEST(mp.mcPatch->x == 0., test_tools::tolerance(1e-4));
		BOOST_TEST(mp.mcPatch->y == 0., test_tools::tolerance(1e-4));
		BOOST_TEST(res == 1. + (cd.preferredMaxMcDist - sqrt(2)*.5*(sz_1 - 1./(sz+1))) / cd.complPrefMaxMcDist, test_tools::tolerance(1e-8));


		// Using a patch with the middle pixels pair on the top row on 255
		// Patch mc is at half width on top row.
		mp.reset(false);
		patchD255 = Mat::zeros(sz, sz, CV_64FC1);
		Mat(patchD255, Rect(sz/2U-1U, 0, 2, 1)) = Scalar(255.);
		BOOST_REQUIRE(countNonZero(patchD255) == 2);
		res = gs.assessMatch(patchD255, sd, mp);
		BOOST_REQUIRE(mp.fg && mp.bg && mp.glyphWeight && mp.mcGlyph && mp.mcPatch);
		BOOST_TEST(*mp.fg == 0., test_tools::tolerance(1e-4));
		BOOST_TEST(*mp.bg == 2*255./(area-1), test_tools::tolerance(1e-4));
		BOOST_TEST(*mp.glyphWeight == 1./area, test_tools::tolerance(1e-8));

		// Symbol's mc migrated diagonally from bottom-right corner to above the center of patch.
		// Migration occurred due to the new fg & bg of the symbol
		BOOST_TEST(mp.mcGlyph->x == cd.patchCenter.x - .5/(sz+1), test_tools::tolerance(1e-8));
		BOOST_TEST(mp.mcGlyph->y == cd.patchCenter.y - .5/(sz+1), test_tools::tolerance(1e-8));
		BOOST_TEST(mp.mcPatch->x == cd.patchCenter.x, test_tools::tolerance(1e-4));
		BOOST_TEST(mp.mcPatch->y == 0., test_tools::tolerance(1e-4));
		double dx = .5/(sz+1), dy = .5*(sz_1 - 1/(sz+1.));
		BOOST_TEST(res == 1. + (cd.preferredMaxMcDist - sqrt(dx*dx + dy*dy)) /
			cd.complPrefMaxMcDist, test_tools::tolerance(1e-8));


		// Using a patch with the last pixel on the top row on 255
		mp.reset(false);
		patchD255 = Mat::zeros(sz, sz, CV_64FC1);
		patchD255.at<double>(0, sz_1) = 255.;
		res = gs.assessMatch(patchD255, sd, mp);
		BOOST_REQUIRE(mp.fg && mp.bg && mp.glyphWeight && mp.mcGlyph && mp.mcPatch);
		BOOST_TEST(*mp.fg == 0., test_tools::tolerance(1e-4));
		BOOST_TEST(*mp.bg == 255./(area-1), test_tools::tolerance(1e-4));
		BOOST_TEST(*mp.glyphWeight == 1./area, test_tools::tolerance(1e-8));

		// Symbol's mc migrated diagonally from bottom-right corner to above the center of patch.
		// Migration occurred due to the new fg & bg of the symbol
		BOOST_TEST(mp.mcGlyph->x == cd.patchCenter.x - .5/(sz+1), test_tools::tolerance(1e-8));
		BOOST_TEST(mp.mcGlyph->y == cd.patchCenter.y - .5/(sz+1), test_tools::tolerance(1e-8));
		BOOST_TEST(mp.mcPatch->x == (double)sz_1, test_tools::tolerance(1e-4));
		BOOST_TEST(mp.mcPatch->y == 0., test_tools::tolerance(1e-4));
		dx = .5*(sz_1 + 1/(sz+1.)); dy = .5*(sz_1 - 1/(sz+1.));
		BOOST_TEST(res == 1. + (cd.preferredMaxMcDist - sqrt(dx*dx + dy*dy)) /
				   cd.complPrefMaxMcDist, test_tools::tolerance(1e-8));
	}

	BOOST_AUTO_TEST_CASE(MatchEngine_CheckDirectionalSmoothnessAspect) {
		BOOST_TEST_MESSAGE("Running MatchEngine_CheckDirectionalSmoothnessAspect ...");

		Config cfg(50U, // symbol size (Use even values >=4 to allow creating some scenarios below)
				   0, // fg correctness coefficient (not relevant here)
				   0, // edge correctness coefficient (not relevant here)
				   0, // bg correctness coefficient (not relevant here)
				   0, // contrast coefficient (not relevant here)
				   0, // gravitational smoothness coefficient (not relevant here)
				   1.); // directional smoothness coefficient set to 1
		const unsigned sz = cfg.getFontSz(), area = sz*sz, sz_1 = sz-1U;
		CachedData cd; cd.useNewSymSize(sz);
		const DirectionalSmoothness ds(cd, cfg);
		MatchParams mp;

		// Checking a symbol that has a single 255 pixel in bottom right corner
		double pixelSum = 1.; // a single pixel set to max
		Point2d origMcSym(sz_1, sz_1);
		Mat fgMask = Mat::zeros(sz, sz, CV_8UC1), bgMask(sz, sz, CV_8UC1, Scalar(255U));
		fgMask.at<unsigned char>(sz_1, sz_1) = 255U;
		bgMask.at<unsigned char>(sz_1, sz_1) = 0U;

		SymData sd(NOT_RELEVANT_UL, // symbol code (not relevant here)
				   NOT_RELEVANT_D, NOT_RELEVANT_D, // min and diff between min..max, each in range 0..1 (not relevant here)
				   pixelSum,
				   origMcSym,
				   SymData::MatArray { { // symAndMasks
						   fgMask, // FG_MASK_IDX
						   bgMask, // BG_MASK_IDX
						   NOT_RELEVANT_MAT, // EDGE_MASK_IDX (not relevant here)
						   NOT_RELEVANT_MAT, // NEG_GLYPH_IDX (not relevant here)
						   NOT_RELEVANT_MAT // GROUNDED_GLYPH_IDX (not relevant here)
					   } });

		// Using a patch with a single 255 pixel in top left corner
		// Same as 1st scenario from Gravitational Smoothness
		Mat patchD255 = Mat::zeros(sz, sz, CV_64FC1);
		patchD255.at<double>(0, 0) = 255.;
		double res = ds.assessMatch(patchD255, sd, mp);
		BOOST_REQUIRE(mp.fg && mp.bg && mp.glyphWeight && mp.mcGlyph && mp.mcPatch);
		// Symbol's mc migrated diagonally from bottom-right corner to above the center of patch.
		// Migration occurred due to the new fg & bg of the symbol
		BOOST_TEST(mp.mcGlyph->x == cd.patchCenter.x - .5/(sz+1), test_tools::tolerance(1e-8));
		BOOST_TEST(mp.mcGlyph->y == cd.patchCenter.y - .5/(sz+1), test_tools::tolerance(1e-8));
		BOOST_TEST(mp.mcPatch->x == 0., test_tools::tolerance(1e-4));
		BOOST_TEST(mp.mcPatch->y == 0., test_tools::tolerance(1e-4));
		BOOST_TEST(res == 2*(2-sqrt(2)), test_tools::tolerance(1e-8)); // angle = 0 => cos = 1

		// Using a patch with the middle pixels pair on the top row on 255
		// Patch mc is at half width on top row.
		// Same as 2nd scenario from Gravitational Smoothness
		mp.reset(false);
		patchD255 = Mat::zeros(sz, sz, CV_64FC1);
		Mat(patchD255, Rect(sz/2U-1U, 0, 2, 1)) = Scalar(255.);
		res = ds.assessMatch(patchD255, sd, mp);
		BOOST_REQUIRE(mp.fg && mp.bg && mp.glyphWeight && mp.mcGlyph && mp.mcPatch);
		// Symbol's mc migrated diagonally from bottom-right corner to above the center of patch.
		// Migration occurred due to the new fg & bg of the symbol
		BOOST_TEST(mp.mcGlyph->x == cd.patchCenter.x - .5/(sz+1), test_tools::tolerance(1e-8));
		BOOST_TEST(mp.mcGlyph->y == cd.patchCenter.y - .5/(sz+1), test_tools::tolerance(1e-8));
		BOOST_TEST(mp.mcPatch->x == cd.patchCenter.x, test_tools::tolerance(1e-4));
		BOOST_TEST(mp.mcPatch->y == 0., test_tools::tolerance(1e-4));
		BOOST_TEST(res == 1., test_tools::tolerance(1e-8)); // angle = 45 => cos = sqrt(2)/2

		// Using a patch with the last pixel on the top row on 255
		// Same as 3rd scenario from Gravitational Smoothness
		mp.reset(false);
		patchD255 = Mat::zeros(sz, sz, CV_64FC1);
		patchD255.at<double>(0, sz_1) = 255.;
		res = ds.assessMatch(patchD255, sd, mp);
		BOOST_REQUIRE(mp.fg && mp.bg && mp.glyphWeight && mp.mcGlyph && mp.mcPatch);
		// Symbol's mc migrated diagonally from bottom-right corner to above the center of patch.
		// Migration occurred due to the new fg & bg of the symbol
		BOOST_TEST(mp.mcGlyph->x == cd.patchCenter.x - .5/(sz+1), test_tools::tolerance(1e-8));
		BOOST_TEST(mp.mcGlyph->y == cd.patchCenter.y - .5/(sz+1), test_tools::tolerance(1e-8));
		BOOST_TEST(mp.mcPatch->x == (double)sz_1, test_tools::tolerance(1e-4));
		BOOST_TEST(mp.mcPatch->y == 0., test_tools::tolerance(1e-4));
		BOOST_TEST(res == 2-sqrt(2), test_tools::tolerance(1e-8)); // angle is 90 => cos = 0
	}

	BOOST_AUTO_TEST_CASE(MatchEngine_CheckLargerSymAspect) {
		BOOST_TEST_MESSAGE("Running MatchEngine_CheckLargerSymAspect ...");

		const Config cfg(50U, // symbol size
						 0, // fg correctness coefficient (not relevant here)
						 0, // edge correctness coefficient (not relevant here)
						 0, // bg correctness coefficient (not relevant here)
						 0, // contrast coefficient (not relevant here)
						 0, // mc offset coefficient (not relevant here)
						 0, // mc angle coefficient (not relevant here)
						 1.); // symbol weight coefficient set to 1
		const unsigned sz = cfg.getFontSz(), area = sz*sz;
		CachedData cd;
		cd.smallGlyphsCoverage = .1; // large glyphs need to cover more than 10% of their box
		cd.sz2 = area;
		const LargerSym ls(cd, cfg);
		MatchParams mp;

		SymData sd(NOT_RELEVANT_UL, // symbol code (not relevant here)
				   NOT_RELEVANT_D, NOT_RELEVANT_D, // min and diff between min..max, each in range 0..1 (not relevant here)
				   0., // pixelSum (INITIALLY, AN EMPTY SYMBOL IS CONSIDERED)
				   NOT_RELEVANT_POINT, // mc sym for original fg & bg (not relevant here)
				   SymData::MatArray { { // symAndMasks
						   NOT_RELEVANT_MAT, // FG_MASK_IDX (not relevant here)
						   NOT_RELEVANT_MAT, // BG_MASK_IDX (not relevant here)
						   NOT_RELEVANT_MAT, // EDGE_MASK_IDX (not relevant here)
						   NOT_RELEVANT_MAT, // NEG_GLYPH_IDX (not relevant here)
						   NOT_RELEVANT_MAT // GROUNDED_GLYPH_IDX (not relevant here)
					   } });

		// Testing with an empty symbol (sd.pixelSum == 0)
		double res = ls.assessMatch(NOT_RELEVANT_MAT, sd, mp);
		BOOST_REQUIRE(mp.glyphWeight);
		BOOST_TEST(*mp.glyphWeight == 0., test_tools::tolerance(1e-4));
		BOOST_TEST(res == 1.-cd.smallGlyphsCoverage, test_tools::tolerance(1e-4));

		// Testing with a symbol that just enters the 'large symbols' category
		mp.reset(); *(const_cast<double*>(&sd.pixelSum)) = area * cd.smallGlyphsCoverage;
		res = ls.assessMatch(NOT_RELEVANT_MAT, sd, mp);
		BOOST_REQUIRE(mp.glyphWeight);
		BOOST_TEST(*mp.glyphWeight == cd.smallGlyphsCoverage, test_tools::tolerance(1e-4));
		BOOST_TEST(res == 1., test_tools::tolerance(1e-4));

		// Testing with largest possible symbol
		mp.reset(); *(const_cast<double*>(&sd.pixelSum)) = area;
		res = ls.assessMatch(NOT_RELEVANT_MAT, sd, mp);
		BOOST_REQUIRE(mp.glyphWeight);
		BOOST_TEST(*mp.glyphWeight == 1., test_tools::tolerance(1e-4));
		BOOST_TEST(res == 2.-cd.smallGlyphsCoverage, test_tools::tolerance(1e-4));
	}

	BOOST_AUTO_TEST_CASE(MatchEngine_CheckAlteredCmapUsingStdDev) {
		BOOST_TEST_MESSAGE("Running MatchEngine_CheckAlteredCmapUsingStdDev ...");
		Config cfg(10U, 1., 1., 1.);
		const unsigned sz = cfg.getFontSz();
		::Controller c(cfg);
		::MatchEngine &me = c.getMatchEngine(cfg);

		// Courier Bold Unicode > 2800 glyphs; There are 2 almost identical COMMA-s and QUOTE-s.
		// Can't identify them exactly using std. dev. for fg, bg and edges, which all appear 0.
		// In both cases, the scores are 1.00000000000000000 (17 decimals!!) 
//  	BOOST_REQUIRE_NO_THROW(c.newFontFamily("C:\\Windows\\Fonts\\courbd.ttf"));

		// Envy Code R Unicode > 600 glyphs
// 		BOOST_REQUIRE_NO_THROW(c.newFontFamily("C:\\Windows\\Fonts\\Envy Code R Bold.ttf"));
// 		BOOST_REQUIRE_NO_THROW(c.newFontEncoding("UNICODE"));

		// Bp Mono Bold - 210 glyphs for Unicode, 134 for Apple Roman
		BOOST_REQUIRE_NO_THROW(c.newFontFamily("res\\BPmonoBold.ttf"));
		BOOST_REQUIRE_NO_THROW(c.newFontEncoding("APPLE_ROMAN"));

		BOOST_REQUIRE(!c.performTransformation()); // no image yet

		BOOST_REQUIRE(!c.newImage(Mat())); // wrong image

		// Ensuring single patch images provided as Mat can be tackled
		Mat testPatch(sz, sz, CV_8UC1, Scalar(127));
		BOOST_REQUIRE(c.newImage(testPatch)); // image ok
		BOOST_REQUIRE(c.performTransformation()); // ok

		Mat testColorPatch(sz, sz, CV_8UC3, Scalar::all(127));
		BOOST_REQUIRE(c.newImage(testColorPatch)); // image ok
		BOOST_REQUIRE(c.performTransformation()); // ok

		// Recognizing the glyphs from current cmap
		vector<std::tuple<const Mat, const Mat, const BestMatch>> mismatches;
		::MatchEngine::VSymDataCIt it, itEnd;
		tie(it, itEnd) = me.getSymsRange(0U, UINT_MAX);
		const unsigned symsCount = (unsigned)distance(it, itEnd), step = symsCount/100U + 1U;
		for(unsigned idx = 0U; it != itEnd; ++idx, ++it) {
			if(idx % step == 0U)
				cout<<fixed<<setprecision(2)<<setw(6)<<idx*100./symsCount<<"%\r";

			const Mat &negGlyph = it->symAndMasks[SymData::NEG_GLYPH_IDX]; // byte 0..255
			Mat patchD255;
			negGlyph.convertTo(patchD255, CV_64FC1);
			alterFgBg(patchD255, it->minVal, it->diffMinMax);
 			addWhiteNoise(patchD255, .2, 10U); // affected % and noise amplitude

			BestMatch best;
			const Mat approximated = me.approxPatch(patchD255, best);

			if(best.symIdx != idx) {
				mismatches.emplace_back(patchD255, approximated, best);
				MatchParams mp;
				cerr<<"Expecting symbol index "<<idx<<" while approximated as "<<best.symIdx<<endl;
				cerr<<"Approximation achieved score="
					<<fixed<<setprecision(17)<<best.score
					<<" while the score for the expected symbol is "
					<<fixed<<setprecision(17)<<me.assessMatch(patchD255, *it, mp)<<endl;
				wcerr<<"Params from approximated symbol: "<<best.params<<endl;
				wcerr<<"Params from expected comparison: "<<mp<<endl<<endl;
			}
		}

		// Normally, less than 3% of the altered symbols are not identified correctly.
		BOOST_CHECK((double)mismatches.size() < .03 * symsCount);

		if(!mismatches.empty()) {
			wcerr<<"The parameters were displayed in this order:"<<endl;
			wcerr<<MatchParams::HEADER<<endl<<endl;

			showMismatches("MatchEngine_CheckAlteredCmapUsingStdDev", mismatches);

		}
	}
BOOST_AUTO_TEST_SUITE_END() // MatchEngine_Tests
