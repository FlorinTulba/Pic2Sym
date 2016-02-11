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

BOOST_FIXTURE_TEST_SUITE(MatchEngine_Tests, ut::Fixt)
	BOOST_AUTO_TEST_CASE(MatchEngine_CheckParams) {
		BOOST_TEST_MESSAGE("Running MatchEngine_CheckParams ...");

		// Patches have pixels with double values 0..255.
		// Glyphs have pixels with double values 0..1.
		// Masks have pixels with byte values 0..255.

		const unsigned sz = 50U; // Select an even sz, as the tests need to set exactly half fixels
		const unsigned char valRand = (unsigned char)(ut::randUnifUint() & 0xFFU);
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
			return (unsigned char)(ut::randUnifUint() & 0xFFU);
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

		// Random patch, empty mask => mean = sdev = 0
		MatchParams::computeSdev(randD255, emptyUc, miu, sdev);
		BOOST_REQUIRE(miu && sdev);
		BOOST_TEST(*miu == 0., test_tools::tolerance(1e-4));
		BOOST_TEST(*sdev == 0., test_tools::tolerance(1e-4));
		miu = sdev = none;

		// Uniform patch, random mask => mean = valRand, sdev = 0
		MatchParams::computeSdev(Mat(sz, sz, CV_64FC1, Scalar(valRand)), randUc!=0U, miu, sdev);
		BOOST_REQUIRE(miu && sdev);
		BOOST_TEST(*miu == valRand, test_tools::tolerance(1e-4));
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
		SymData sdHorizEdgeMask(ULONG_MAX,	// glyph code doesn't matter
					0., // min glyph value (0..1 range)
					1.,	// difference between min and max glyph (0..1 range)
				   sz*sz/2.,	// pixelSum = 255*(sz^2/2)/255 = sz^2/2
				   Point2d(cd.patchCenter.x, (sz/2.-1U)/2.), // glyph's mass center
				   SymData::MatArray { {
						   halfD1,			// the glyph in 0..1 range
						   halfUc,			// fg byte mask (0 or 255)
						   invHalfUc,		// bg byte mask (0 or 255)
						   horBeltUc,		// edge byte mask (0 or 255)
						   invHalfUc,		// glyph inverse in 0..255 byte range
						   halfD1 } });		// grounded glyph is same as glyph (min is already 0)
		SymData sdVertEdgeMask(sdHorizEdgeMask);
		*(const_cast<Mat*>(&sdVertEdgeMask.symAndMasks[SymData::EDGE_MASK_IDX])) = verBeltUc;

		// Checking glyph density for the given glyph
		mp.resetSymData();
		mp.computeRhoApproxSym(sdHorizEdgeMask, cd);
		BOOST_REQUIRE(mp.glyphWeight);
		BOOST_TEST(*mp.glyphWeight == 0.5, test_tools::tolerance(1e-4));
		
		// Testing the mentioned glyph on an uniform patch (all pixels are 'valRand') =>
		// Adapting glyph's fg & bg to match the patch => the glyph becomes all 'valRand' =>
		// Its mass-center will be ( (sz-1)/2 , (sz-1)/2 )
		Mat unifPatch(sz, sz, CV_64FC1, Scalar(valRand));
		mp.resetSymData();
		mp.computeMcApproxSym(unifPatch, sdHorizEdgeMask, cd);
		BOOST_REQUIRE(mp.mcGlyph);
		BOOST_TEST(mp.mcGlyph->x == cd.patchCenter.x, test_tools::tolerance(1e-4));
		BOOST_TEST(mp.mcGlyph->y == cd.patchCenter.y, test_tools::tolerance(1e-4));
		mp.computeSdevEdge(unifPatch, sdHorizEdgeMask);
		BOOST_REQUIRE(mp.sdevEdge);
		BOOST_TEST(*mp.sdevEdge == 0., test_tools::tolerance(1e-4));
		mp.resetSymData();
		mp.computeSdevEdge(unifPatch, sdVertEdgeMask);
		BOOST_REQUIRE(mp.sdevEdge);
		BOOST_TEST(*mp.sdevEdge == 0., test_tools::tolerance(1e-4));

		// Testing the mentioned glyph on a patch equal to glyph's inverse =>
		// Adapting glyph's fg & bg to match the patch => the glyph inverses =>
		// Its mass-center will be ( (sz-1)/2 ,  (3*sz/2-1)/2 )
		mp.resetSymData();
		mp.computeMcApproxSym(invHalfD255, sdHorizEdgeMask, cd);
		BOOST_REQUIRE(mp.mcGlyph);
		BOOST_TEST(mp.mcGlyph->x == cd.patchCenter.x, test_tools::tolerance(1e-4));
		BOOST_TEST(mp.mcGlyph->y == (3*sz/2.-1U)/2., test_tools::tolerance(1e-4));
		mp.computeSdevEdge(invHalfD255, sdHorizEdgeMask);
		BOOST_REQUIRE(mp.sdevEdge);
		BOOST_TEST(*mp.sdevEdge == 0., test_tools::tolerance(1e-4));
		mp.resetSymData();
		mp.computeSdevEdge(invHalfD255, sdVertEdgeMask);
		BOOST_REQUIRE(mp.sdevEdge);
		BOOST_TEST(*mp.sdevEdge == 0., test_tools::tolerance(1e-4));

		// Testing the mentioned glyph on a patch half 85, half 170=2*85 =>
		// Adapting glyph's fg & bg to match the patch => the glyph looses contrast =>
		// Its mass-center will be ( (sz-1)/2 ,  (5*sz-6)/12 )
		mp.resetSymData();
		Mat twoBands = emptyD.clone();
		twoBands.rowRange(0, sz/2) = 170.; twoBands.rowRange(sz/2, sz) = 85.;
		mp.computeMcApproxSym(twoBands, sdHorizEdgeMask, cd);
		BOOST_REQUIRE(mp.mcGlyph);
		BOOST_TEST(mp.mcGlyph->x == cd.patchCenter.x, test_tools::tolerance(1e-4));
		BOOST_TEST(mp.mcGlyph->y == (5*sz-6)/12., test_tools::tolerance(1e-4));
		mp.computeSdevEdge(twoBands, sdHorizEdgeMask);
		BOOST_REQUIRE(mp.sdevEdge);
		BOOST_TEST(*mp.sdevEdge == 0., test_tools::tolerance(1e-4));
		mp.resetSymData();
		mp.computeSdevEdge(twoBands, sdVertEdgeMask);
		BOOST_REQUIRE(mp.sdevEdge);
		BOOST_TEST(*mp.sdevEdge == 0., test_tools::tolerance(1e-4));

		// Testing on a patch with uniform rows of values gradually growing from 0 to sz-1
		Mat szBands = emptyD.clone();
		for(unsigned i = 0U; i<sz; ++i)
			szBands.row(i) = i;
		double expectedFgAndSdevHorEdge = (sz-2)/4., expectedSdev = 0.;
		for(unsigned i = 0U; i<sz/2; ++i) {
			double diff = i - expectedFgAndSdevHorEdge;
			expectedSdev += diff*diff;
		}
		expectedSdev = sqrt(expectedSdev / (sz/2));
		mp.resetSymData(); mp.mcPatch = none;
		mp.computeMcPatch(szBands, cd);
		BOOST_REQUIRE(mp.mcPatch);
		BOOST_TEST(mp.mcPatch->x == cd.patchCenter.x, test_tools::tolerance(1e-4));
		BOOST_TEST(mp.mcPatch->y == (2*sz-1)/3., test_tools::tolerance(1e-4));
		mp.computeFg(szBands, sdHorizEdgeMask);
		BOOST_REQUIRE(mp.fg);
		BOOST_TEST(*mp.fg == expectedFgAndSdevHorEdge, test_tools::tolerance(1e-4));
		mp.computeBg(szBands, sdHorizEdgeMask);
		BOOST_REQUIRE(mp.bg);
		BOOST_TEST(*mp.bg == (3*sz-2)/4., test_tools::tolerance(1e-4));
		mp.computeSdevFg(szBands, sdHorizEdgeMask);
		BOOST_REQUIRE(mp.sdevFg);
		BOOST_TEST(*mp.sdevFg == expectedSdev, test_tools::tolerance(1e-4));
		mp.computeSdevBg(szBands, sdHorizEdgeMask);
		BOOST_REQUIRE(mp.sdevBg);
		BOOST_TEST(*mp.sdevBg == expectedSdev, test_tools::tolerance(1e-4));
		mp.computeMcApproxSym(szBands, sdHorizEdgeMask, cd);
		BOOST_REQUIRE(mp.mcGlyph);
		BOOST_TEST(mp.mcGlyph->x == cd.patchCenter.x, test_tools::tolerance(1e-4));
		BOOST_TEST(mp.mcGlyph->y == (*mp.fg * *mp.fg + *mp.bg * *mp.bg)/(sz-1.), test_tools::tolerance(1e-4));
		mp.computeSdevEdge(szBands, sdHorizEdgeMask);
		BOOST_REQUIRE(mp.sdevEdge);
		BOOST_TEST(*mp.sdevEdge == expectedFgAndSdevHorEdge, test_tools::tolerance(1e-4));
		mp.resetSymData();
		mp.computeSdevEdge(szBands, sdVertEdgeMask);
		BOOST_REQUIRE(mp.sdevEdge);
		BOOST_TEST(*mp.sdevEdge == expectedSdev, test_tools::tolerance(1e-4));
	}

	BOOST_AUTO_TEST_CASE(MatchEngine_Check) {
		BOOST_TEST_MESSAGE("Running MatchEngine_Check ...");
		Config cfg(10U, 1., 1., 1.);
		const unsigned sz = cfg.getFontSz();
		Controller c(cfg);
		FontEngine &fe = c.getFontEngine();
		MatchEngine &me = c.getMatchEngine(cfg);

		// Courier Bold Unicode > 2800 glyphs; There are 2 almost identical COMMA-s and QUOTE-s.
		// Can't identify them exactly using std. dev. for fg, bg and edges, which all appear 0.
		// In both cases, the scores are 1.00000000000000000 (17 decimals!!) 
//  	BOOST_REQUIRE_NO_THROW(c.newFontFamily("C:\\Windows\\Fonts\\courbd.ttf"));

		// Envy Code R > 600 glyphs
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
		MatchEngine::VSymDataCIt it, itEnd;
		tie(it, itEnd) = me.getSymsRange(0U, UINT_MAX);
		const unsigned symsCount = (unsigned)distance(it, itEnd), step = symsCount/100U + 1U;
		for(unsigned idx = 0U; it != itEnd; ++idx, ++it) {
			if(idx % step == 0U)
				cout<<fixed<<setprecision(2)<<setw(6)<<idx*100./symsCount<<"%\r";

			const Mat &negGlyph = it->symAndMasks[SymData::NEG_GLYPH_IDX]; // byte 0..255
			Mat patchD255;
			negGlyph.convertTo(patchD255, CV_64FC1);
			ut::alterFgBg(patchD255, it->minVal, it->diffMinMax);
 			ut::addWhiteNoise(patchD255, .2, 10U); // affected % and amplitude

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

		if(!mismatches.empty()) {
			wcerr<<"The parameters were displayed in this order:"<<endl;
			wcerr<<MatchParams::HEADER<<endl<<endl;

			ut::showMismatches("MatchEngine_Check", mismatches);

			// Normally, less than 3% of the altered symbols are not identified correctly.
			BOOST_CHECK((double)mismatches.size() < .03 * symsCount);
		}
	}
BOOST_AUTO_TEST_SUITE_END() // MatchEngine_Tests
