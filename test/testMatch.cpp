/**************************************************************************************
 This file belongs to the 'Pic2Sym' application, which
 approximates images by a grid of colored symbols with colored backgrounds.

 Project:     UnitTesting 
 File:        testMatch.cpp
 
 Author:      Florin Tulba
 Created on:  2016-2-8

 Copyrights from the libraries used by 'Pic2Sym':
 - © 2015 Boost (www.boost.org)
   License: http://www.boost.org/LICENSE_1_0.txt
            or doc/licenses/Boost.lic
 - © 2015 The FreeType Project (www.freetype.org)
   License: http://git.savannah.gnu.org/cgit/freetype/freetype2.git/plain/docs/FTL.TXT
	        or doc/licenses/FTL.txt
 - © 2015 OpenCV (www.opencv.org)
   License: http://opencv.org/license.html
            or doc/licenses/OpenCV.lic
 
 © 2016 Florin Tulba <florintulba@yahoo.com>

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
 **************************************************************************************/

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

	// Returns next unsigned int in an uniform distribution
	unsigned randUnifUint() {
		static random_device rd;
		static mt19937 gen(rd());
		static uniform_int_distribution<unsigned> uid;
		return uid(gen);
	}

	/*
	Changes randomly the foreground and background of patchD255.
	However, it will keep fg & bg at least 30 brightness units apart.

	patchD255 has min value 255*minVal01 and max value = min value + 255*diffMinMax01.
	*/
	void alterFgBg(Mat &patchD255, double minVal01, double diffMinMax01) {
		const unsigned char newFg = (unsigned char)randUnifUint()&0xFFU;
		unsigned char newBg;
		int newDiff;
		do {
			newBg = (unsigned char)(randUnifUint() & 0xFFU);
			newDiff = (int)newFg - (int)newBg;
		} while(abs(newDiff) < 30); // keep fg & bg at least 30 brightness units apart
		
		patchD255 = (patchD255 - (minVal01*255)) * (newDiff/(255*diffMinMax01)) + newBg;
#ifdef _DEBUG
		double newMinVal, newMaxVal;
		minMaxIdx(patchD255, &newMinVal, &newMaxVal);
		assert(newMinVal > -.5);
		assert(newMaxVal < 255.5);
#endif
	}

	/*
	Adds white noise to patchD255 with max amplitude maxAmplitude0255.
	The percentage of affected pixels from patchD255 is affectedPercentage01.
	*/
	void addWhiteNoise(Mat &patchD255, double affectedPercentage01, unsigned char maxAmplitude0255) {
		const int side = patchD255.rows, area = side*side,
			affectedCount = (int)(affectedPercentage01 * area);

		int noise;
		const unsigned twiceMaxAmplitude = ((unsigned)maxAmplitude0255)<<1;
		int prevVal, below, above, newVal;
		set<unsigned> affected;
		unsigned linearized;
		div_t pos; // coordinate inside the Mat, expressed as quotient and remainder of linearized
		for(int i = 0; i<affectedCount; ++i) {
			do {
				linearized = randUnifUint() % (unsigned)area;
			} while(affected.find(linearized) != affected.cend());
			affected.insert(linearized);
			pos = div((int)linearized, side);

			prevVal = (int)round(patchD255.at<double>(pos.quot, pos.rem));

			below = max(0, min((int)maxAmplitude0255, prevVal));
			above = max(0, min((int)maxAmplitude0255, 255 - prevVal));
			do {
				noise = (int)(randUnifUint() % (unsigned)(above+below+1)) - below;
			} while(noise == 0);

			newVal = prevVal + noise;
			patchD255.at<double>(pos.quot, pos.rem) = newVal;
			assert(newVal > -.5 && newVal < 255.5);
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
		//		mean = sdev = 127.5
		//		mass-center = ( (sz-1)/2 ,  255*((sz/2-1)*(sz/2)/2)/(255*sz/2) = (sz/2-1)/2 )
		MatchParams::computeSdev(halfD255, wholeUc, miu, sdev);
		BOOST_REQUIRE(miu && sdev);
		BOOST_TEST(*miu == 127.5, test_tools::tolerance(1e-4));
		BOOST_TEST(*sdev == CachedData::sdevMaxFgBg, test_tools::tolerance(1e-4));
		mp.computeMcPatch(halfD255, cd);
		BOOST_REQUIRE(mp.mcPatch);
		BOOST_TEST(mp.mcPatch->x == cd.patchCenter.x, test_tools::tolerance(1e-4));
		BOOST_TEST(mp.mcPatch->y == (sz/2.-1U)/2., test_tools::tolerance(1e-4));
		
		// Testing a glyph half 0, half 255
		SymData sdWithHorizEdgeMask(NOT_RELEVANT_UL,	// glyph code (not relevant here)
					0., // min glyph value (0..1 range)
					1.,	// difference between min and max glyph (0..1 range)
				   sz*sz/2.,	// pixelSum = 255*(sz^2/2)/255 = sz^2/2
				   Point2d(cd.patchCenter.x, (sz/2.-1U)/2.), // glyph's mass center
				   SymData::MatArray { {
						   halfUc,			// fg byte mask (0 or 255)
						   invHalfUc,		// bg byte mask (0 or 255)
						   horBeltUc,		// edge byte mask (0 or 255)
						   invHalfUc,		// glyph inverse in 0..255 byte range
						   halfD1,			// grounded glyph is same as glyph (min is already 0)
						   NOT_RELEVANT_MAT,// blur of grounded glyph (not relevant here)
						   NOT_RELEVANT_MAT // variance of grounded glyph (not relevant here)
					   } });
		SymData sdWithVertEdgeMask(sdWithHorizEdgeMask); // copy sdWithHorizEdgeMask and adapt it for vert. edge
		const_cast<Mat&>(sdWithVertEdgeMask.symAndMasks[SymData::EDGE_MASK_IDX]) = verBeltUc;

		// Checking glyph density for the given glyph
		mp.reset();
		mp.computeSymDensity(sdWithHorizEdgeMask, cd);
		BOOST_REQUIRE(mp.symDensity);
		BOOST_TEST(*mp.symDensity == 0.5, test_tools::tolerance(1e-4));
		
		// Testing the mentioned glyph on an uniform patch (all pixels are 'valRand') =>
		// Adapting glyph's fg & bg to match the patch => the glyph becomes all 'valRand' =>
		// Its mass-center will be ( (sz-1)/2 , (sz-1)/2 )
		mp.reset();
		Mat unifPatchD255(sz, sz, CV_64FC1, Scalar(valRand));
		mp.computePatchApprox(unifPatchD255, sdWithHorizEdgeMask);
		BOOST_REQUIRE(mp.patchApprox);
		double minV, maxV;
		minMaxIdx(mp.patchApprox.value(), &minV, &maxV);
		BOOST_TEST(minV == (double)valRand, test_tools::tolerance(1e-4));
		BOOST_TEST(maxV == (double)valRand, test_tools::tolerance(1e-4));
		mp.computeMcPatchApprox(unifPatchD255, sdWithHorizEdgeMask, cd);
		BOOST_REQUIRE(mp.mcPatchApprox);
		BOOST_TEST(mp.mcPatchApprox->x == cd.patchCenter.x, test_tools::tolerance(1e-4));
		BOOST_TEST(mp.mcPatchApprox->y == cd.patchCenter.y, test_tools::tolerance(1e-4));
		mp.computeSdevEdge(unifPatchD255, sdWithHorizEdgeMask);
		BOOST_REQUIRE(mp.sdevEdge);
		BOOST_TEST(*mp.sdevEdge == 0., test_tools::tolerance(1e-4));
		mp.reset();
		mp.computePatchApprox(unifPatchD255, sdWithVertEdgeMask);
		BOOST_REQUIRE(mp.patchApprox);
		minMaxIdx(mp.patchApprox.value(), &minV, &maxV);
		BOOST_TEST(minV == (double)valRand, test_tools::tolerance(1e-4));
		BOOST_TEST(maxV == (double)valRand, test_tools::tolerance(1e-4));
		mp.computeSdevEdge(unifPatchD255, sdWithVertEdgeMask);
		BOOST_REQUIRE(mp.sdevEdge);
		BOOST_TEST(*mp.sdevEdge == 0., test_tools::tolerance(1e-4));

		// Testing the mentioned glyph on a patch equal to glyph's inverse =>
		// Adapting glyph's fg & bg to match the patch => the glyph inverses =>
		// Its mass-center will be ( (sz-1)/2 ,  (3*sz/2-1)/2 )
		mp.reset();
		mp.computePatchApprox(invHalfD255, sdWithHorizEdgeMask);
		BOOST_REQUIRE(mp.patchApprox);
		minMaxIdx(mp.patchApprox.value()-invHalfD255, &minV, &maxV);
		BOOST_TEST(minV == 0., test_tools::tolerance(1e-4));
		BOOST_TEST(maxV == 0., test_tools::tolerance(1e-4));
		mp.computeMcPatchApprox(invHalfD255, sdWithHorizEdgeMask, cd);
		BOOST_REQUIRE(mp.mcPatchApprox);
		BOOST_TEST(mp.mcPatchApprox->x == cd.patchCenter.x, test_tools::tolerance(1e-4));
		BOOST_TEST(mp.mcPatchApprox->y == (3*sz/2.-1U)/2., test_tools::tolerance(1e-4));
		mp.computeSdevEdge(invHalfD255, sdWithHorizEdgeMask);
		BOOST_REQUIRE(mp.sdevEdge);
		BOOST_TEST(*mp.sdevEdge == 0., test_tools::tolerance(1e-4));
		mp.reset();
		mp.computePatchApprox(invHalfD255, sdWithVertEdgeMask);
		BOOST_REQUIRE(mp.patchApprox);
		minMaxIdx(mp.patchApprox.value()-invHalfD255, &minV, &maxV);
		BOOST_TEST(minV == 0., test_tools::tolerance(1e-4));
		BOOST_TEST(maxV == 0., test_tools::tolerance(1e-4));
		mp.computeSdevEdge(invHalfD255, sdWithVertEdgeMask);
		BOOST_REQUIRE(mp.sdevEdge);
		BOOST_TEST(*mp.sdevEdge == 0., test_tools::tolerance(1e-4));

		// Testing the mentioned glyph on a patch half 85, half 170=2*85 =>
		// Adapting glyph's fg & bg to match the patch => the glyph looses contrast =>
		// Its mass-center will be ( (sz-1)/2 ,  (5*sz-6)/12 )
		mp.reset();
		Mat twoBandsD255 = emptyD.clone();
		twoBandsD255.rowRange(0, sz/2) = 170.; twoBandsD255.rowRange(sz/2, sz) = 85.;
		mp.computePatchApprox(twoBandsD255, sdWithHorizEdgeMask);
		BOOST_REQUIRE(mp.patchApprox);
		minMaxIdx(mp.patchApprox.value()-twoBandsD255, &minV, &maxV);
		BOOST_TEST(minV == 0., test_tools::tolerance(1e-4));
		BOOST_TEST(maxV == 0., test_tools::tolerance(1e-4));
		mp.computeMcPatchApprox(twoBandsD255, sdWithHorizEdgeMask, cd);
		BOOST_REQUIRE(mp.mcPatchApprox);
		BOOST_TEST(mp.mcPatchApprox->x == cd.patchCenter.x, test_tools::tolerance(1e-4));
		BOOST_TEST(mp.mcPatchApprox->y == (5*sz-6)/12., test_tools::tolerance(1e-4));
		mp.computeSdevEdge(twoBandsD255, sdWithHorizEdgeMask);
		BOOST_REQUIRE(mp.sdevEdge);
		BOOST_TEST(*mp.sdevEdge == 0., test_tools::tolerance(1e-4));
		mp.reset();
		mp.computePatchApprox(twoBandsD255, sdWithVertEdgeMask);
		BOOST_REQUIRE(mp.patchApprox);
		minMaxIdx(mp.patchApprox.value()-twoBandsD255, &minV, &maxV);
		BOOST_TEST(minV == 0., test_tools::tolerance(1e-4));
		BOOST_TEST(maxV == 0., test_tools::tolerance(1e-4));
		mp.computeSdevEdge(twoBandsD255, sdWithVertEdgeMask);
		BOOST_REQUIRE(mp.sdevEdge);
		BOOST_TEST(*mp.sdevEdge == 0., test_tools::tolerance(1e-4));

		// Testing on a patch with uniform rows of values gradually growing from 0 to sz-1
		mp.reset(false);
		Mat szBandsD255 = emptyD.clone();
		for(unsigned i = 0U; i<sz; ++i)
			szBandsD255.row(i) = i;
		const double expectedFgAndSdevHorEdge = (sz-2)/4.,
					expectedBg = (3*sz-2)/4.;
		double expectedSdev = 0.;
		for(unsigned i = 0U; i<sz/2; ++i) {
			double diff = i - expectedFgAndSdevHorEdge;
			expectedSdev += diff*diff;
		}
		expectedSdev = sqrt(expectedSdev / (sz/2));
		Mat expectedPatchApprox(sz, sz, CV_64FC1, Scalar(expectedBg));
		expectedPatchApprox.rowRange(0, sz/2) = expectedFgAndSdevHorEdge;
		mp.computePatchApprox(szBandsD255, sdWithHorizEdgeMask);
		BOOST_REQUIRE(mp.patchApprox);
		minMaxIdx(mp.patchApprox.value()-expectedPatchApprox, &minV, &maxV);
		BOOST_TEST(minV == 0., test_tools::tolerance(1e-4));
		BOOST_TEST(maxV == 0., test_tools::tolerance(1e-4));
		mp.computeMcPatch(szBandsD255, cd);
		BOOST_REQUIRE(mp.mcPatch);
		BOOST_TEST(mp.mcPatch->x == cd.patchCenter.x, test_tools::tolerance(1e-4));
		BOOST_TEST(mp.mcPatch->y == (2*sz-1)/3., test_tools::tolerance(1e-4));
		mp.computeFg(szBandsD255, sdWithHorizEdgeMask);
		BOOST_REQUIRE(mp.fg);
		BOOST_TEST(*mp.fg == expectedFgAndSdevHorEdge, test_tools::tolerance(1e-4));
		mp.computeBg(szBandsD255, sdWithHorizEdgeMask);
		BOOST_REQUIRE(mp.bg);
		BOOST_TEST(*mp.bg == expectedBg, test_tools::tolerance(1e-4));
		mp.computeSdevFg(szBandsD255, sdWithHorizEdgeMask);
		BOOST_REQUIRE(mp.sdevFg);
		BOOST_TEST(*mp.sdevFg == expectedSdev, test_tools::tolerance(1e-4));
		mp.computeSdevBg(szBandsD255, sdWithHorizEdgeMask);
		BOOST_REQUIRE(mp.sdevBg);
		BOOST_TEST(*mp.sdevBg == expectedSdev, test_tools::tolerance(1e-4));
		mp.computeMcPatchApprox(szBandsD255, sdWithHorizEdgeMask, cd);
		BOOST_REQUIRE(mp.mcPatchApprox);
		BOOST_TEST(mp.mcPatchApprox->x == cd.patchCenter.x, test_tools::tolerance(1e-4));
		BOOST_TEST(mp.mcPatchApprox->y == (*mp.fg * *mp.fg + *mp.bg * *mp.bg)/(sz-1.), test_tools::tolerance(1e-4));
		mp.computeSdevEdge(szBandsD255, sdWithHorizEdgeMask);
		BOOST_REQUIRE(mp.sdevEdge);
		BOOST_TEST(*mp.sdevEdge == expectedFgAndSdevHorEdge, test_tools::tolerance(1e-4));
		mp.reset();
		mp.computePatchApprox(szBandsD255, sdWithVertEdgeMask);
		BOOST_REQUIRE(mp.patchApprox);
		minMaxIdx(mp.patchApprox.value()-expectedPatchApprox, &minV, &maxV);
		BOOST_TEST(minV == 0., test_tools::tolerance(1e-4));
		BOOST_TEST(maxV == 0., test_tools::tolerance(1e-4));
		mp.computeSdevEdge(szBandsD255, sdWithVertEdgeMask);
		BOOST_REQUIRE(mp.sdevEdge);
		BOOST_TEST(*mp.sdevEdge == expectedSdev, test_tools::tolerance(1e-4));
	}

	BOOST_AUTO_TEST_CASE(MatchEngine_CheckStructuralSimilarityAspect) {
		BOOST_TEST_MESSAGE("Running MatchEngine_CheckStructuralSimilarityAspect ...");

		const SymSettings ss(50U);
		const unsigned sz = ss.getFontSz(), area = sz*sz;
		const MatchSettings cfg(1.); // structural similarity coefficient set to 1
		const CachedData cd;
		const StructuralSimilarity strSim(cd, cfg);
		const unsigned char valRand = (unsigned char)(randUnifUint() & 0xFF);
		MatchParams mp;

		// Using a symbol with a diagonal foreground mask
		const Mat diagFgMask = Mat::diag(Mat(1, sz, CV_8UC1, Scalar(255U))),
				diagSymD1 = Mat::diag(Mat(1, sz, CV_64FC1, Scalar(1.))),
				diagSymD255 = Mat::diag(Mat(1, sz, CV_64FC1, Scalar(255.)));
		Mat blurOfGroundedGlyph, varOfGroundedGlyph,
			allButMainDiagBgMask = Mat(sz, sz, CV_8UC1, Scalar(255U));
		allButMainDiagBgMask.diag() = 0U;
		const unsigned cnz = area - sz;
		BOOST_REQUIRE(countNonZero(allButMainDiagBgMask) == cnz);
		GaussianBlur(diagSymD1, blurOfGroundedGlyph,
					 StructuralSimilarity::WIN_SIZE, StructuralSimilarity::SIGMA, 0.,
					 cv::BORDER_REPLICATE);
		GaussianBlur(diagSymD1.mul(diagSymD1), varOfGroundedGlyph,
					 StructuralSimilarity::WIN_SIZE, StructuralSimilarity::SIGMA, 0.,
					 cv::BORDER_REPLICATE);
		varOfGroundedGlyph -= blurOfGroundedGlyph.mul(blurOfGroundedGlyph);

		SymData sd(NOT_RELEVANT_UL, // symbol code (not relevant here)
				   NOT_RELEVANT_D, // min in range 0..1 (not relevant here)
				   1., // diff between min..max, each in range 0..1
				   NOT_RELEVANT_D, // pixelSum (not relevant here)
				   NOT_RELEVANT_POINT, // mc sym for original fg & bg (not relevant here)
				   SymData::MatArray { { // symAndMasks
						   diagFgMask,				// FG_MASK_IDX
						   allButMainDiagBgMask,	// BG_MASK_IDX
						   NOT_RELEVANT_MAT,		// EDGE_MASK_IDX (not relevant here)
						   NOT_RELEVANT_MAT,		// NEG_SYM_IDX (not relevant here)
						   diagSymD1,				// GROUNDED_GLYPH_IDX - same as the glyph
						   blurOfGroundedGlyph,		// BLURRED_GLYPH_IDX
						   varOfGroundedGlyph		// VARIANCE_GR_SYM_IDX
					   } });

		// Testing on an uniform patch
		Mat patchD255(sz, sz, CV_64FC1, Scalar((double)valRand));
		double res = strSim.assessMatch(patchD255, sd, mp), minV, maxV;
		BOOST_REQUIRE(mp.fg && mp.bg && mp.ssim &&
					  mp.patchApprox && mp.blurredPatch && mp.blurredPatchSq && mp.variancePatch);
		BOOST_TEST(*mp.fg == (double)valRand, test_tools::tolerance(1e-4));
		BOOST_TEST(*mp.bg == (double)valRand, test_tools::tolerance(1e-4));
		minMaxIdx(mp.patchApprox.value(), &minV, &maxV);
		BOOST_TEST(minV == (double)valRand, test_tools::tolerance(1e-4));
		BOOST_TEST(maxV == (double)valRand, test_tools::tolerance(1e-4));
		BOOST_TEST(res == 1., test_tools::tolerance(1e-4));

		// Testing on a patch with the background = valRand and diagonal = valRand's complementary value
		mp.reset(false);
		const double complVal = (double)((128U + valRand) & 0xFFU);
		patchD255.diag() = complVal;
		res = strSim.assessMatch(patchD255, sd, mp);
		BOOST_REQUIRE(mp.fg && mp.bg && mp.patchApprox);
		minMaxIdx(mp.patchApprox.value() - patchD255, &minV, &maxV);
		BOOST_TEST(minV == 0., test_tools::tolerance(1e-4));
		BOOST_TEST(maxV == 0., test_tools::tolerance(1e-4));
		BOOST_TEST(*mp.fg == complVal, test_tools::tolerance(1e-4));
		BOOST_TEST(*mp.bg == (double)valRand, test_tools::tolerance(1e-4));
		BOOST_TEST(res == 1., test_tools::tolerance(1e-4));
	}

	BOOST_AUTO_TEST_CASE(MatchEngine_CheckFgAspect) {
		BOOST_TEST_MESSAGE("Running MatchEngine_CheckFgAspect ...");

		const SymSettings ss(50U); // symbol size (use even values, as some tests set half of the patch)
		const unsigned sz = ss.getFontSz();
		const MatchSettings cfg(0.,	// structural similarity coefficient (not relevant here)
								1.); // fg correctness coefficient set to 1
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
						   diagFgMask,			// FG_MASK_IDX
						   NOT_RELEVANT_MAT,	// BG_MASK_IDX (not relevant here)
						   NOT_RELEVANT_MAT,	// EDGE_MASK_IDX (not relevant here)
						   NOT_RELEVANT_MAT,	// NEG_SYM_IDX (not relevant here)
						   NOT_RELEVANT_MAT,	// GROUNDED_GLYPH_IDX (not relevant here)
						   NOT_RELEVANT_MAT,	// BLURRED_GLYPH_IDX (not relevant here)
						   NOT_RELEVANT_MAT		// VARIANCE_GR_SYM_IDX (not relevant here)
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

		const SymSettings ss(50U); // symbol size (use even >=4 values, as some tests below need that)
		const unsigned sz = ss.getFontSz(), area = sz*sz;
		const MatchSettings cfg(
							0.,	// structural similarity coefficient (not relevant here)
							0, // fg correctness coefficient (not relevant here)
							1); // edge correctness coefficient set to 1
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
						   diagFgMask,			// FG_MASK_IDX
						   allBut3DiagsBgMask,	// BG_MASK_IDX
						   sideDiagsEdgeMask,	// EDGE_MASK_IDX
						   NOT_RELEVANT_MAT,	// NEG_SYM_IDX (not relevant here)
						   groundedGlyph,		// GROUNDED_GLYPH_IDX
						   NOT_RELEVANT_MAT,	// BLURRED_GLYPH_IDX (not relevant here)
						   NOT_RELEVANT_MAT		// VARIANCE_GR_SYM_IDX (not relevant here)
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
		BOOST_TEST(res == 1.-valRand/510., test_tools::tolerance(1e-4));

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
		BOOST_TEST(res == 1.-expectedSdev/255, test_tools::tolerance(1e-4));

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
		BOOST_TEST(res == 1.-expectedSdevEdge/255, test_tools::tolerance(1e-4));
	}

	BOOST_AUTO_TEST_CASE(MatchEngine_CheckBgAspect) {
		BOOST_TEST_MESSAGE("Running MatchEngine_CheckBgAspect ...");

		const SymSettings ss(50U); // symbol size (use even >=4 values, as some tests below need that)
		const MatchSettings cfg(
						0.,	// structural similarity coefficient (not relevant here)
						0, // fg correctness coefficient (not relevant here)
						0, // edge correctness coefficient (not relevant here)
						1.); // bg correctness coefficient set to 1
		const unsigned sz = ss.getFontSz(), area = sz*sz;
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
						   NOT_RELEVANT_MAT,	// FG_MASK_IDX (not relevant here)
						   allBut3DiagsBgMask,	// BG_MASK_IDX
						   NOT_RELEVANT_MAT,	// EDGE_MASK_IDX (not relevant here)
						   NOT_RELEVANT_MAT,	// NEG_SYM_IDX (not relevant here)
						   NOT_RELEVANT_MAT,	// GROUNDED_GLYPH_IDX (not relevant here)
						   NOT_RELEVANT_MAT,	// BLURRED_GLYPH_IDX
						   NOT_RELEVANT_MAT		// VARIANCE_GR_SYM_IDX (not relevant here)
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

		const SymSettings ss(50U); // symbol size (use even >=4 values, as some tests below need that)
		const MatchSettings cfg(
						0.,	// structural similarity coefficient (not relevant here)
						0, // fg correctness coefficient (not relevant here)
						0, // edge correctness coefficient (not relevant here)
						0, // bg correctness coefficient (not relevant here)
						1.); // contrast coefficient set to 1
		const unsigned sz = ss.getFontSz(), area = sz*sz;
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
						   diagFgMask,			// FG_MASK_IDX
						   allBut3DiagsBgMask,	// BG_MASK_IDX
						   NOT_RELEVANT_MAT,	// EDGE_MASK_IDX (not relevant here)
						   NOT_RELEVANT_MAT,	// NEG_SYM_IDX (not relevant here)
						   NOT_RELEVANT_MAT,	// GROUNDED_GLYPH_IDX (not relevant here)
						   NOT_RELEVANT_MAT,	// BLURRED_GLYPH_IDX (not relevant here)
						   NOT_RELEVANT_MAT		// VARIANCE_GR_SYM_IDX (not relevant here)
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

		const SymSettings ss(50U); // symbol size
		MatchSettings cfg(
						0.,	// structural similarity coefficient (not relevant here)
						0, // fg correctness coefficient (not relevant here)
						0, // edge correctness coefficient (not relevant here)
						0, // bg correctness coefficient (not relevant here)
						0, // contrast coefficient (not relevant here)
						1.); // gravitational smoothness coefficient set to 1
		const unsigned sz = ss.getFontSz(), area = sz*sz, sz_1 = sz-1U;
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
						   fgMask,				// FG_MASK_IDX
						   bgMask,				// BG_MASK_IDX
						   NOT_RELEVANT_MAT,	// EDGE_MASK_IDX (not relevant here)
						   NOT_RELEVANT_MAT,	// NEG_SYM_IDX (not relevant here)
						   NOT_RELEVANT_MAT,	// GROUNDED_GLYPH_IDX (not relevant here)
						   NOT_RELEVANT_MAT,	// BLURRED_GLYPH_IDX (not relevant here)
						   NOT_RELEVANT_MAT		// VARIANCE_GR_SYM_IDX (not relevant here)
					   } });

		// Using a patch with a single 255 pixel in top left corner
		Mat patchD255 = Mat::zeros(sz, sz, CV_64FC1);
		patchD255.at<double>(0, 0) = 255.;
		double res = gs.assessMatch(patchD255, sd, mp);
		BOOST_REQUIRE(mp.fg && mp.bg && mp.symDensity && mp.mcPatchApprox && mp.mcPatch);
		BOOST_TEST(*mp.fg == 0., test_tools::tolerance(1e-4));
		BOOST_TEST(*mp.bg == 255./(area-1), test_tools::tolerance(1e-4));
		BOOST_TEST(*mp.symDensity == 1./area, test_tools::tolerance(1e-8));

		// Symbol's mc migrated diagonally from bottom-right corner to above the center of patch.
		// Migration occurred due to the new fg & bg of the symbol
		BOOST_TEST(mp.mcPatchApprox->x == cd.patchCenter.x - .5/(sz+1), test_tools::tolerance(1e-8));
		BOOST_TEST(mp.mcPatchApprox->y == cd.patchCenter.y - .5/(sz+1), test_tools::tolerance(1e-8));
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
		BOOST_REQUIRE(mp.fg && mp.bg && mp.symDensity && mp.mcPatchApprox && mp.mcPatch);
		BOOST_TEST(*mp.fg == 0., test_tools::tolerance(1e-4));
		BOOST_TEST(*mp.bg == 2*255./(area-1), test_tools::tolerance(1e-4));
		BOOST_TEST(*mp.symDensity == 1./area, test_tools::tolerance(1e-8));

		// Symbol's mc migrated diagonally from bottom-right corner to above the center of patch.
		// Migration occurred due to the new fg & bg of the symbol
		BOOST_TEST(mp.mcPatchApprox->x == cd.patchCenter.x - .5/(sz+1), test_tools::tolerance(1e-8));
		BOOST_TEST(mp.mcPatchApprox->y == cd.patchCenter.y - .5/(sz+1), test_tools::tolerance(1e-8));
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
		BOOST_REQUIRE(mp.fg && mp.bg && mp.symDensity && mp.mcPatchApprox && mp.mcPatch);
		BOOST_TEST(*mp.fg == 0., test_tools::tolerance(1e-4));
		BOOST_TEST(*mp.bg == 255./(area-1), test_tools::tolerance(1e-4));
		BOOST_TEST(*mp.symDensity == 1./area, test_tools::tolerance(1e-8));

		// Symbol's mc migrated diagonally from bottom-right corner to above the center of patch.
		// Migration occurred due to the new fg & bg of the symbol
		BOOST_TEST(mp.mcPatchApprox->x == cd.patchCenter.x - .5/(sz+1), test_tools::tolerance(1e-8));
		BOOST_TEST(mp.mcPatchApprox->y == cd.patchCenter.y - .5/(sz+1), test_tools::tolerance(1e-8));
		BOOST_TEST(mp.mcPatch->x == (double)sz_1, test_tools::tolerance(1e-4));
		BOOST_TEST(mp.mcPatch->y == 0., test_tools::tolerance(1e-4));
		dx = .5*(sz_1 + 1/(sz+1.)); dy = .5*(sz_1 - 1/(sz+1.));
		BOOST_TEST(res == 1. + (cd.preferredMaxMcDist - sqrt(dx*dx + dy*dy)) /
				   cd.complPrefMaxMcDist, test_tools::tolerance(1e-8));
	}

	BOOST_AUTO_TEST_CASE(MatchEngine_CheckDirectionalSmoothnessAspect) {
		BOOST_TEST_MESSAGE("Running MatchEngine_CheckDirectionalSmoothnessAspect ...");

		const SymSettings ss(50U); // symbol size (Use even values >=4 to allow creating some scenarios below)
		MatchSettings cfg(
					0.,	// structural similarity coefficient (not relevant here)
					0, // fg correctness coefficient (not relevant here)
					0, // edge correctness coefficient (not relevant here)
					0, // bg correctness coefficient (not relevant here)
					0, // contrast coefficient (not relevant here)
					0, // gravitational smoothness coefficient (not relevant here)
					1.); // directional smoothness coefficient set to 1
		const unsigned sz = ss.getFontSz(), area = sz*sz, sz_1 = sz-1U;
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
						   fgMask,				// FG_MASK_IDX
						   bgMask,				// BG_MASK_IDX
						   NOT_RELEVANT_MAT,	// EDGE_MASK_IDX (not relevant here)
						   NOT_RELEVANT_MAT,	// NEG_SYM_IDX (not relevant here)
						   NOT_RELEVANT_MAT,	// GROUNDED_GLYPH_IDX (not relevant here)
						   NOT_RELEVANT_MAT,	// BLURRED_GLYPH_IDX (not relevant here)
						   NOT_RELEVANT_MAT		// VARIANCE_GR_SYM_IDX (not relevant here)
					   } });

		// Using a patch with a single 255 pixel in top left corner
		// Same as 1st scenario from Gravitational Smoothness
		Mat patchD255 = Mat::zeros(sz, sz, CV_64FC1);
		patchD255.at<double>(0, 0) = 255.;
		double res = ds.assessMatch(patchD255, sd, mp);
		BOOST_REQUIRE(mp.fg && mp.bg && mp.symDensity && mp.mcPatchApprox && mp.mcPatch);
		// Symbol's mc migrated diagonally from bottom-right corner to above the center of patch.
		// Migration occurred due to the new fg & bg of the symbol
		BOOST_TEST(mp.mcPatchApprox->x == cd.patchCenter.x - .5/(sz+1), test_tools::tolerance(1e-8));
		BOOST_TEST(mp.mcPatchApprox->y == cd.patchCenter.y - .5/(sz+1), test_tools::tolerance(1e-8));
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
		BOOST_REQUIRE(mp.fg && mp.bg && mp.symDensity && mp.mcPatchApprox && mp.mcPatch);
		// Symbol's mc migrated diagonally from bottom-right corner to above the center of patch.
		// Migration occurred due to the new fg & bg of the symbol
		BOOST_TEST(mp.mcPatchApprox->x == cd.patchCenter.x - .5/(sz+1), test_tools::tolerance(1e-8));
		BOOST_TEST(mp.mcPatchApprox->y == cd.patchCenter.y - .5/(sz+1), test_tools::tolerance(1e-8));
		BOOST_TEST(mp.mcPatch->x == cd.patchCenter.x, test_tools::tolerance(1e-4));
		BOOST_TEST(mp.mcPatch->y == 0., test_tools::tolerance(1e-4));
		BOOST_TEST(res == 1., test_tools::tolerance(1e-8)); // angle = 45 => cos = sqrt(2)/2

		// Using a patch with the last pixel on the top row on 255
		// Same as 3rd scenario from Gravitational Smoothness
		mp.reset(false);
		patchD255 = Mat::zeros(sz, sz, CV_64FC1);
		patchD255.at<double>(0, sz_1) = 255.;
		res = ds.assessMatch(patchD255, sd, mp);
		BOOST_REQUIRE(mp.fg && mp.bg && mp.symDensity && mp.mcPatchApprox && mp.mcPatch);
		// Symbol's mc migrated diagonally from bottom-right corner to above the center of patch.
		// Migration occurred due to the new fg & bg of the symbol
		BOOST_TEST(mp.mcPatchApprox->x == cd.patchCenter.x - .5/(sz+1), test_tools::tolerance(1e-8));
		BOOST_TEST(mp.mcPatchApprox->y == cd.patchCenter.y - .5/(sz+1), test_tools::tolerance(1e-8));
		BOOST_TEST(mp.mcPatch->x == (double)sz_1, test_tools::tolerance(1e-4));
		BOOST_TEST(mp.mcPatch->y == 0., test_tools::tolerance(1e-4));
		BOOST_TEST(res == 2-sqrt(2), test_tools::tolerance(1e-8)); // angle is 90 => cos = 0
	}

	BOOST_AUTO_TEST_CASE(MatchEngine_CheckLargerSymAspect) {
		BOOST_TEST_MESSAGE("Running MatchEngine_CheckLargerSymAspect ...");

		const SymSettings ss(50U); // symbol size
		const MatchSettings cfg(
						0.,	// structural similarity coefficient (not relevant here)
						0, // fg correctness coefficient (not relevant here)
						0, // edge correctness coefficient (not relevant here)
						0, // bg correctness coefficient (not relevant here)
						0, // contrast coefficient (not relevant here)
						0, // mc offset coefficient (not relevant here)
						0, // mc angle coefficient (not relevant here)
						1.); // symbol weight coefficient set to 1
		const unsigned sz = ss.getFontSz(), area = sz*sz;
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
						   NOT_RELEVANT_MAT, // NEG_SYM_IDX (not relevant here)
						   NOT_RELEVANT_MAT, // GROUNDED_GLYPH_IDX (not relevant here)
						   NOT_RELEVANT_MAT, // BLURRED_GLYPH_IDX (not relevant here)
						   NOT_RELEVANT_MAT		// VARIANCE_GR_SYM_IDX (not relevant here)
					   } });

		// Testing with an empty symbol (sd.pixelSum == 0)
		double res = ls.assessMatch(NOT_RELEVANT_MAT, sd, mp);
		BOOST_REQUIRE(mp.symDensity);
		BOOST_TEST(*mp.symDensity == 0., test_tools::tolerance(1e-4));
		BOOST_TEST(res == 1.-cd.smallGlyphsCoverage, test_tools::tolerance(1e-4));

		// Testing with a symbol that just enters the 'large symbols' category
		mp.reset(); const_cast<double&>(sd.pixelSum) = area * cd.smallGlyphsCoverage;
		res = ls.assessMatch(NOT_RELEVANT_MAT, sd, mp);
		BOOST_REQUIRE(mp.symDensity);
		BOOST_TEST(*mp.symDensity == cd.smallGlyphsCoverage, test_tools::tolerance(1e-4));
		BOOST_TEST(res == 1., test_tools::tolerance(1e-4));

		// Testing with largest possible symbol
		mp.reset(); const_cast<double&>(sd.pixelSum) = area;
		res = ls.assessMatch(NOT_RELEVANT_MAT, sd, mp);
		BOOST_REQUIRE(mp.symDensity);
		BOOST_TEST(*mp.symDensity == 1., test_tools::tolerance(1e-4));
		BOOST_TEST(res == 2.-cd.smallGlyphsCoverage, test_tools::tolerance(1e-4));
	}

	BOOST_AUTO_TEST_CASE(MatchEngine_CheckAlteredCmapUsingStructuralSimilarity) {
		BOOST_TEST_MESSAGE("Running MatchEngine_CheckAlteredCmapUsingStructuralSimilarity ...");

		Settings s(std::move(MatchSettings(1.))); // structural similarity factor set to 1
		const unsigned sz = s.symSettings().getFontSz(); // default font size is 10
		::Controller c(s);
		::MatchEngine &me = c.getMatchEngine(s);

		// Courier Bold Unicode > 2800 glyphs; There are 2 almost identical COMMA-s and QUOTE-s.
// 	  	BOOST_REQUIRE_NO_THROW(c.newFontFamily("C:\\Windows\\Fonts\\courbd.ttf"));

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

			const Mat &negGlyph = it->symAndMasks[SymData::NEG_SYM_IDX]; // byte 0..255
			Mat patchD255;
			negGlyph.convertTo(patchD255, CV_64FC1, -1., 255.);
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

			showMismatches("MatchEngine_CheckAlteredCmapUsingStructuralSimilarity", mismatches);
		}
	}

	BOOST_AUTO_TEST_CASE(MatchEngine_CheckAlteredCmapUsingStdDev) {
		BOOST_TEST_MESSAGE("Running MatchEngine_CheckAlteredCmapUsingStdDev ...");

		// sdev-s factors for fg, edges and bg set to 1
		Settings s(std::move(MatchSettings(0., 1., 1., 1.)));
		const unsigned sz = s.symSettings().getFontSz(); // default font size is 10
		::Controller c(s);
		::MatchEngine &me = c.getMatchEngine(s);

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

			const Mat &negGlyph = it->symAndMasks[SymData::NEG_SYM_IDX]; // byte 0..255
			Mat patchD255;
			negGlyph.convertTo(patchD255, CV_64FC1, -1., 255.);
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
