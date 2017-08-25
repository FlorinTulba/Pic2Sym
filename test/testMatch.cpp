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
 
 (c) 2016, 2017 Florin Tulba <florintulba@yahoo.com>

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
#include "preselectSyms.h"
#include "scoreThresholds.h"
#include "preselectionHelper.h"
#include "fileIterationHelper.h"
#include "misc.h"
#include "settings.h"
#include "symSettings.h"
#include "imgSettings.h"
#include "matchSettings.h"
#include "matchParams.h"
#include "matchEngine.h"
#include "patch.h"
#include "bestMatch.h"
#include "controller.h"
#include "matchAspects.h"
#include "matchAssessment.h"
#include "matchSupportWithPreselection.h"
#include "matchProgressWithPreselection.h"
#include "structuralSimilarity.h"
#include "blur.h"
#include "controlPanelActionsBase.h"
#include "symData.h"

#pragma warning ( push, 0 )

#include <random>
#include <algorithm>
#include <numeric>
#include <iterator>
#include <set>

#include <boost/optional/optional.hpp>
#include <boost/optional/optional_io.hpp>
#include <boost/test/data/test_case.hpp>

#include <opencv2/imgproc/imgproc.hpp>

#pragma warning ( pop )

using namespace cv;
using namespace std;
using namespace boost;

extern const string StructuralSimilarity_BlurType;
extern const string ClusterAlgName;
extern const unsigned ShortListLength;
extern const double AdmitOnShortListEvenForInferiorScoreFactor;
extern const bool Transform_BlurredPatches_InsteadOf_Originals;
extern unsigned TinySymsSz();
static const unsigned TinySymsSize = TinySymsSz();

namespace ut {
	/// dummy value
	const double NOT_RELEVANT_D = 0.;	
	/// dummy value
	const unsigned long NOT_RELEVANT_UL = ULONG_MAX;
	/// dummy value
	const size_t NOT_RELEVANT_SZ = 0U;
	/// dummy value
	const Point2d NOT_RELEVANT_POINT;
	/// dummy value
	const Mat NOT_RELEVANT_MAT;

	/**
	Changes randomly the foreground and background of patchD255.

	However, it will keep fg & bg at least 30 brightness units apart.

	patchD255 has min value 255*minVal01 and max value = min value + 255*diffMinMax01.
	*/
	void alterFgBg(Mat &patchD255, double minVal01, double diffMinMax01) {
		const auto newFg = randUnsignedChar();
		unsigned char newBg;
		int newDiff;
		do {
			newBg = randUnsignedChar();
			newDiff = (int)newFg - (int)newBg;
		} while(abs(newDiff) < 30); // keep fg & bg at least 30 brightness units apart
		
		patchD255 = (patchD255 - (minVal01*255)) * (newDiff/(255*diffMinMax01)) + newBg;
#ifdef _DEBUG
		double newMinVal, newMaxVal;
		minMaxIdx(patchD255, &newMinVal, &newMaxVal);
		assert(newMinVal > -.5);
		assert(newMaxVal < 255.5);
#endif // _DEBUG
	}

	/**
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

	/// Creates the matrix 'randUc' of sz x sz random unsigned chars
	void randInit(unsigned sz, Mat &randUc) {
		vector<unsigned char> randV;
		generate_n(back_inserter(randV), sz*sz, [] {return randUnsignedChar(); });
		randUc = Mat(sz, sz, CV_8UC1, (void*)randV.data());
	}

	/// Creates matrix 'invHalfD255' sz x sz with first half black (0) and second half white (255)
	void updateInvHalfD255(unsigned sz, Mat &invHalfD255) {
		invHalfD255 = Mat(sz, sz, CV_64FC1, Scalar(0.));
		invHalfD255.rowRange(sz/2, sz) = 255.;
	}

	/**
	Creates 2 shared_ptr to 2 symbol data objects.

	@param sz patch side length
	@param sdWithHorizEdgeMask symbol data shared_ptr for a glyph whose vertical halves are white and black. The 2 rows mid height define a horizontal edge mask
	@param sdWithVertEdgeMask symbol data shared_ptr for a glyph whose vertical halves are white and black. The 2 columns mid width define a vertical edge mask, which simply instructs where to look for edges within this glyph. It doesn't correspond with the actual horizontal edge of the glyph, but it will check the patches for a vertical edge.
	*/
	void updateSymDataOfHalfFullGlyphs(unsigned sz,
									   std::shared_ptr<SymData> &sdWithHorizEdgeMask, 
									   std::shared_ptr<SymData> &sdWithVertEdgeMask) {
		// 2 rows mid height
		Mat horBeltUc = Mat(sz, sz, CV_8UC1, Scalar(0U)); horBeltUc.rowRange(sz/2-1, sz/2+1) = 255U;

		// 2 columns mid width
		Mat verBeltUc = Mat(sz, sz, CV_8UC1, Scalar(0U)); verBeltUc.colRange(sz/2-1, sz/2+1) = 255U;

		// 1st horizontal half - full
		Mat halfUc = Mat(sz, sz, CV_8UC1, Scalar(0U)); halfUc.rowRange(0, sz/2) = 255U;
		Mat halfD1 = Mat(sz, sz, CV_64FC1, Scalar(0.)); halfD1.rowRange(0, sz/2) = 1.;

		// 2nd horizontal half - full
		Mat invHalfUc = 255U - halfUc;

		// A glyph half 0, half 255
		sdWithHorizEdgeMask = std::make_shared<SymData>(
				NOT_RELEVANT_UL,	// glyph code (not relevant here)
				NOT_RELEVANT_SZ,	// symbol index (not relevant here)
				0., // min glyph value (0..1 range)
				1.,	// difference between min and max glyph (0..1 range)
				.5,	// avgPixVal = 255*(sz^2/2)/(255*sz^2) = 1/2
				Point2d(.5, (sz/2.-1.) / (2. * (sz-1U))), // glyph's mass center downscaled by (sz-1)
				SymData::IdxMatMap {
					{ ISymData::FG_MASK_IDX, halfUc },
					{ ISymData::BG_MASK_IDX, invHalfUc },
					{ ISymData::EDGE_MASK_IDX, horBeltUc },

					// grounded glyph is same as glyph (min is already 0)
					{ ISymData::GROUNDED_SYM_IDX, halfD1 }},
				invHalfUc);
		
		// copy sdWithHorizEdgeMask and adapt it for vert. edge
		sdWithVertEdgeMask = std::make_shared<SymData>(*sdWithHorizEdgeMask);
		sdWithVertEdgeMask->masks[ISymData::EDGE_MASK_IDX] = verBeltUc;
	}

	/// Fixture helping tests computing matching parameters 
	template<bool PreselMode>
	struct MatchParamsFixt : PreselFixt<PreselMode> {
		unsigned sz;	///< patch side length (tests should use provided public accessor methods)
		Mat emptyUc;	///< Empty matrix sz x sz of unsigned chars
		Mat emptyD;		///< Empty matrix sz x sz of doubles
		Mat fullUc;		///< sz x sz matrix filled with 255 (unsigned char)
		Mat invHalfD255;///< sz x sz matrix vertically split in 2. One half white, the other black
		Mat consec;		///< 0 .. sz-1 consecutive double values
		CachedData cd;	///< cached data based on sz
		Mat randUc;		///< sz x sz random unsigned chars
		Mat randD1;		///< sz x sz random doubles (0 .. 1)
		Mat randD255;	///< sz x sz random doubles (0 .. 255)
		std::shared_ptr<SymData> sdWithHorizEdgeMask, sdWithVertEdgeMask;

		/// Random initialization of randUc and computing corresponding randD1 and randD255
		void randInitPatch() {
			randInit(sz, randUc);
			randUc.convertTo(randD1, CV_64FC1, 1./255);
			randUc.convertTo(randD255, CV_64FC1);
		}

	public:
		optional<double> miu;	///< mean computed within tests
		optional<double> sdev;	///< standard deviation computed within tests
		MatchParams mp;			///< matching parameters computed during tests
		double minV;			///< min of the error between the patch and its approximation
		double maxV;			///< max of the error between the patch and its approximation

		const Mat& getEmptyUc() const { return emptyUc; }
		const Mat& getEmptyD() const { return emptyD; }
		const Mat& getFullUc() const { return fullUc; }
		const Mat& getConsec() const { return consec; }
		const Mat& getRandUc() const { return randUc; }
		const Mat& getRandD255() const { return randD255; }
		const Mat& getInvHalfD255() const { return invHalfD255; }
		const CachedData& getCd() const { return cd; }
		const std::shared_ptr<SymData> getSdWithHorizEdgeMask() const { return sdWithHorizEdgeMask; }
		const std::shared_ptr<SymData> getSdWithVertEdgeMask() const { return sdWithVertEdgeMask; }
		unsigned getSz() const { return sz; }

		/// Updates sz, cd, consec and the matrices empty, random and full
		void setSz(unsigned sz_) {
			sz = sz_;
			emptyUc = Mat(sz, sz, CV_8UC1, Scalar(0U));
			emptyD = Mat(sz, sz, CV_64FC1, Scalar(0.));
			fullUc = Mat(sz, sz, CV_8UC1, Scalar(255U));
			cd.useNewSymSize(sz);
			consec = cd.getConsec().clone();

			randInitPatch();

			updateInvHalfD255(sz, invHalfD255);
			updateSymDataOfHalfFullGlyphs(sz, sdWithHorizEdgeMask, sdWithVertEdgeMask);
		}

		/**
		Creates a fixture useful for the tests computing match parameters.

		@param sz_ patch side length. Select an even value, as the tests need to set exactly half pixels
		*/
		MatchParamsFixt(unsigned sz_ = 50U) : PreselFixt(), cd(PreselectionByTinySyms) {
			setSz(sz_);
		}
	};

	/// Fixture for the matching aspects
	template<bool PreselMode>
	struct MatchAspectsFixt : PreselFixt<PreselMode> {
		unsigned sz;	///< patch side length (tests should use provided public accessor methods)
		unsigned area;	///< sz^2 (Use getter within tests)

	protected:
		CachedData cd;		///< cached data that can be changed during tests
		MatchSettings ms;	///< determines which aspect is tested

	public:
		MatchParams mp;	///< tests compute these match parameters
		Mat patchD255;	///< declares the patch to be approximated
		double res;		///< assessment of the match between the patch and the resulted approximation
		double minV;	///< min of the error between the patch and its approximation
		double maxV;	///< max of the error between the patch and its approximation

		/// Updates sz and area
		void setSz(unsigned sz_) {
			sz = sz_;
			area = sz*sz;
		}
		unsigned getSz() const { return sz; }
		unsigned getArea() const { return area; }

		/**
		Creates a fixture useful for the tests checking the match aspects.

		@param sz_ patch side length. Select an even value, as the tests need to set exactly half pixels
		*/
		MatchAspectsFixt(unsigned sz_ = 50U) : PreselFixt(), cd(PreselectionByTinySyms) {
			setSz(sz_);
		}

		/// help for blur
		const IBlurEngine& blurSupport = BlurEngine::byName(StructuralSimilarity_BlurType);
	};

	template<bool PreselMode>
	struct AlteredCmapFixture : PreselFixt<PreselMode> {
		const string oldClusterAlgName = ClusterAlgName;

		AlteredCmapFixture(const string &clustAlgName) : PreselFixt() {
			if(0 != ClusterAlgName.compare(clustAlgName))
				const_cast<string&>(ClusterAlgName) = clustAlgName;
		}
		
		~AlteredCmapFixture() {
			if(0 != ClusterAlgName.compare(oldClusterAlgName))
				const_cast<string&>(ClusterAlgName) = oldClusterAlgName;
		}
	};

	/// Parameterized test case. Uniform patches get approximated by completely faded glyphs.
	void checkParams_UniformPatch_GlyphConvergesToPatch(unsigned sz,
														const CachedData &cd,
														const SymData &symData) {
		const auto valRand = randUnsignedChar(1U);
		Mat unifPatchD255(sz, sz, CV_64FC1, Scalar(valRand));
		MatchParams mp;
		double minV, maxV;
		mp.computePatchApprox(unifPatchD255, symData);
		BOOST_REQUIRE(mp.getPatchApprox());
		minMaxIdx(mp.getPatchApprox().value(), &minV, &maxV);
		BOOST_TEST(minV == (double)valRand, test_tools::tolerance(1e-4));
		BOOST_TEST(maxV == (double)valRand, test_tools::tolerance(1e-4));
		// Its mass-center will be ( (sz-1)/2 , (sz-1)/2 )
		mp.computeMcPatchApprox(unifPatchD255, symData, cd);
		BOOST_REQUIRE(mp.getMcPatchApprox());
		BOOST_TEST(mp.getMcPatchApprox()->x == .5, test_tools::tolerance(1e-4));
		BOOST_TEST(mp.getMcPatchApprox()->y == .5, test_tools::tolerance(1e-4));
		mp.computeSdevEdge(unifPatchD255, symData);
		BOOST_REQUIRE(mp.getSdevEdge());
		BOOST_TEST(*mp.getSdevEdge() == 0., test_tools::tolerance(1e-4));
	}

	/// Parameterized test case. A glyph which is the inverse of a patch converges to the patch.
	void checkParams_TestedGlyphIsInverseOfPatch_GlyphConvergesToPatch(unsigned sz,
																	   const Mat &invHalfD255,
																	   const CachedData &cd,
																	   const SymData &symData) {
		MatchParams mp;
		double minV, maxV;
		mp.computePatchApprox(invHalfD255, symData);
		BOOST_REQUIRE(mp.getPatchApprox());
		minMaxIdx(mp.getPatchApprox().value()-invHalfD255, &minV, &maxV);
		BOOST_TEST(minV == 0., test_tools::tolerance(1e-4));
		BOOST_TEST(maxV == 0., test_tools::tolerance(1e-4));
		// Its mass-center will be ( (sz-1)/2 ,  (3*sz/2-1)/2 ) all downsized by (sz-1)
		mp.computeMcPatchApprox(invHalfD255, symData, cd); \
		BOOST_REQUIRE(mp.getMcPatchApprox());
		BOOST_TEST(mp.getMcPatchApprox()->x == .5, test_tools::tolerance(1e-4));
		BOOST_TEST(mp.getMcPatchApprox()->y == (3*sz/2.-1U) / (2. * (sz-1U)), test_tools::tolerance(1e-4));
		mp.computeSdevEdge(invHalfD255, symData);
		BOOST_REQUIRE(mp.getSdevEdge());
		BOOST_TEST(*mp.getSdevEdge() == 0., test_tools::tolerance(1e-4));
	}
	
	/// Parameterized test case. A glyph which is the highest-contrast version of a patch converges to the patch.
	void checkParams_TestHalfFullGlyphOnDimmerPatch_GlyphLoosesContrast(unsigned sz,
																		const Mat &emptyD,
																		const CachedData &cd,
																		const SymData &symData) {
		MatchParams mp;
		double minV, maxV;
		// Testing the mentioned glyph on a patch half 85, half 170=2*85
		Mat twoBandsD255 = emptyD.clone();
		twoBandsD255.rowRange(0, sz/2) = 170.; twoBandsD255.rowRange(sz/2, sz) = 85.;

		mp.computePatchApprox(twoBandsD255, symData);
		BOOST_REQUIRE(mp.getPatchApprox());
		minMaxIdx(mp.getPatchApprox().value()-twoBandsD255, &minV, &maxV);
		BOOST_TEST(minV == 0., test_tools::tolerance(1e-4));
		BOOST_TEST(maxV == 0., test_tools::tolerance(1e-4));
		// Its mass-center will be ( (sz-1)/2 ,  (5*sz-6)/12 )
		mp.computeMcPatchApprox(twoBandsD255, symData, cd);
		BOOST_REQUIRE(mp.getMcPatchApprox());
		BOOST_TEST(mp.getMcPatchApprox()->x == .5, test_tools::tolerance(1e-4));
		BOOST_TEST(mp.getMcPatchApprox()->y == (5*sz-6) / (12. * (sz-1)), test_tools::tolerance(1e-4));
		mp.computeSdevEdge(twoBandsD255, symData);
		BOOST_REQUIRE(mp.getSdevEdge());
		BOOST_TEST(*mp.getSdevEdge() == 0., test_tools::tolerance(1e-4));
	}

	// msArray and fonts from below are used to generate the data sets used within
	// CheckAlteredCmap_UsingAspects_ExpectLessThan3or55PercentErrors test below
	
	/// array of all MatchSetting-s configurations to be tested for all selected font configurations
	const MatchSettings msArray[] = {
		MatchSettings().set_kSsim(1.),
		MatchSettings().set_kSdevFg(1.).set_kSdevEdge(1.).set_kSdevBg(1.) };


	/**
	map of fonts to be tested.

	The elements are the full combination of font family and the desired encoding.

	Below are some variants:
	Courier Bold UNICODE ("C:\\Windows\\Fonts\\courbd.ttf") > 2800 glyphs; There are 2 almost identical COMMA-s and QUOTE-s.
	Envy Code R UNICODE ("C:\\Windows\\Fonts\\Envy Code R Bold.ttf") > 600 glyphs
	Bp Mono Bold ("res\\BPmonoBold.ttf") - 210 glyphs for UNICODE, 134 for APPLE_ROMAN
	*/
	map<string, string> fonts { { "res\\BPmonoBold.ttf", "APPLE_ROMAN" } };

	typedef decltype(fonts)::value_type StrStrPair; // used to specify that such pairs shouldn't be displayed
}

using namespace ut;

BOOST_TEST_DONT_PRINT_LOG_VALUE(StrStrPair)

/*
Iterating this file 2 times, with counter values from 0 to 1.
0 will be used for PreselectionByTinySyms set on false
1 will be used for PreselectionByTinySyms set on true
*/
#	define BOOST_PP_ITERATION_LIMITS		(0, 1)
#	define BOOST_PP_FILENAME_1				"testMatch.cpp" /* __FILE__ didn't work! */
#	include BOOST_PP_ITERATE()

#else // BOOST_PP_IS_ITERATING is 1 (true) -- The rest of the file is iterated twice

#	if BOOST_PP_ITERATION() == 0
#		define UsePreselection		false
#		define SuiteSuffix	_noPreselection

#	elif BOOST_PP_ITERATION() == 1
#		undef UsePreselection
#		define UsePreselection		true
#		undef SuiteSuffix
#		define SuiteSuffix	_withPreselection

#	else // BOOST_PP_ITERATION() >= 2
#		undef UsePreselection
#		undef SuiteSuffix
#	endif // BOOST_PP_ITERATION()

/**
Trying to identify all the glyphs from a font family based on certain match settings.
The symbols within a given font family are altered:
- changing their foreground and background to random values while ensuring a minimal contrast
- using additive noise

Applying this test for more combinations of MatchSetting-s (msArray) and font families (fonts),
with and without the preselection mechanism enabled (UsePreselection).

Normally, when the preselection is disabled, there are less than 3% misidentified symbols.
For non-altered symbols, there should be no such errors.

Unfortunately, the preselection enforces the identification to be performed MOSTLY based on
tiny versions of the glyphs, and the percentage of the errors increases with the size of the symbol set.
This is because an increasing number of such tiny symbols will be hard to distinguish based on masks
that became really inadequate (for instance - a large circular mask becomes a small square or a dot).
So, BpMono Bold family with around 200 symbols generates less than 13% errors,
but Courier Bold Unicode family with &gt; 2000 symbols generates a bit less than 55% misidentifications

GENERALLY, no matter the preselection, Structural Similarity identifies correctly more symbols
than (Foreground, Background and Edge) aspects.
*/
DataTestCase(CheckAlteredCmap_UsingAspects_ExpectLessThan3or55PercentErrors, SuiteSuffix,
			// For Cartesian product (all combinations) use below '*'
			// For zip (only the combinations formed by pairs with same indices) use '^'
			boost::unit_test::data::make(msArray) * fonts,
			ms, font) {
	// Disabling clustering for this test,
	// as the result is highly sensitive to the accuracy of the clustering,
	// which depends on quite a number of adjustable parameters.
	// Every time one of these parameters is modified,
	// this test might cross the accepted mismatches threshold.
	AlteredCmapFixture<UsePreselection> fixt("None"); // mandatory

	const string &fontFamily = font.first, &encoding = font.second;

	ostringstream oss;
	oss<<"CheckAlteredCmap_UsingAspects_ExpectLessThan3or55PercentErrors for "
		<<fontFamily<<'('<<encoding<<") with "<<ms<<' '
		<<(PreselectionByTinySyms ? "with" : "without")<<" Preselection";
	string nameOfTest = oss.str();
	for(const char toReplace : string(" \t\n\\/():"))
		replace(BOUNDS(nameOfTest), toReplace, '_');
	BOOST_TEST_MESSAGE("Running " + nameOfTest);

	Settings s(ms);
	::Controller c(s);
	std::shared_ptr<IControlPanelActions> cpa = c.getControlPanelActions();
	::MatchEngine &me = dynamic_cast<::MatchEngine&>(c.getMatchEngine(s));

	cpa->newFontFamily(fontFamily);
	cpa->newFontEncoding(encoding);

	// default font size is 10 - normally at most 10% errors
	//c.newFontSize(40U); // apparently larger fonts produce less errors (for 40 rarely any)
	me.getReady();

	// Recognizing the glyphs from current cmap
	VSymData::const_iterator it = cbegin(me.symsSet), itEnd = cend(me.symsSet);
	const unsigned symsCount = (unsigned)distance(it, itEnd),
					SymsBatchSz = 25U;
	vector<unique_ptr<BestMatch>> mismatches;
	const auto &assessor = me.assessor();
	for(unsigned idx = 0U; it != itEnd; ++idx, ++it) {
		const Mat &negGlyph = (*it)->getNegSym(); // byte 0..255
		Mat patchD255, blurredPatch;
		std::shared_ptr<Patch> thePatch;
		int attempts = 10;
		do {
			negGlyph.convertTo(patchD255, CV_64FC1, -1., 255.);
			alterFgBg(patchD255, (*it)->getMinVal(), (*it)->getDiffMinMax());
			addWhiteNoise(patchD255, .2, 10U); // affected % and noise amplitude
			thePatch = std::make_shared<Patch>(patchD255);
		} while(--attempts >= 0 && !thePatch->nonUniform());

		if(!thePatch->nonUniform())
			continue; // couldn't generate a noisy non-uniform glyph

		auto best = make_unique<BestMatch>(*thePatch);

		if(PreselectionByTinySyms) {
			Mat tinyPatchMat, blurredTinyPatch;
			resize(thePatch->getOrig(), tinyPatchMat, Size(TinySymsSize, TinySymsSize), 0., 0., INTER_AREA);
			resize(thePatch->getBlurred(), blurredTinyPatch, Size(TinySymsSize, TinySymsSize), 0., 0., INTER_AREA);
			Patch tinyPatch(tinyPatchMat, blurredTinyPatch, tinyPatchMat.channels()>1);
			if(!tinyPatch.nonUniform()) {
				tinyPatch.forceApproximation(); // forcing the approximation for the tiny patch

				const Mat &patch2Process = Transform_BlurredPatches_InsteadOf_Originals ?
											blurredTinyPatch : tinyPatchMat;
				Mat grayD;
				patch2Process.clone().convertTo(grayD, CV_64FC1);
				tinyPatch.setMatrixToApprox(grayD);
			}
			BestMatch bestTiny(tinyPatch);

			// Below using 3 instead of ShortListLength, to avoid surprises caused by altered configurations
			TopCandidateMatches tcm(3U);
			MatchProgressWithPreselection mpwp(tcm);
			MatchSupportWithPreselection mswp(me.cachedData, me.symsSet,
											  me.matchAssessor, ms);

			for(unsigned s = 0U; s < symsCount; s += SymsBatchSz) {
				// Value from below is 0.8 instead of AdmitOnShortListEvenForInferiorScoreFactor
				// to avoid surprises caused by altered configurations
				tcm.reset(bestTiny.setScore(best->getScore() * 0.8).getScore());
				if(me.improvesBasedOnBatch(s, min(s+SymsBatchSz, symsCount), bestTiny, mpwp)) {
					tcm.prepareReport();
					auto &&shortList = tcm.getShortList();
					mswp.improvesBasedOnBatchShortList(std::move(shortList), *best);
				}
			}

		} else { // PreselectionByTinySyms == false here
			MatchProgress dummy;
			for(unsigned s = 0U; s < symsCount; s += SymsBatchSz)
				me.improvesBasedOnBatch(s, min(s+SymsBatchSz, symsCount), *best, dummy);
		}

		if(best->getSymIdx() != idx) {
			MatchParams mp;
			double score;
			ScoreThresholds scoresToBeat;
			assessor.scoresToBeat(best->getScore(), scoresToBeat);
			assessor.isBetterMatch(patchD255, **it, me.cachedData, scoresToBeat, mp, score);
			cerr<<"Expecting symbol index "<<idx<<" while approximated as "<<best->getSymIdx()<<endl;
			cerr<<"Approximation achieved score="
				<<fixed<<setprecision(17)<<best->getScore()
				<<" while the score for the expected symbol is "
				<<fixed<<setprecision(17)<<score<<endl;
			wcerr<<"Params from approximated symbol: "<<*best->getParams()<<endl;
			wcerr<<"Params from expected comparison: "<<mp<<endl<<endl;
			mismatches.push_back(std::move(best));
		}
	}

	if(PreselectionByTinySyms) { // Preselection with tiny symbols - less accurate approximation allowed
		// Normally, less than 55% of the altered symbols are not identified correctly.
		BOOST_CHECK((double)mismatches.size() < .55 * symsCount);
	} else { // No Preselection - much more accurate approximation expected
		// Normally, less than 3% of the altered symbols are not identified correctly.
		BOOST_CHECK((double)mismatches.size() < .03 * symsCount);
	}

	if(!mismatches.empty()) {
		extern const wstring MatchParams_HEADER;
		wcerr<<"The parameters were displayed in this order:"<<endl;
		wcerr<<MatchParams_HEADER<<endl<<endl;

		showMismatches(nameOfTest, mismatches);
	}
}

FixtureTestSuiteSuffix(MatchParamsFixt<UsePreselection>, MeanSdevMassCenterComputation_Tests, SuiteSuffix)
	// Patches have pixels with double values 0..255.
	// Glyphs have pixels with double values 0..1.
	// Masks have pixels with byte values 0..255.

	AutoTestCase1(CheckSdevComputation_PrecededOrNotByMeanComputation_SameSdevExpected, SuiteSuffix);
		optional<double> miu1, sdev1;

		// Check that computeSdev performs the same when preceded or not by computeMean
		MatchParams::computeMean(getRandD255(), getFullUc(), miu);
		MatchParams::computeSdev(getRandD255(), getFullUc(), miu, sdev); // miu already computed

		MatchParams::computeSdev(getRandD255(), getFullUc(), miu1, sdev1); // miu1 not computed yet
		BOOST_REQUIRE(miu && sdev && miu1 && sdev1);
		BOOST_TEST(*miu == *miu1, test_tools::tolerance(1e-4));
		BOOST_TEST(*sdev == *sdev1, test_tools::tolerance(1e-4));
	}

	AutoTestCase1(CheckMeanAndSdev_RandomPatchWithEmptyMask_ZeroMeanAndSdev, SuiteSuffix);
		MatchParams::computeSdev(getRandD255(), getEmptyUc(), miu, sdev);
		BOOST_REQUIRE(miu && sdev);
		BOOST_TEST(*miu == 0., test_tools::tolerance(1e-4));
		BOOST_TEST(*sdev == 0., test_tools::tolerance(1e-4));
	}

	AutoTestCase1(CheckMeanAndSdev_UniformPatchWithRandomMask_PixelsValueAsMeanAndZeroSdev, SuiteSuffix);
		const auto valRand = randUnsignedChar();

		MatchParams::computeSdev(Mat(getSz(), getSz(), CV_64FC1, Scalar(valRand)), getRandUc()!=0U, miu, sdev);
		BOOST_REQUIRE(miu && sdev);
		BOOST_TEST(*miu == (double)valRand, test_tools::tolerance(1e-4));
		BOOST_TEST(*sdev == 0., test_tools::tolerance(1e-4));
	}

	AutoTestCase1(CheckMeanSdevMassCenter_Half0Half255AndNoMask_127dot5MeanAndSdevAndPredictedMassCenter, SuiteSuffix);
		const auto valRand = randUnsignedChar();
		Mat halfD255 = getEmptyD().clone(); halfD255.rowRange(0, getSz()/2) = 255.;

		MatchParams::computeSdev(halfD255, getFullUc(), miu, sdev);
		BOOST_REQUIRE(miu && sdev);
		BOOST_TEST(*miu == 127.5, test_tools::tolerance(1e-4));
		BOOST_TEST(*sdev == CachedData::MaxSdev::forFgOrBg(), test_tools::tolerance(1e-4));
		MatchParams mp;
		mp.computeMcPatch(halfD255, getCd());
		BOOST_REQUIRE(mp.getMcPatch());
		// mass-center = ( (sz-1)/2 ,  255*((sz/2-1)*(sz/2)/2)/(255*sz/2) = (sz/2-1)/2 ) all downsized by (sz-1)
		BOOST_TEST(mp.getMcPatch()->x == .5, test_tools::tolerance(1e-4));
		BOOST_TEST(mp.getMcPatch()->y == (getSz()/2.-1.) / (2. * (getSz() - 1)), test_tools::tolerance(1e-4));
	}

	AutoTestCase1(ComputeSymDensity_SuperiorHalfOfPatchFull_0dot5, SuiteSuffix);
		mp.computeSymDensity(*getSdWithHorizEdgeMask());
		BOOST_REQUIRE(mp.getSymDensity());
		BOOST_TEST(*mp.getSymDensity() == 0.5, test_tools::tolerance(1e-4));
	}

	AutoTestCase1(CheckParams_UniformPatchHorizontalEdge_GlyphConvergesToPatch, SuiteSuffix);
		checkParams_UniformPatch_GlyphConvergesToPatch(getSz(), getCd(), *getSdWithHorizEdgeMask());
	}

	AutoTestCase1(CheckParams_UniformPatchVerticalEdge_GlyphConvergesToPatch, SuiteSuffix);
		checkParams_UniformPatch_GlyphConvergesToPatch(getSz(), getCd(), *getSdWithVertEdgeMask());
	}

	AutoTestCase1(CheckParams_TestedGlyphWithHorizontalEdgeIsInverseOfPatch_GlyphConvergesToPatch, SuiteSuffix);
		checkParams_TestedGlyphIsInverseOfPatch_GlyphConvergesToPatch(getSz(), getInvHalfD255(), getCd(), *getSdWithHorizEdgeMask());
	}

	AutoTestCase1(CheckParams_TestedGlyphWithVerticalEdgeIsInverseOfPatch_GlyphConvergesToPatch, SuiteSuffix);
		checkParams_TestedGlyphIsInverseOfPatch_GlyphConvergesToPatch(getSz(), getInvHalfD255(), getCd(), *getSdWithVertEdgeMask());
	}

	AutoTestCase1(CheckParams_TestHalfFullGlypWithHorizontalEdgehOnDimmerPatch_GlyphLoosesContrast, SuiteSuffix);
		checkParams_TestHalfFullGlyphOnDimmerPatch_GlyphLoosesContrast(getSz(), getEmptyD(), getCd(), *getSdWithHorizEdgeMask());
	}

	AutoTestCase1(CheckParams_TestHalfFullGlypWithVerticalEdgehOnDimmerPatch_GlyphLoosesContrast, SuiteSuffix);
		checkParams_TestHalfFullGlyphOnDimmerPatch_GlyphLoosesContrast(getSz(), getEmptyD(), getCd(), *getSdWithVertEdgeMask());
	}

	AutoTestCase1(CheckParams_RowValuesSameAsRowIndices_PredictedParams, SuiteSuffix);
		// Testing on a patch with uniform rows of values gradually growing from 0 to sz-1
		Mat szBandsD255 = getEmptyD().clone();
		for(unsigned i = 0U; i<getSz(); ++i)
			szBandsD255.row(i) = i;
		const double expectedFgAndSdevHorEdge = (getSz()-2)/4.,
			expectedBg = (3*getSz()-2)/4.;
		double expectedSdev = 0.;
		for(unsigned i = 0U; i<getSz()/2; ++i) {
			double diff = i - expectedFgAndSdevHorEdge;
			expectedSdev += diff*diff;
		}
		expectedSdev = sqrt(expectedSdev / (getSz()/2));
		Mat expectedPatchApprox(getSz(), getSz(), CV_64FC1, Scalar(expectedBg));
		expectedPatchApprox.rowRange(0, getSz()/2) = expectedFgAndSdevHorEdge;
		mp.computePatchApprox(szBandsD255, *getSdWithHorizEdgeMask());
		BOOST_REQUIRE(mp.getPatchApprox());
		minMaxIdx(mp.getPatchApprox().value()-expectedPatchApprox, &minV, &maxV);
		BOOST_TEST(minV == 0., test_tools::tolerance(1e-4));
		BOOST_TEST(maxV == 0., test_tools::tolerance(1e-4));
		mp.computeMcPatch(szBandsD255, getCd());
		BOOST_REQUIRE(mp.getMcPatch());
		BOOST_TEST(mp.getMcPatch()->x == .5, test_tools::tolerance(1e-4));
		BOOST_TEST(mp.getMcPatch()->y == (2*getSz()-1) / (3. * (getSz()-1)), test_tools::tolerance(1e-4));
		mp.computeFg(szBandsD255, *getSdWithHorizEdgeMask());
		BOOST_REQUIRE(mp.getFg());
		BOOST_TEST(*mp.getFg() == expectedFgAndSdevHorEdge, test_tools::tolerance(1e-4));
		mp.computeBg(szBandsD255, *getSdWithHorizEdgeMask());
		BOOST_REQUIRE(mp.getBg());
		BOOST_TEST(*mp.getBg() == expectedBg, test_tools::tolerance(1e-4));
		mp.computeSdevFg(szBandsD255, *getSdWithHorizEdgeMask());
		BOOST_REQUIRE(mp.getSdevFg());
		BOOST_TEST(*mp.getSdevFg() == expectedSdev, test_tools::tolerance(1e-4));
		mp.computeSdevBg(szBandsD255, *getSdWithHorizEdgeMask());
		BOOST_REQUIRE(mp.getSdevBg());
		BOOST_TEST(*mp.getSdevBg() == expectedSdev, test_tools::tolerance(1e-4));
		mp.computeMcPatchApprox(szBandsD255, *getSdWithHorizEdgeMask(), getCd());
		BOOST_REQUIRE(mp.getMcPatchApprox());
		BOOST_TEST(mp.getMcPatchApprox()->x == .5, test_tools::tolerance(1e-4));
		BOOST_TEST(mp.getMcPatchApprox()->y == (*mp.getFg() * *mp.getFg() + *mp.getBg() * *mp.getBg()) / pow(getSz()-1U, 2), test_tools::tolerance(1e-4));
		mp.computeSdevEdge(szBandsD255, *getSdWithHorizEdgeMask());
		BOOST_REQUIRE(mp.getSdevEdge());
		BOOST_TEST(*mp.getSdevEdge() == expectedFgAndSdevHorEdge, test_tools::tolerance(1e-4));
		
		// Do some tests also for a glyph with a vertical edge
		mp.reset();
		mp.computePatchApprox(szBandsD255, *getSdWithVertEdgeMask());
		BOOST_REQUIRE(mp.getPatchApprox());
		minMaxIdx(mp.getPatchApprox().value()-expectedPatchApprox, &minV, &maxV);
		BOOST_TEST(minV == 0., test_tools::tolerance(1e-4));
		BOOST_TEST(maxV == 0., test_tools::tolerance(1e-4));
		mp.computeSdevEdge(szBandsD255, *getSdWithVertEdgeMask());
		BOOST_REQUIRE(mp.getSdevEdge());
		BOOST_TEST(*mp.getSdevEdge() == expectedSdev, test_tools::tolerance(1e-4));
	}
BOOST_AUTO_TEST_SUITE_END() // CheckMatchParams

FixtureTestSuiteSuffix(MatchAspectsFixt<UsePreselection>, MatchAspects_Tests, SuiteSuffix)
	AutoTestCase1(CheckStructuralSimilarity_UniformPatchAndDiagGlyph_GlyphBecomesPatch, SuiteSuffix);
		ms.set_kSsim(1.);
		const StructuralSimilarity strSim(ms);
		const auto valRand = randUnsignedChar(1U);

		// Using a symbol with a diagonal foreground mask
		const Mat diagFgMask = Mat::diag(Mat(1, getSz(), CV_8UC1, Scalar(255U))),
			diagSymD1 = Mat::diag(Mat(1, getSz(), CV_64FC1, Scalar(1.))),
			diagSymD255 = Mat::diag(Mat(1, getSz(), CV_64FC1, Scalar(255.)));
		Mat blurOfGroundedGlyph, varOfGroundedGlyph,
			allButMainDiagBgMask = Mat(getSz(), getSz(), CV_8UC1, Scalar(255U));
		allButMainDiagBgMask.diag() = 0U;
		const unsigned cnz = getArea() - getSz();
		BOOST_REQUIRE(countNonZero(allButMainDiagBgMask) == cnz);
		blurSupport.process(diagSymD1, blurOfGroundedGlyph, PreselectionByTinySyms);
		blurSupport.process(diagSymD1.mul(diagSymD1), varOfGroundedGlyph, PreselectionByTinySyms);
		varOfGroundedGlyph -= blurOfGroundedGlyph.mul(blurOfGroundedGlyph);

		SymData sd(NOT_RELEVANT_UL, // symbol code (not relevant here)
				   NOT_RELEVANT_SZ,	// symbol index (not relevant here)
				   NOT_RELEVANT_D, // min in range 0..1 (not relevant here)
				   1., // diff between min..max, each in range 0..1
				   NOT_RELEVANT_D, // avgPixVal (not relevant here)
				   NOT_RELEVANT_POINT, // mc sym for original fg & bg (not relevant here)
				   SymData::IdxMatMap {
					   { ISymData::FG_MASK_IDX, diagFgMask },
					   { ISymData::BG_MASK_IDX, allButMainDiagBgMask },
					   { ISymData::GROUNDED_SYM_IDX, diagSymD1 }, // same as the glyph
					   { ISymData::BLURRED_GR_SYM_IDX, blurOfGroundedGlyph },
					   { ISymData::VARIANCE_GR_SYM_IDX, varOfGroundedGlyph }});

		// Testing on an uniform patch
		patchD255 = Mat(getSz(), getSz(), CV_64FC1, Scalar((double)valRand));
		res = strSim.assessMatch(patchD255, sd, cd, mp);
		BOOST_REQUIRE(mp.getFg() && mp.getBg() && mp.getSsim() &&
					  mp.getPatchApprox() && mp.getBlurredPatch() && mp.getBlurredPatchSq() && mp.getVariancePatch());
		BOOST_TEST(*mp.getFg() == (double)valRand, test_tools::tolerance(1e-4));
		BOOST_TEST(*mp.getBg() == (double)valRand, test_tools::tolerance(1e-4));
		minMaxIdx(mp.getPatchApprox().value(), &minV, &maxV);
		BOOST_TEST(minV == (double)valRand, test_tools::tolerance(1e-4));
		BOOST_TEST(maxV == (double)valRand, test_tools::tolerance(1e-4));
		BOOST_TEST(res == 1., test_tools::tolerance(1e-4));
	}

	AutoTestCase1(CheckStructuralSimilarity_PatchIsDimmedGlyph_GlyphBecomesPatch, SuiteSuffix);
		ms.set_kSsim(1.);
		const StructuralSimilarity strSim(ms);
		const auto valRand = randUnsignedChar(1U);

		// Using a symbol with a diagonal foreground mask
		const Mat diagFgMask = Mat::diag(Mat(1, getSz(), CV_8UC1, Scalar(255U))),
			diagSymD1 = Mat::diag(Mat(1, getSz(), CV_64FC1, Scalar(1.))),
			diagSymD255 = Mat::diag(Mat(1, getSz(), CV_64FC1, Scalar(255.)));
		Mat blurOfGroundedGlyph, varOfGroundedGlyph,
			allButMainDiagBgMask = Mat(getSz(), getSz(), CV_8UC1, Scalar(255U));
		allButMainDiagBgMask.diag() = 0U;
		const unsigned cnz = getArea() - getSz();
		BOOST_REQUIRE(countNonZero(allButMainDiagBgMask) == cnz);
		blurSupport.process(diagSymD1, blurOfGroundedGlyph, PreselectionByTinySyms);
		blurSupport.process(diagSymD1.mul(diagSymD1), varOfGroundedGlyph, PreselectionByTinySyms);
		varOfGroundedGlyph -= blurOfGroundedGlyph.mul(blurOfGroundedGlyph);

		SymData sd(NOT_RELEVANT_UL, // symbol code (not relevant here)
				   NOT_RELEVANT_SZ,	// symbol index (not relevant here)
				   NOT_RELEVANT_D, // min in range 0..1 (not relevant here)
				   1., // diff between min..max, each in range 0..1
				   NOT_RELEVANT_D, // avgPixVal (not relevant here)
				   NOT_RELEVANT_POINT, // mc sym for original fg & bg (not relevant here)
				   SymData::IdxMatMap {
					   { ISymData::FG_MASK_IDX, diagFgMask },
					   { ISymData::BG_MASK_IDX, allButMainDiagBgMask },
					   { ISymData::GROUNDED_SYM_IDX, diagSymD1 }, // same as the glyph
					   { ISymData::BLURRED_GR_SYM_IDX, blurOfGroundedGlyph },
					   { ISymData::VARIANCE_GR_SYM_IDX, varOfGroundedGlyph } });

		// Testing on a patch with the background = valRand and diagonal = valRand's complementary value
		const double complVal = (double)((128U + valRand) & 0xFFU);
		patchD255 = Mat(getSz(), getSz(), CV_64FC1, Scalar((double)valRand));
		patchD255.diag() = complVal;
		res = strSim.assessMatch(patchD255, sd, cd, mp);
		BOOST_REQUIRE(mp.getFg() && mp.getBg() && mp.getPatchApprox());
		minMaxIdx(mp.getPatchApprox().value() - patchD255, &minV, &maxV);
		BOOST_TEST(minV == 0., test_tools::tolerance(1e-4));
		BOOST_TEST(maxV == 0., test_tools::tolerance(1e-4));
		BOOST_TEST(*mp.getFg() == complVal, test_tools::tolerance(1e-4));
		BOOST_TEST(*mp.getBg() == (double)valRand, test_tools::tolerance(1e-4));
		BOOST_TEST(res == 1., test_tools::tolerance(1e-4));
	}

	AutoTestCase1(CheckFgAspect_UniformPatchAndDiagGlyph_CompleteMatchAndFgBecomesPatchValue, SuiteSuffix);
		ms.set_kSdevFg(1.);
		const FgMatch fm(ms);
		const auto valRand = randUnsignedChar(1U);

		// Using a symbol with a diagonal foreground mask
		const Mat diagFgMask = Mat::diag(Mat(1, getSz(), CV_8UC1, Scalar(255U)));

		SymData sd(NOT_RELEVANT_UL, // symbol code (not relevant here)
				   NOT_RELEVANT_SZ,	// symbol index (not relevant here)
				   NOT_RELEVANT_D, NOT_RELEVANT_D, // min and diff between min..max, each in range 0..1 (not relevant here)
				   NOT_RELEVANT_D, // avgPixVal (not relevant here)
				   NOT_RELEVANT_POINT, // mc sym for original fg & bg (not relevant here)
				   SymData::IdxMatMap {{ ISymData::FG_MASK_IDX, diagFgMask }});

		// Testing on a uniform patch
		patchD255 = Mat(getSz(), getSz(), CV_64FC1, Scalar((double)valRand));
		res = fm.assessMatch(patchD255, sd, cd, mp);
		BOOST_REQUIRE(mp.getFg() && mp.getSdevFg());
		BOOST_TEST(*mp.getFg() == (double)valRand, test_tools::tolerance(1e-4));
		BOOST_TEST(*mp.getSdevFg() == 0., test_tools::tolerance(1e-4));
		BOOST_TEST(res == 1., test_tools::tolerance(1e-4));
	}

	AutoTestCase1(CheckFgAspect_DiagGlyphAndPatchWithUpperHalfEmptyAndUniformLowerHalf_ImperfectMatch, SuiteSuffix);
		ms.set_kSdevFg(1.);
		const FgMatch fm(ms);
		const auto valRand = randUnsignedChar(1U);

		// Using a symbol with a diagonal foreground mask
		const Mat diagFgMask = Mat::diag(Mat(1, getSz(), CV_8UC1, Scalar(255U)));

		SymData sd(NOT_RELEVANT_UL, // symbol code (not relevant here)
				   NOT_RELEVANT_SZ,	// symbol index (not relevant here)
				   NOT_RELEVANT_D, NOT_RELEVANT_D, // min and diff between min..max, each in range 0..1 (not relevant here)
				   NOT_RELEVANT_D, // avgPixVal (not relevant here)
				   NOT_RELEVANT_POINT, // mc sym for original fg & bg (not relevant here)
				   SymData::IdxMatMap {{ ISymData::FG_MASK_IDX, diagFgMask }});

		// Testing on a patch with upper half empty and an uniform lower half (valRand)
		patchD255 = Mat(getSz(), getSz(), CV_64FC1, Scalar((double)valRand));
		patchD255.rowRange(0, getSz()/2) = 0.;
		res = fm.assessMatch(patchD255, sd, cd, mp);
		BOOST_REQUIRE(mp.getFg() && mp.getSdevFg());
		BOOST_TEST(*mp.getFg() == valRand/2., test_tools::tolerance(1e-4));
		BOOST_TEST(*mp.getSdevFg() == valRand/2., test_tools::tolerance(1e-4));
		BOOST_TEST(res == 1.-valRand/255., test_tools::tolerance(1e-4));
	}

	AutoTestCase1(CheckFgAspect_DiagGlyphAndPachRowValuesSameAsRowIndices_ImperfectMatch, SuiteSuffix);
		ms.set_kSdevFg(1.);
		const FgMatch fm(ms);
		const auto valRand = randUnsignedChar(1U);

		// Using a symbol with a diagonal foreground mask
		const Mat diagFgMask = Mat::diag(Mat(1, getSz(), CV_8UC1, Scalar(255U)));

		SymData sd(NOT_RELEVANT_UL, // symbol code (not relevant here)
				   NOT_RELEVANT_SZ,	// symbol index (not relevant here)
				   NOT_RELEVANT_D, NOT_RELEVANT_D, // min and diff between min..max, each in range 0..1 (not relevant here)
				   NOT_RELEVANT_D, // avgPixVal (not relevant here)
				   NOT_RELEVANT_POINT, // mc sym for original fg & bg (not relevant here)
				   SymData::IdxMatMap {{ ISymData::FG_MASK_IDX, diagFgMask }});

		double expectedMiu = (getSz()-1U)/2., expectedSdev = 0., aux;
		patchD255 = Mat::zeros(getSz(), getSz(), CV_64FC1);
		for(unsigned i = 0U; i<getSz(); ++i) {
			patchD255.row(i) = (double)i;
			aux = i-expectedMiu;
			expectedSdev += aux*aux;
		}
		expectedSdev = sqrt(expectedSdev/getSz());
		res = fm.assessMatch(patchD255, sd, cd, mp);
		BOOST_REQUIRE(mp.getFg() && mp.getSdevFg());
		BOOST_TEST(*mp.getFg() == expectedMiu, test_tools::tolerance(1e-4));
		BOOST_TEST(*mp.getSdevFg() == expectedSdev, test_tools::tolerance(1e-4));
		BOOST_TEST(res == 1.-expectedSdev/127.5, test_tools::tolerance(1e-4));
	}

	AutoTestCase1(CheckEdgeAspect_UniformPatch_PerfectMatch, SuiteSuffix);
		ms.set_kSdevEdge(1);
		const EdgeMatch em(ms);
		const auto valRand = randUnsignedChar(1U);

		// Using a symbol with a diagonal foreground mask
		const Mat diagFgMask = Mat::diag(Mat(1, getSz(), CV_8UC1, Scalar(255U)));

		// Using a symbol with an edge mask formed by 2 neighbor diagonals of the main diagonal
		Mat sideDiagsEdgeMask = Mat::zeros(getSz(), getSz(), CV_8UC1);
		sideDiagsEdgeMask.diag(1) = 255U; // 2nd diagonal lower half
		sideDiagsEdgeMask.diag(-1) = 255U; // 2nd diagonal upper half
		const unsigned cnzEdge = 2U*(getSz() - 1U);
		BOOST_REQUIRE(countNonZero(sideDiagsEdgeMask) == cnzEdge);

		// Using a symbol with a background mask full except the main 3 diagonals
		Mat allBut3DiagsBgMask = Mat(getSz(), getSz(), CV_8UC1, Scalar(255U));
		allBut3DiagsBgMask.diag() = 0U;
		allBut3DiagsBgMask.diag(1) = 0U; // 2nd diagonal lower half
		allBut3DiagsBgMask.diag(-1) = 0U; // 2nd diagonal upper half
		const unsigned cnzBg = getArea() - 3U*getSz() + 2U;
		BOOST_REQUIRE(countNonZero(allBut3DiagsBgMask) == cnzBg);

		Mat groundedGlyph = Mat::zeros(getSz(), getSz(), CV_8UC1); // bg will stay 0
		const unsigned char edgeLevel = 125U,
			maxGlyph = edgeLevel<<1; // 250
		add(groundedGlyph, maxGlyph, groundedGlyph, diagFgMask); // set fg
		add(groundedGlyph, edgeLevel, groundedGlyph, sideDiagsEdgeMask); // set edges
		groundedGlyph.convertTo(groundedGlyph, CV_64FC1, 1./255);

		SymData sd(NOT_RELEVANT_UL, // symbol code (not relevant here)
				   NOT_RELEVANT_SZ,	// symbol index (not relevant here)
				   NOT_RELEVANT_D, // min brightness value 0..1 range (not relevant here)
				   maxGlyph/255., // diff between min..max, each in range 0..1
				   NOT_RELEVANT_D, // avgPixVal (not relevant here)
				   NOT_RELEVANT_POINT, // mc sym for original fg & bg (not relevant here)
				   SymData::IdxMatMap {
					   { ISymData::FG_MASK_IDX, diagFgMask },
					   { ISymData::BG_MASK_IDX, allBut3DiagsBgMask },
					   { ISymData::EDGE_MASK_IDX, sideDiagsEdgeMask },
					   { ISymData::GROUNDED_SYM_IDX, groundedGlyph }});

		// Testing on a uniform patch
		patchD255 = Mat(getSz(), getSz(), CV_64FC1, Scalar((double)valRand));
		res = em.assessMatch(patchD255, sd, cd, mp);
		BOOST_REQUIRE(mp.getFg() && mp.getBg() && mp.getSdevEdge());
		BOOST_TEST(*mp.getFg() == (double)valRand, test_tools::tolerance(1e-4));
		BOOST_TEST(*mp.getBg() == (double)valRand, test_tools::tolerance(1e-4));
		BOOST_TEST(*mp.getSdevEdge() == 0., test_tools::tolerance(1e-4));
		BOOST_TEST(res == 1., test_tools::tolerance(1e-4));
	}

	AutoTestCase1(CheckEdgeAspect_EdgyDiagGlyphAndPatchWithUpperHalfEmptyAndUniformLowerPart_ImperfectMatch, SuiteSuffix);
		ms.set_kSdevEdge(1);
		const EdgeMatch em(ms);
		const auto valRand = randUnsignedChar(1U);

		// Using a symbol with a diagonal foreground mask
		const Mat diagFgMask = Mat::diag(Mat(1, getSz(), CV_8UC1, Scalar(255U)));

		// Using a symbol with an edge mask formed by 2 neighbor diagonals of the main diagonal
		Mat sideDiagsEdgeMask = Mat::zeros(getSz(), getSz(), CV_8UC1);
		sideDiagsEdgeMask.diag(1) = 255U; // 2nd diagonal lower half
		sideDiagsEdgeMask.diag(-1) = 255U; // 2nd diagonal upper half
		const unsigned cnzEdge = 2U*(getSz() - 1U);
		BOOST_REQUIRE(countNonZero(sideDiagsEdgeMask) == cnzEdge);

		// Using a symbol with a background mask full except the main 3 diagonals
		Mat allBut3DiagsBgMask = Mat(getSz(), getSz(), CV_8UC1, Scalar(255U));
		allBut3DiagsBgMask.diag() = 0U;
		allBut3DiagsBgMask.diag(1) = 0U; // 2nd diagonal lower half
		allBut3DiagsBgMask.diag(-1) = 0U; // 2nd diagonal upper half
		const unsigned cnzBg = getArea() - 3U*getSz() + 2U;
		BOOST_REQUIRE(countNonZero(allBut3DiagsBgMask) == cnzBg);

		Mat groundedGlyph = Mat::zeros(getSz(), getSz(), CV_8UC1); // bg will stay 0
		const unsigned char edgeLevel = 125U,
			maxGlyph = edgeLevel<<1; // 250
		add(groundedGlyph, maxGlyph, groundedGlyph, diagFgMask); // set fg
		add(groundedGlyph, edgeLevel, groundedGlyph, sideDiagsEdgeMask); // set edges
		groundedGlyph.convertTo(groundedGlyph, CV_64FC1, 1./255);

		SymData sd(NOT_RELEVANT_UL, // symbol code (not relevant here)
				   NOT_RELEVANT_SZ,	// symbol index (not relevant here)
				   NOT_RELEVANT_D, // min brightness value 0..1 range (not relevant here)
				   maxGlyph/255., // diff between min..max, each in range 0..1
				   NOT_RELEVANT_D, // avgPixVal (not relevant here)
				   NOT_RELEVANT_POINT, // mc sym for original fg & bg (not relevant here)
				   SymData::IdxMatMap {
					   { ISymData::FG_MASK_IDX, diagFgMask },
					   { ISymData::BG_MASK_IDX, allBut3DiagsBgMask },
					   { ISymData::EDGE_MASK_IDX, sideDiagsEdgeMask },
					   { ISymData::GROUNDED_SYM_IDX, groundedGlyph } });

		// Testing on a patch with upper half empty and an uniform lower half (valRand)
		patchD255 = Mat(getSz(), getSz(), CV_64FC1, Scalar((double)valRand));
		patchD255.rowRange(0, getSz()/2) = 0.;
		res = em.assessMatch(patchD255, sd, cd, mp);
		BOOST_REQUIRE(mp.getFg() && mp.getBg() && mp.getSdevEdge());
		BOOST_TEST(*mp.getFg() == valRand/2., test_tools::tolerance(1e-4));
		BOOST_TEST(*mp.getBg() == valRand/2., test_tools::tolerance(1e-4));
		BOOST_TEST(*mp.getSdevEdge() == valRand/2., test_tools::tolerance(1e-4));
		BOOST_TEST(res == 1.-valRand/510., test_tools::tolerance(1e-4));
	}

	AutoTestCase1(CheckEdgeAspect_EdgyDiagGlyphAndPatchRowValuesSameAsRowIndices_ImperfectMatch, SuiteSuffix);
		ms.set_kSdevEdge(1);
		const EdgeMatch em(ms);
		const auto valRand = randUnsignedChar(1U);

		// Using a symbol with a diagonal foreground mask
		const Mat diagFgMask = Mat::diag(Mat(1, getSz(), CV_8UC1, Scalar(255U)));

		// Using a symbol with an edge mask formed by 2 neighbor diagonals of the main diagonal
		Mat sideDiagsEdgeMask = Mat::zeros(getSz(), getSz(), CV_8UC1);
		sideDiagsEdgeMask.diag(1) = 255U; // 2nd diagonal lower half
		sideDiagsEdgeMask.diag(-1) = 255U; // 2nd diagonal upper half
		const unsigned cnzEdge = 2U*(getSz() - 1U);
		BOOST_REQUIRE(countNonZero(sideDiagsEdgeMask) == cnzEdge);

		// Using a symbol with a background mask full except the main 3 diagonals
		Mat allBut3DiagsBgMask = Mat(getSz(), getSz(), CV_8UC1, Scalar(255U));
		allBut3DiagsBgMask.diag() = 0U;
		allBut3DiagsBgMask.diag(1) = 0U; // 2nd diagonal lower half
		allBut3DiagsBgMask.diag(-1) = 0U; // 2nd diagonal upper half
		const unsigned cnzBg = getArea() - 3U*getSz() + 2U;
		BOOST_REQUIRE(countNonZero(allBut3DiagsBgMask) == cnzBg);

		Mat groundedGlyph = Mat::zeros(getSz(), getSz(), CV_8UC1); // bg will stay 0
		const unsigned char edgeLevel = 125U,
			maxGlyph = edgeLevel<<1; // 250
		add(groundedGlyph, maxGlyph, groundedGlyph, diagFgMask); // set fg
		add(groundedGlyph, edgeLevel, groundedGlyph, sideDiagsEdgeMask); // set edges
		groundedGlyph.convertTo(groundedGlyph, CV_64FC1, 1./255);

		SymData sd(NOT_RELEVANT_UL, // symbol code (not relevant here)
				   NOT_RELEVANT_SZ,	// symbol index (not relevant here)
				   NOT_RELEVANT_D, // min brightness value 0..1 range (not relevant here)
				   maxGlyph/255., // diff between min..max, each in range 0..1
				   NOT_RELEVANT_D, // avgPixVal (not relevant here)
				   NOT_RELEVANT_POINT, // mc sym for original fg & bg (not relevant here)
				   SymData::IdxMatMap {
					   { ISymData::FG_MASK_IDX, diagFgMask },
					   { ISymData::BG_MASK_IDX, allBut3DiagsBgMask },
					   { ISymData::EDGE_MASK_IDX, sideDiagsEdgeMask },
					   { ISymData::GROUNDED_SYM_IDX, groundedGlyph } });

		// Testing on a patch with uniform rows, but gradually brighter, from top to bottom
		patchD255 = Mat::zeros(getSz(), getSz(), CV_64FC1);
		double expectedMiu = (getSz()-1U)/2.,
			expectedSdev = 0., aux;
		for(unsigned i = 0U; i<getSz(); ++i) {
			patchD255.row(i) = (double)i;
			aux = i-expectedMiu;
			expectedSdev += aux*aux;
		}
		expectedSdev *= 2.;
		expectedSdev -= 2 * expectedMiu*expectedMiu;
		expectedSdev = sqrt(expectedSdev / cnzEdge);
		res = em.assessMatch(patchD255, sd, cd, mp);
		BOOST_REQUIRE(mp.getFg() && mp.getBg() && mp.getSdevEdge());
		BOOST_TEST(*mp.getFg() == expectedMiu, test_tools::tolerance(1e-4));
		BOOST_TEST(*mp.getBg() == expectedMiu, test_tools::tolerance(1e-4));
		BOOST_TEST(*mp.getSdevEdge() == expectedSdev, test_tools::tolerance(1e-4));
		BOOST_TEST(res == 1.-expectedSdev/255, test_tools::tolerance(1e-4));
	}

	AutoTestCase1(CheckEdgeAspect_EdgyDiagGlyphAndUniformLowerTriangularPatch_ImperfectMatch, SuiteSuffix);
		ms.set_kSdevEdge(1);
		const EdgeMatch em(ms);
		const auto valRand = randUnsignedChar(1U);

		// Using a symbol with a diagonal foreground mask
		const Mat diagFgMask = Mat::diag(Mat(1, getSz(), CV_8UC1, Scalar(255U)));

		// Using a symbol with an edge mask formed by 2 neighbor diagonals of the main diagonal
		Mat sideDiagsEdgeMask = Mat::zeros(getSz(), getSz(), CV_8UC1);
		sideDiagsEdgeMask.diag(1) = 255U; // 2nd diagonal lower half
		sideDiagsEdgeMask.diag(-1) = 255U; // 2nd diagonal upper half
		const unsigned cnzEdge = 2U*(getSz() - 1U);
		BOOST_REQUIRE(countNonZero(sideDiagsEdgeMask) == cnzEdge);
		
		// Using a symbol with a background mask full except the main 3 diagonals
		Mat allBut3DiagsBgMask = Mat(getSz(), getSz(), CV_8UC1, Scalar(255U));
		allBut3DiagsBgMask.diag() = 0U;
		allBut3DiagsBgMask.diag(1) = 0U; // 2nd diagonal lower half
		allBut3DiagsBgMask.diag(-1) = 0U; // 2nd diagonal upper half
		const unsigned cnzBg = getArea() - 3U*getSz() + 2U;
		BOOST_REQUIRE(countNonZero(allBut3DiagsBgMask) == cnzBg);
		
		Mat groundedGlyph = Mat::zeros(getSz(), getSz(), CV_8UC1); // bg will stay 0
		const unsigned char edgeLevel = 125U,
							maxGlyph = edgeLevel<<1; // 250
		add(groundedGlyph, maxGlyph, groundedGlyph, diagFgMask); // set fg
		add(groundedGlyph, edgeLevel, groundedGlyph, sideDiagsEdgeMask); // set edges
		groundedGlyph.convertTo(groundedGlyph, CV_64FC1, 1./255);

		SymData sd(NOT_RELEVANT_UL, // symbol code (not relevant here)
				   NOT_RELEVANT_SZ,	// symbol index (not relevant here)
				   NOT_RELEVANT_D, // min brightness value 0..1 range (not relevant here)
				   maxGlyph/255., // diff between min..max, each in range 0..1
				   NOT_RELEVANT_D, // avgPixVal (not relevant here)
				   NOT_RELEVANT_POINT, // mc sym for original fg & bg (not relevant here)
				   SymData::IdxMatMap {
					   { ISymData::FG_MASK_IDX, diagFgMask },
					   { ISymData::BG_MASK_IDX, allBut3DiagsBgMask },
					   { ISymData::EDGE_MASK_IDX, sideDiagsEdgeMask },
					   { ISymData::GROUNDED_SYM_IDX, groundedGlyph } });

		// Testing on an uniform lower triangular patch
		patchD255 = Mat::zeros(getSz(), getSz(), CV_64FC1);
		double expectedFg = valRand, expectedBg = valRand/2.,
			expectedSdevEdge = valRand * sqrt(5) / 4.;
		for(int i = 0; i<(int)getSz(); ++i)
			patchD255.diag(-i) = (double)valRand; // i-th lower diagonal set on valRand
		BOOST_REQUIRE(countNonZero(patchD255) == (getArea() + getSz())/2);
		res = em.assessMatch(patchD255, sd, cd, mp);
		BOOST_REQUIRE(mp.getFg() && mp.getBg() && mp.getSdevEdge());
		BOOST_TEST(*mp.getFg() == expectedFg, test_tools::tolerance(1e-4));
		BOOST_TEST(*mp.getBg() == expectedBg, test_tools::tolerance(1e-4));
		BOOST_TEST(*mp.getSdevEdge() == expectedSdevEdge, test_tools::tolerance(1e-4));
		BOOST_TEST(res == 1.-expectedSdevEdge/255, test_tools::tolerance(1e-4));
	}

	AutoTestCase1(CheckBgAspect_UniformPatch_PerfectMatch, SuiteSuffix);
		ms.set_kSdevBg(1.);
		const BgMatch bm(ms);
		const auto valRand = randUnsignedChar(1U);

		// Using a symbol with a background mask full except the main 3 diagonals
		Mat allBut3DiagsBgMask = Mat(getSz(), getSz(), CV_8UC1, Scalar(255U));
		allBut3DiagsBgMask.diag() = 0U;
		allBut3DiagsBgMask.diag(1) = 0U; // 2nd diagonal lower half
		allBut3DiagsBgMask.diag(-1) = 0U; // 2nd diagonal upper half
		const unsigned cnz = getArea() - 3U*getSz() + 2U;
		BOOST_REQUIRE(countNonZero(allBut3DiagsBgMask) == cnz);
		SymData sd(NOT_RELEVANT_UL, // symbol code (not relevant here)
				   NOT_RELEVANT_SZ,	// symbol index (not relevant here)
				   NOT_RELEVANT_D, NOT_RELEVANT_D, // min and diff between min..max, each in range 0..1 (not relevant here)
				   NOT_RELEVANT_D, // avgPixVal (not relevant here)
				   NOT_RELEVANT_POINT, // mc sym for original fg & bg (not relevant here)
				   SymData::IdxMatMap {{ ISymData::BG_MASK_IDX, allBut3DiagsBgMask }});

		patchD255 = Mat(getSz(), getSz(), CV_64FC1, Scalar((double)valRand));
		res = bm.assessMatch(patchD255, sd, cd, mp);
		BOOST_REQUIRE(mp.getBg() && mp.getSdevBg());
		BOOST_TEST(*mp.getBg() == (double)valRand, test_tools::tolerance(1e-4));
		BOOST_TEST(*mp.getSdevBg() == 0., test_tools::tolerance(1e-4));
		BOOST_TEST(res == 1., test_tools::tolerance(1e-4));
	}

	AutoTestCase1(CheckBgAspect_GlyphWith3MainDiagsOn0AndPatchWithUpperHalfEmptyAndUniformLowerPart_ImperfectMatch, SuiteSuffix);
		ms.set_kSdevBg(1.);
		const BgMatch bm(ms);
		const auto valRand = randUnsignedChar(1U);

		// Using a symbol with a background mask full except the main 3 diagonals
		Mat allBut3DiagsBgMask = Mat(getSz(), getSz(), CV_8UC1, Scalar(255U));
		allBut3DiagsBgMask.diag() = 0U;
		allBut3DiagsBgMask.diag(1) = 0U; // 2nd diagonal lower half
		allBut3DiagsBgMask.diag(-1) = 0U; // 2nd diagonal upper half
		const unsigned cnz = getArea() - 3U*getSz() + 2U;
		BOOST_REQUIRE(countNonZero(allBut3DiagsBgMask) == cnz);
		SymData sd(NOT_RELEVANT_UL, // symbol code (not relevant here)
				   NOT_RELEVANT_SZ,	// symbol index (not relevant here)
				   NOT_RELEVANT_D, NOT_RELEVANT_D, // min and diff between min..max, each in range 0..1 (not relevant here)
				   NOT_RELEVANT_D, // avgPixVal (not relevant here)
				   NOT_RELEVANT_POINT, // mc sym for original fg & bg (not relevant here)
				   SymData::IdxMatMap {{ ISymData::BG_MASK_IDX, allBut3DiagsBgMask }});

		patchD255 = Mat(getSz(), getSz(), CV_64FC1, Scalar((double)valRand));
		patchD255.rowRange(0, getSz()/2) = 0.;
		res = bm.assessMatch(patchD255, sd, cd, mp);
		BOOST_REQUIRE(mp.getBg() && mp.getSdevBg());
		BOOST_TEST(*mp.getBg() == valRand/2., test_tools::tolerance(1e-4));
		BOOST_TEST(*mp.getSdevBg() == valRand/2., test_tools::tolerance(1e-4));
		BOOST_TEST(res == 1.-valRand/255., test_tools::tolerance(1e-4));
	}

	AutoTestCase1(CheckBgAspect_GlyphWith3MainDiagsOn0AndPatchRowValuesSameAsRowIndices_ImperfectMatch, SuiteSuffix);
		ms.set_kSdevBg(1.);
		const BgMatch bm(ms);
		const auto valRand = randUnsignedChar(1U);

		// Using a symbol with a background mask full except the main 3 diagonals
		Mat allBut3DiagsBgMask = Mat(getSz(), getSz(), CV_8UC1, Scalar(255U));
		allBut3DiagsBgMask.diag() = 0U;
		allBut3DiagsBgMask.diag(1) = 0U; // 2nd diagonal lower half
		allBut3DiagsBgMask.diag(-1) = 0U; // 2nd diagonal upper half
		const unsigned cnz = getArea() - 3U*getSz() + 2U;
		BOOST_REQUIRE(countNonZero(allBut3DiagsBgMask) == cnz);
		SymData sd(NOT_RELEVANT_UL, // symbol code (not relevant here)
				   NOT_RELEVANT_SZ,	// symbol index (not relevant here)
				   NOT_RELEVANT_D, NOT_RELEVANT_D, // min and diff between min..max, each in range 0..1 (not relevant here)
				   NOT_RELEVANT_D, // avgPixVal (not relevant here)
				   NOT_RELEVANT_POINT, // mc sym for original fg & bg (not relevant here)
				   SymData::IdxMatMap {{ ISymData::BG_MASK_IDX, allBut3DiagsBgMask }});

		patchD255 = Mat::zeros(getSz(), getSz(), CV_64FC1);
		double expectedMiu = (getSz()-1U)/2.,
			expectedSdev = 0., aux;
		for(unsigned i = 0U; i<getSz(); ++i) {
			patchD255.row(i) = (double)i;
			aux = i-expectedMiu;
			expectedSdev += aux*aux;
		}
		expectedSdev *= getSz() - 3.;
		expectedSdev += 2 * expectedMiu*expectedMiu;
		expectedSdev = sqrt(expectedSdev / cnz);
		res = bm.assessMatch(patchD255, sd, cd, mp);
		BOOST_REQUIRE(mp.getBg() && mp.getSdevBg());
		BOOST_TEST(*mp.getBg() == expectedMiu, test_tools::tolerance(1e-4));
		BOOST_TEST(*mp.getSdevBg() == expectedSdev, test_tools::tolerance(1e-4));
		BOOST_TEST(res == 1.-expectedSdev/127.5, test_tools::tolerance(1e-4));
	}

	AutoTestCase1(CheckContrastAspect_UniformPatch_0Contrast, SuiteSuffix);
		ms.set_kContrast(1.);
		const BetterContrast bc(ms);
		const auto valRand = randUnsignedChar(1U);

		// Using a symbol with a diagonal foreground mask
		const Mat diagFgMask = Mat::diag(Mat(1, getSz(), CV_8UC1, Scalar(255U)));

		// Using a symbol with a background mask full except the main 3 diagonals
		Mat allBut3DiagsBgMask = Mat(getSz(), getSz(), CV_8UC1, Scalar(255U));
		allBut3DiagsBgMask.diag() = 0U;
		allBut3DiagsBgMask.diag(1) = 0U; // 2nd diagonal lower half
		allBut3DiagsBgMask.diag(-1) = 0U; // 2nd diagonal upper half
		const unsigned cnz = getArea() - 3U*getSz() + 2U;
		BOOST_REQUIRE(countNonZero(allBut3DiagsBgMask) == cnz);
		SymData sd(NOT_RELEVANT_UL, // symbol code (not relevant here)
				   NOT_RELEVANT_SZ,	// symbol index (not relevant here)
				   NOT_RELEVANT_D, NOT_RELEVANT_D, // min and diff between min..max, each in range 0..1 (not relevant here)
				   NOT_RELEVANT_D, // avgPixVal (not relevant here)
				   NOT_RELEVANT_POINT, // mc sym for original fg & bg (not relevant here)
				   SymData::IdxMatMap {
					   { ISymData::FG_MASK_IDX, diagFgMask },
					   { ISymData::BG_MASK_IDX, allBut3DiagsBgMask }});

		// Testing on a uniform patch
		patchD255 = Mat(getSz(), getSz(), CV_64FC1, Scalar((double)valRand));
		res = bc.assessMatch(patchD255, sd, cd, mp);
		BOOST_REQUIRE(mp.getFg() && mp.getBg());
		BOOST_TEST(*mp.getFg() == (double)valRand, test_tools::tolerance(1e-4));
		BOOST_TEST(*mp.getBg() == (double)valRand, test_tools::tolerance(1e-4));
		BOOST_TEST(res == 0., test_tools::tolerance(1e-4));
	}

	AutoTestCase1(CheckContrastAspect_EdgyDiagGlyphAndDiagPatchWithMaxContrast_ContrastFromPatch, SuiteSuffix);
		ms.set_kContrast(1.);
		const BetterContrast bc(ms);
		const auto valRand = randUnsignedChar(1U);

		// Using a symbol with a diagonal foreground mask
		const Mat diagFgMask = Mat::diag(Mat(1, getSz(), CV_8UC1, Scalar(255U)));

		// Using a symbol with a background mask full except the main 3 diagonals
		Mat allBut3DiagsBgMask = Mat(getSz(), getSz(), CV_8UC1, Scalar(255U));
		allBut3DiagsBgMask.diag() = 0U;
		allBut3DiagsBgMask.diag(1) = 0U; // 2nd diagonal lower half
		allBut3DiagsBgMask.diag(-1) = 0U; // 2nd diagonal upper half
		const unsigned cnz = getArea() - 3U*getSz() + 2U;
		BOOST_REQUIRE(countNonZero(allBut3DiagsBgMask) == cnz);
		SymData sd(NOT_RELEVANT_UL, // symbol code (not relevant here)
				   NOT_RELEVANT_SZ,	// symbol index (not relevant here)
				   NOT_RELEVANT_D, NOT_RELEVANT_D, // min and diff between min..max, each in range 0..1 (not relevant here)
				   NOT_RELEVANT_D, // avgPixVal (not relevant here)
				   NOT_RELEVANT_POINT, // mc sym for original fg & bg (not relevant here)
				   SymData::IdxMatMap {
					   { ISymData::FG_MASK_IDX, diagFgMask },
					   { ISymData::BG_MASK_IDX, allBut3DiagsBgMask }});

		patchD255 = Mat::diag(Mat(1, getSz(), CV_64FC1, Scalar(255.)));
		res = bc.assessMatch(patchD255, sd, cd, mp);
		BOOST_REQUIRE(mp.getFg() && mp.getBg());
		BOOST_TEST(*mp.getFg() == 255., test_tools::tolerance(1e-4));
		BOOST_TEST(*mp.getBg() == 0., test_tools::tolerance(1e-4));
		BOOST_TEST(res == 1., test_tools::tolerance(1e-4));
	}

	AutoTestCase1(CheckContrastAspect_EdgyDiagGlyphAndDiagPatchWithHalfContrast_ContrastFromPatch, SuiteSuffix);
		ms.set_kContrast(1.);
		const BetterContrast bc(ms);
		const auto valRand = randUnsignedChar(1U);

		// Using a symbol with a diagonal foreground mask
		const Mat diagFgMask = Mat::diag(Mat(1, getSz(), CV_8UC1, Scalar(255U)));

		// Using a symbol with a background mask full except the main 3 diagonals
		Mat allBut3DiagsBgMask = Mat(getSz(), getSz(), CV_8UC1, Scalar(255U));
		allBut3DiagsBgMask.diag() = 0U;
		allBut3DiagsBgMask.diag(1) = 0U; // 2nd diagonal lower half
		allBut3DiagsBgMask.diag(-1) = 0U; // 2nd diagonal upper half
		const unsigned cnz = getArea() - 3U*getSz() + 2U;
		BOOST_REQUIRE(countNonZero(allBut3DiagsBgMask) == cnz);
		SymData sd(NOT_RELEVANT_UL, // symbol code (not relevant here)
				   NOT_RELEVANT_SZ,	// symbol index (not relevant here)
				   NOT_RELEVANT_D, NOT_RELEVANT_D, // min and diff between min..max, each in range 0..1 (not relevant here)
				   NOT_RELEVANT_D, // avgPixVal (not relevant here)
				   NOT_RELEVANT_POINT, // mc sym for original fg & bg (not relevant here)
				   SymData::IdxMatMap {
					   { ISymData::FG_MASK_IDX, diagFgMask },
					   { ISymData::BG_MASK_IDX, allBut3DiagsBgMask }});

		patchD255 = Mat::diag(Mat(1, getSz(), CV_64FC1, Scalar(127.5)));
		res = bc.assessMatch(patchD255, sd, cd, mp);
		BOOST_REQUIRE(mp.getFg() && mp.getBg());
		BOOST_TEST(*mp.getFg() == 127.5, test_tools::tolerance(1e-4));
		BOOST_TEST(*mp.getBg() == 0., test_tools::tolerance(1e-4));
		BOOST_TEST(res == .5, test_tools::tolerance(1e-4));
	}

	AutoTestCase1(CheckGravitationalSmoothness_PatchAndGlyphArePixelsOnOppositeCorners_ImperfectMatch, SuiteSuffix);
		ms.set_kMCsOffset(1.);
		const unsigned sz_1 = getSz()-1U;
		cd.useNewSymSize(getSz());
		const GravitationalSmoothness gs(ms);

		// Checking a symbol that has a single 255 pixel in bottom right corner
		double avgPixVal = 1. / getArea(); // a single pixel set to max
		Point2d origMcSym(1., 1.);
		Mat fgMask = Mat::zeros(getSz(), getSz(), CV_8UC1),
			bgMask(getSz(), getSz(), CV_8UC1, Scalar(255U));
		fgMask.at<unsigned char>(sz_1, sz_1) = 255U;
		bgMask.at<unsigned char>(sz_1, sz_1) = 0U;

		SymData sd(NOT_RELEVANT_UL, // symbol code (not relevant here)
				   NOT_RELEVANT_SZ,	// symbol index (not relevant here)
				   NOT_RELEVANT_D, NOT_RELEVANT_D, // min and diff between min..max, each in range 0..1 (not relevant here)
				   avgPixVal,
				   origMcSym,
				   SymData::IdxMatMap {
					   { ISymData::FG_MASK_IDX, fgMask },
					   { ISymData::BG_MASK_IDX, bgMask }});

		// Using a patch with a single 255 pixel in top left corner
		patchD255 = Mat::zeros(getSz(), getSz(), CV_64FC1);
		patchD255.at<double>(0, 0) = 255.;
		res = gs.assessMatch(patchD255, sd, cd, mp);
		BOOST_REQUIRE(mp.getFg() && mp.getBg() && mp.getSymDensity() && mp.getMcPatchApprox() && mp.getMcPatch());
		BOOST_TEST(*mp.getFg() == 0., test_tools::tolerance(1e-4));
		BOOST_TEST(*mp.getBg() == 255./(getArea()-1), test_tools::tolerance(1e-4));
		BOOST_TEST(*mp.getSymDensity() == 1./getArea(), test_tools::tolerance(1e-8));

		// Symbol's mc migrated diagonally from bottom-right corner to above the center of patch.
		// Migration occurred due to the new fg & bg of the symbol
		BOOST_TEST(mp.getMcPatchApprox()->x == .5 - .5/(pow(getSz(), 2) - 1U), test_tools::tolerance(1e-8));
		BOOST_TEST(mp.getMcPatchApprox()->y == .5 - .5/(pow(getSz(), 2) - 1U), test_tools::tolerance(1e-8));
		BOOST_TEST(mp.getMcPatch()->x == 0., test_tools::tolerance(1e-4));
		BOOST_TEST(mp.getMcPatch()->y == 0., test_tools::tolerance(1e-4));
		BOOST_TEST(res == 1. + (CachedData::MassCenters::preferredMaxMcDist() - sqrt(2)*.5*(1. - 1./(pow(getSz(), 2) - 1U))) *
				   CachedData::MassCenters::invComplPrefMaxMcDist(), test_tools::tolerance(1e-8));
	}

	AutoTestCase1(CheckGravitationalSmoothness_CornerPixelAsGlyphAndCenterOfEdgeAsPatch_McGlyphCenters, SuiteSuffix);
		ms.set_kMCsOffset(1.);
		const unsigned sz_1 = getSz()-1U;
		cd.useNewSymSize(getSz());
		const GravitationalSmoothness gs(ms);

		// Checking a symbol that has a single 255 pixel in bottom right corner
		double avgPixVal = 1. / getArea(); // a single pixel set to max
		Point2d origMcSym(1., 1.);
		Mat fgMask = Mat::zeros(getSz(), getSz(), CV_8UC1),
			bgMask(getSz(), getSz(), CV_8UC1, Scalar(255U));
		fgMask.at<unsigned char>(sz_1, sz_1) = 255U;
		bgMask.at<unsigned char>(sz_1, sz_1) = 0U;

		SymData sd(NOT_RELEVANT_UL, // symbol code (not relevant here)
				   NOT_RELEVANT_SZ,	// symbol index (not relevant here)
				   NOT_RELEVANT_D, NOT_RELEVANT_D, // min and diff between min..max, each in range 0..1 (not relevant here)
				   avgPixVal,
				   origMcSym,
				   SymData::IdxMatMap {
					   { ISymData::FG_MASK_IDX, fgMask },
					   { ISymData::BG_MASK_IDX, bgMask } });

		// Using a patch with the middle pixels pair on the top row on 255
		// Patch mc is at half width on top row.
		patchD255 = Mat::zeros(getSz(), getSz(), CV_64FC1);
		Mat(patchD255, Rect(getSz()/2U-1U, 0, 2, 1)) = Scalar(255.);
		BOOST_REQUIRE(countNonZero(patchD255) == 2);
		res = gs.assessMatch(patchD255, sd, cd, mp);
		BOOST_REQUIRE(mp.getFg() && mp.getBg() && mp.getSymDensity() && mp.getMcPatchApprox() && mp.getMcPatch());
		BOOST_TEST(*mp.getFg() == 0., test_tools::tolerance(1e-4));
		BOOST_TEST(*mp.getBg() == 2*255./(getArea()-1), test_tools::tolerance(1e-4));
		BOOST_TEST(*mp.getSymDensity() == 1./getArea(), test_tools::tolerance(1e-8));

		// Symbol's mc migrated diagonally from bottom-right corner to above the center of patch.
		// Migration occurred due to the new fg & bg of the symbol
		BOOST_TEST(mp.getMcPatchApprox()->x == .5 - .5/(pow(getSz(), 2) - 1U), test_tools::tolerance(1e-8));
		BOOST_TEST(mp.getMcPatchApprox()->y == .5 - .5/(pow(getSz(), 2) - 1U), test_tools::tolerance(1e-8));
		BOOST_TEST(mp.getMcPatch()->x == .5, test_tools::tolerance(1e-4));
		BOOST_TEST(mp.getMcPatch()->y == 0., test_tools::tolerance(1e-4));
		double dx = .5/(pow(getSz(), 2) - 1U), dy = .5*(1. - 1./(pow(getSz(), 2) - 1U));
		BOOST_TEST(res == 1. + (CachedData::MassCenters::preferredMaxMcDist() - sqrt(dx*dx + dy*dy)) *
				   CachedData::MassCenters::invComplPrefMaxMcDist(), test_tools::tolerance(1e-8));
	}

	AutoTestCase1(CheckGravitationalSmoothness_CornerPixelAsGlyphAndOtherCornerAsPatch_McGlyphCenters, SuiteSuffix);
		ms.set_kMCsOffset(1.);
		const unsigned sz_1 = getSz()-1U;
		cd.useNewSymSize(getSz());
		const GravitationalSmoothness gs(ms);

		// Checking a symbol that has a single 255 pixel in bottom right corner
		double avgPixVal = 1. / getArea(); // a single pixel set to max
		Point2d origMcSym(1., 1.);
		Mat fgMask = Mat::zeros(getSz(), getSz(), CV_8UC1),
			bgMask(getSz(), getSz(), CV_8UC1, Scalar(255U));
		fgMask.at<unsigned char>(sz_1, sz_1) = 255U;
		bgMask.at<unsigned char>(sz_1, sz_1) = 0U;

		SymData sd(NOT_RELEVANT_UL, // symbol code (not relevant here)
				   NOT_RELEVANT_SZ,	// symbol index (not relevant here)
				   NOT_RELEVANT_D, NOT_RELEVANT_D, // min and diff between min..max, each in range 0..1 (not relevant here)
				   avgPixVal,
				   origMcSym,
				   SymData::IdxMatMap {
					   { ISymData::FG_MASK_IDX, fgMask },
					   { ISymData::BG_MASK_IDX, bgMask } });

		// Using a patch with the last pixel on the top row on 255
		patchD255 = Mat::zeros(getSz(), getSz(), CV_64FC1);
		patchD255.at<double>(0, sz_1) = 255.;
		res = gs.assessMatch(patchD255, sd, cd, mp);
		BOOST_REQUIRE(mp.getFg() && mp.getBg() && mp.getSymDensity() && mp.getMcPatchApprox() && mp.getMcPatch());
		BOOST_TEST(*mp.getFg() == 0., test_tools::tolerance(1e-4));
		BOOST_TEST(*mp.getBg() == 255./(getArea()-1), test_tools::tolerance(1e-4));
		BOOST_TEST(*mp.getSymDensity() == 1./getArea(), test_tools::tolerance(1e-8));

		// Symbol's mc migrated diagonally from bottom-right corner to above the center of patch.
		// Migration occurred due to the new fg & bg of the symbol
		BOOST_TEST(mp.getMcPatchApprox()->x == .5 - .5/(pow(getSz(), 2) - 1U), test_tools::tolerance(1e-8));
		BOOST_TEST(mp.getMcPatchApprox()->y == .5 - .5/(pow(getSz(), 2) - 1U), test_tools::tolerance(1e-8));
		BOOST_TEST(mp.getMcPatch()->x == 1., test_tools::tolerance(1e-4));
		BOOST_TEST(mp.getMcPatch()->y == 0., test_tools::tolerance(1e-4));
		double dx = .5*(1. + 1/(pow(getSz(), 2) - 1U)), dy = .5*(1. - 1/(pow(getSz(), 2) - 1U));
		BOOST_TEST(res == 1. + (CachedData::MassCenters::preferredMaxMcDist() - sqrt(dx*dx + dy*dy)) *
				   CachedData::MassCenters::invComplPrefMaxMcDist(), test_tools::tolerance(1e-8));
	}

	AutoTestCase1(CheckDirectionalSmoothness_PatchAndGlyphArePixelsOnOppositeCorners_ImperfectMatch, SuiteSuffix);
		ms.set_kCosAngleMCs(1.);
		const unsigned sz_1 = getSz()-1U;
		cd.useNewSymSize(getSz());
		const DirectionalSmoothness ds(ms);

		// Checking a symbol that has a single 255 pixel in bottom right corner
		double avgPixVal = 1. / getArea(); // a single pixel set to max
		Point2d origMcSym(1., 1.);
		Mat fgMask = Mat::zeros(getSz(), getSz(), CV_8UC1), bgMask(getSz(), getSz(), CV_8UC1, Scalar(255U));
		fgMask.at<unsigned char>(sz_1, sz_1) = 255U;
		bgMask.at<unsigned char>(sz_1, sz_1) = 0U;

		SymData sd(NOT_RELEVANT_UL, // symbol code (not relevant here)
				   NOT_RELEVANT_SZ,	// symbol index (not relevant here)
				   NOT_RELEVANT_D, NOT_RELEVANT_D, // min and diff between min..max, each in range 0..1 (not relevant here)
				   avgPixVal,
				   origMcSym,
				   SymData::IdxMatMap {
					   { ISymData::FG_MASK_IDX, fgMask },
					   { ISymData::BG_MASK_IDX, bgMask } });

		// Using a patch with a single 255 pixel in top left corner
		// Same as 1st scenario from Gravitational Smoothness
		patchD255 = Mat::zeros(getSz(), getSz(), CV_64FC1);
		patchD255.at<double>(0, 0) = 255.;
		res = ds.assessMatch(patchD255, sd, cd, mp);
		BOOST_REQUIRE(mp.getFg() && mp.getBg() && mp.getSymDensity() && mp.getMcPatchApprox() && mp.getMcPatch());
		// Symbol's mc migrated diagonally from bottom-right corner to above the center of patch.
		// Migration occurred due to the new fg & bg of the symbol
		BOOST_TEST(mp.getMcPatchApprox()->x == .5 - .5/(pow(getSz(), 2) - 1U), test_tools::tolerance(1e-8));
		BOOST_TEST(mp.getMcPatchApprox()->y == .5 - .5/(pow(getSz(), 2) - 1U), test_tools::tolerance(1e-8));
		BOOST_TEST(mp.getMcPatch()->x == 0., test_tools::tolerance(1e-4));
		BOOST_TEST(mp.getMcPatch()->y == 0., test_tools::tolerance(1e-4));
		BOOST_TEST(res == 2.*(2.-sqrt(2.)) * (CachedData::MassCenters::a_mcsOffsetFactor() * *mp.getMcsOffset() + CachedData::MassCenters::b_mcsOffsetFactor()),
				   test_tools::tolerance(1e-8)); // angle = 0 => cos = 1
	}

	AutoTestCase1(CheckDirectionalSmoothness_CornerPixelAsGlyphAndCenterOfEdgeAsPatch_McGlyphCenters, SuiteSuffix);
		ms.set_kCosAngleMCs(1.);
		const unsigned sz_1 = getSz()-1U;
		cd.useNewSymSize(getSz());
		const DirectionalSmoothness ds(ms);

		// Checking a symbol that has a single 255 pixel in bottom right corner
		double avgPixVal = 1. / getArea(); // a single pixel set to max
		Point2d origMcSym(1., 1.);
		Mat fgMask = Mat::zeros(getSz(), getSz(), CV_8UC1), bgMask(getSz(), getSz(), CV_8UC1, Scalar(255U));
		fgMask.at<unsigned char>(sz_1, sz_1) = 255U;
		bgMask.at<unsigned char>(sz_1, sz_1) = 0U;

		SymData sd(NOT_RELEVANT_UL, // symbol code (not relevant here)
				   NOT_RELEVANT_SZ,	// symbol index (not relevant here)
				   NOT_RELEVANT_D, NOT_RELEVANT_D, // min and diff between min..max, each in range 0..1 (not relevant here)
				   avgPixVal,
				   origMcSym,
				   SymData::IdxMatMap {
					   { ISymData::FG_MASK_IDX, fgMask },
					   { ISymData::BG_MASK_IDX, bgMask } });

		// Using a patch with the middle pixels pair on the top row on 255
		// Patch mc is at half width on top row.
		// Same as 2nd scenario from Gravitational Smoothness
		patchD255 = Mat::zeros(getSz(), getSz(), CV_64FC1);
		Mat(patchD255, Rect(getSz()/2U-1U, 0, 2, 1)) = Scalar(255.);
		res = ds.assessMatch(patchD255, sd, cd, mp);
		BOOST_REQUIRE(mp.getFg() && mp.getBg() && mp.getSymDensity() && mp.getMcPatchApprox() && mp.getMcPatch());
		// Symbol's mc migrated diagonally from bottom-right corner to above the center of patch.
		// Migration occurred due to the new fg & bg of the symbol
		BOOST_TEST(mp.getMcPatchApprox()->x == .5 - .5/(pow(getSz(), 2) - 1U), test_tools::tolerance(1e-8));
		BOOST_TEST(mp.getMcPatchApprox()->y == .5 - .5/(pow(getSz(), 2) - 1U), test_tools::tolerance(1e-8));
		BOOST_TEST(mp.getMcPatch()->x == .5, test_tools::tolerance(1e-4));
		BOOST_TEST(mp.getMcPatch()->y == 0., test_tools::tolerance(1e-4));
		BOOST_TEST(res == (CachedData::MassCenters::a_mcsOffsetFactor() * *mp.getMcsOffset() + CachedData::MassCenters::b_mcsOffsetFactor()),
				   test_tools::tolerance(1e-8)); // angle = 45 => cos = sqrt(2)/2
	}

	AutoTestCase1(CheckDirectionalSmoothness_CornerPixelAsGlyphAndOtherCornerAsPatch_McGlyphCenters, SuiteSuffix);
		ms.set_kCosAngleMCs(1.);
		const unsigned sz_1 = getSz()-1U;
		cd.useNewSymSize(getSz());
		const DirectionalSmoothness ds(ms);

		// Checking a symbol that has a single 255 pixel in bottom right corner
		double avgPixVal = 1. / getArea(); // a single pixel set to max
		Point2d origMcSym(1., 1.);
		Mat fgMask = Mat::zeros(getSz(), getSz(), CV_8UC1), bgMask(getSz(), getSz(), CV_8UC1, Scalar(255U));
		fgMask.at<unsigned char>(sz_1, sz_1) = 255U;
		bgMask.at<unsigned char>(sz_1, sz_1) = 0U;

		SymData sd(NOT_RELEVANT_UL, // symbol code (not relevant here)
				   NOT_RELEVANT_SZ,	// symbol index (not relevant here)
				   NOT_RELEVANT_D, NOT_RELEVANT_D, // min and diff between min..max, each in range 0..1 (not relevant here)
				   avgPixVal,
				   origMcSym,
				   SymData::IdxMatMap {
					   { ISymData::FG_MASK_IDX, fgMask },
					   { ISymData::BG_MASK_IDX, bgMask } });

		// Using a patch with the last pixel on the top row on 255
		// Same as 3rd scenario from Gravitational Smoothness
		patchD255 = Mat::zeros(getSz(), getSz(), CV_64FC1);
		patchD255.at<double>(0, sz_1) = 255.;
		res = ds.assessMatch(patchD255, sd, cd, mp);
		BOOST_REQUIRE(mp.getFg() && mp.getBg() && mp.getSymDensity() && mp.getMcPatchApprox() && mp.getMcPatch());
		// Symbol's mc migrated diagonally from bottom-right corner to above the center of patch.
		// Migration occurred due to the new fg & bg of the symbol
		BOOST_TEST(mp.getMcPatchApprox()->x == .5 - .5/(pow(getSz(), 2) - 1U), test_tools::tolerance(1e-8));
		BOOST_TEST(mp.getMcPatchApprox()->y == .5 - .5/(pow(getSz(), 2) - 1U), test_tools::tolerance(1e-8));
		BOOST_TEST(mp.getMcPatch()->x == 1., test_tools::tolerance(1e-4));
		BOOST_TEST(mp.getMcPatch()->y == 0., test_tools::tolerance(1e-4));
		BOOST_TEST(res == (2.-sqrt(2.)) * (CachedData::MassCenters::a_mcsOffsetFactor() * *mp.getMcsOffset() + CachedData::MassCenters::b_mcsOffsetFactor()),
				   test_tools::tolerance(1e-8)); // angle is 90 => cos = 0
	}

	AutoTestCase1(CheckLargerSymAspect_EmptyGlyph_Density0, SuiteSuffix);
		ms.set_kSymDensity(1.);
		cd.smallGlyphsCoverage = .1; // large glyphs need to cover more than 10% of their box
		const LargerSym ls(ms);

		SymData sd(NOT_RELEVANT_UL, // symbol code (not relevant here)
				   NOT_RELEVANT_SZ,	// symbol index (not relevant here)
				   NOT_RELEVANT_D, NOT_RELEVANT_D, // min and diff between min..max, each in range 0..1 (not relevant here)
				   0., // avgPixVal (INITIALLY, AN EMPTY SYMBOL IS CONSIDERED)
				   NOT_RELEVANT_POINT, // mc sym for original fg & bg (not relevant here)
				   SymData::IdxMatMap {});

		// Testing with an empty symbol (sd.avgPixVal == 0)
		res = ls.assessMatch(NOT_RELEVANT_MAT, sd, cd, mp);
		BOOST_REQUIRE(mp.getSymDensity());
		BOOST_TEST(*mp.getSymDensity() == 0., test_tools::tolerance(1e-4));
		BOOST_TEST(res == 1.-cd.getSmallGlyphsCoverage(), test_tools::tolerance(1e-4));
	}

	AutoTestCase1(CheckLargerSymAspect_InferiorLimitOfLargeSymbols_QualifiesAsLarge, SuiteSuffix);
		ms.set_kSymDensity(1.);
		cd.smallGlyphsCoverage = .1; // large glyphs need to cover more than 10% of their box
		const LargerSym ls(ms);

		SymData sd(NOT_RELEVANT_UL, // symbol code (not relevant here)
				   NOT_RELEVANT_SZ,	// symbol index (not relevant here)
				   NOT_RELEVANT_D, NOT_RELEVANT_D, // min and diff between min..max, each in range 0..1 (not relevant here)
				   cd.smallGlyphsCoverage, // avgPixVal (symbol that just enters the 'large symbols' category)
				   NOT_RELEVANT_POINT, // mc sym for original fg & bg (not relevant here)
				   SymData::IdxMatMap {});

		res = ls.assessMatch(NOT_RELEVANT_MAT, sd, cd, mp);
		BOOST_REQUIRE(mp.getSymDensity());
		BOOST_TEST(*mp.getSymDensity() == cd.getSmallGlyphsCoverage(), test_tools::tolerance(1e-4));
		BOOST_TEST(res == 1., test_tools::tolerance(1e-4));
	}

	AutoTestCase1(CheckLargerSymAspect_LargestPosibleSymbol_LargestScore, SuiteSuffix);
		ms.set_kSymDensity(1.);
		cd.smallGlyphsCoverage = .1; // large glyphs need to cover more than 10% of their box
		const LargerSym ls(ms);

		SymData sd(NOT_RELEVANT_UL, // symbol code (not relevant here)
				   NOT_RELEVANT_SZ,	// symbol index (not relevant here)
				   NOT_RELEVANT_D, NOT_RELEVANT_D, // min and diff between min..max, each in range 0..1 (not relevant here)
				   1., // avgPixVal (largest possible symbol)
				   NOT_RELEVANT_POINT, // mc sym for original fg & bg (not relevant here)
				   SymData::IdxMatMap {});

		res = ls.assessMatch(NOT_RELEVANT_MAT, sd, cd, mp);
		BOOST_REQUIRE(mp.getSymDensity());
		BOOST_TEST(*mp.getSymDensity() == 1., test_tools::tolerance(1e-4));
		BOOST_TEST(res == 2.-cd.getSmallGlyphsCoverage(), test_tools::tolerance(1e-4));
	}
BOOST_AUTO_TEST_SUITE_END() // MatchAspects_Tests

#endif // BOOST_PP_IS_ITERATING
