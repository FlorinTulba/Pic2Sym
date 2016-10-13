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
#include "fontEngine.h"
#include "settings.h"
#include "controller.h"
#include "jobMonitor.h"

#include <iostream>
#include <numeric>
#include <random>

using namespace std;
using namespace boost;
using namespace cv;

extern const bool PreserveRemovableSymbolsForExamination;

namespace ut {
	/// Fixture reducing some declarative effort from tests
	class FontEngineFixtComputations : public Fixt {
		unsigned sz;			///< patch side length. Use setSz and getSz to access it within tests
		double area;			///< square of sz
		double maxGlyphSum;		///< max pixel sum for a glyph
		Mat consec, revConsec;	///< column/row vectors of consecutive values (updated by setSz; getters available)

	protected:
		vector<unsigned char> pixels;		///< Data defining the glyph. Uses ASCENDING vertical axis
		unsigned char rows = 0U, cols = 0U; ///< pixels will have rows x cols elements
		unsigned char left = 0U, top = 0U;	///< location of glyph within the wrapping square

	public:
		double apv = 0.;	///< average pixel value to be computed within each test case
		Point2d mc;			///< mass-center to be computed within each test case

		const Mat& getConsec() const { return consec; }
		const Mat& getRevConsec() const { return revConsec; }
		unsigned getSz() const { return sz; }
		double getArea() const { return area; }
		double getMaxGlyphSum() const { return maxGlyphSum; }

		/// Setter of sz. Updates consec and revConsec
		void setSz(unsigned sz_) {
			sz = sz_;
			area = double(sz * sz);
			maxGlyphSum = 255. * area;
			consec = Mat(1, sz_, CV_64FC1);
			iota(BOUNDS_FOR_ITEM_TYPE(consec, double), (double)0.); // 0..sz-1
			flip(consec, revConsec, 1);	// sz-1..0
			revConsec = revConsec.t();
		}

		/// Creates a fixture with the provided sz value
		FontEngineFixtComputations(unsigned sz_ = 10U) : Fixt() {
			setSz(sz_);
		}
	};

	/// Provides a FontEngine object to the tests using this specialized fixture
	class FontEngineFixtConfig : public Fixt {
		Settings s;		///< default settings. Tests shouldn't touch it
		::Controller c;	///< controller using default settings. Tests shouldn't touch it

	protected:
		JobMonitor jm;
		FontEngine *pfe = nullptr; ///< pointer to the FontEngine object needed within tests

		/// initial value of PreserveRemovableSymbolsForExamination
		bool origPreserveRemovableSymbolsForExamination = PreserveRemovableSymbolsForExamination;

		/// reference to non-const PreserveRemovableSymbolsForExamination
		bool &refPreserveRemovableSymbolsForExamination = const_cast<bool&>(PreserveRemovableSymbolsForExamination);

	public:
		/// Creates the fixture providing a FontEngine object
		FontEngineFixtConfig() : Fixt(), s(), c(s) {
			try {
				pfe = &c.getFontEngine(s.symSettings()).useSymsMonitor(jm);

				// Forcing PreserveRemovableSymbolsForExamination on true during each test case.
				// Disposing removable symbols (marked by symbol filters) would provide a variable
				// set of input symbols to the tests, which would need to update some expected values
				// for each change within the filter configuration.
				// Thus, keeping all removable symbols lets these tests unaffected by filter changes.
				// Besides, testing on all symbols is sufficient to check the correctness of the covered cases.
				if(!PreserveRemovableSymbolsForExamination)
					refPreserveRemovableSymbolsForExamination = true;
			} catch(runtime_error&) {
				cerr<<"Couldn't create FontEngine"<<endl;
			}
		}

		/// Reestablishes old value of PreserveRemovableSymbolsForExamination
		~FontEngineFixtConfig() {
			if(PreserveRemovableSymbolsForExamination != origPreserveRemovableSymbolsForExamination)
				refPreserveRemovableSymbolsForExamination = origPreserveRemovableSymbolsForExamination;
		}
	};
}

// Test suite checking PixMapSym functionality
BOOST_FIXTURE_TEST_SUITE(FontEngine_Tests_Computations, ut::FontEngineFixtComputations)
	BOOST_AUTO_TEST_CASE(ComputeMassCenterAndGlyphSum_0RowsOfData_CenterAnd0) {
		BOOST_TEST_MESSAGE("Running ComputeMassCenterAndGlyphSum_0RowsOfData_CenterAnd0");
		cols = 5U; top = getSz()-1U;

		PixMapSym::computeMcAndAvgPixVal(getSz(), getMaxGlyphSum(), pixels, rows, cols, left, top,
										 getConsec(), getRevConsec(), mc, apv); // measured based on a DESCENDING vertical axis
		BOOST_REQUIRE(apv == 0.);
		BOOST_TEST(mc.x == .5, test_tools::tolerance(1e-4));
		BOOST_TEST(mc.y == .5, test_tools::tolerance(1e-4));
	}

	BOOST_AUTO_TEST_CASE(ComputeMassCenterAndGlyphSum_0ColumnsOfData_CenterAnd0) {
		BOOST_TEST_MESSAGE("Running ComputeMassCenterAndGlyphSum_0ColumnsOfData_CenterAnd0");
		rows = 4U; top = getSz()-1U;

		PixMapSym::computeMcAndAvgPixVal(getSz(), getMaxGlyphSum(), pixels, rows, cols, left, top, 
										 getConsec(), getRevConsec(), mc, apv); // measured based on a DESCENDING vertical axis
		BOOST_REQUIRE(apv == 0.);
		BOOST_TEST(mc.x == .5, test_tools::tolerance(1e-4));
		BOOST_TEST(mc.y == .5, test_tools::tolerance(1e-4));
	}

	BOOST_AUTO_TEST_CASE(ComputeMassCenterAndGlyphSum_AllDataIs0_CenterAnd0) {
		BOOST_TEST_MESSAGE("Running ComputeMassCenterAndGlyphSum_AllDataIs0_CenterAnd0");
		rows = cols = 5U; top = getSz()-1U;
		pixels.assign(rows*cols, 0U);

		PixMapSym::computeMcAndAvgPixVal(getSz(), getMaxGlyphSum(), pixels, rows, cols, left, top, 
										 getConsec(), getRevConsec(), mc, apv); // measured based on a DESCENDING vertical axis
		BOOST_TEST(apv == 0., test_tools::tolerance(1e-4));
		BOOST_TEST(mc.x == .5, test_tools::tolerance(1e-4));
		BOOST_TEST(mc.y == .5, test_tools::tolerance(1e-4));
	}

	BOOST_AUTO_TEST_CASE(ComputeMassCenterAndGlyphSum_2ChosenPixels_ExpectedValues) {
		BOOST_TEST_MESSAGE("Running ComputeMassCenterAndGlyphSum_2ChosenPixels_ExpectedValues");
		rows = 1U; cols = 7U; left = 2U;
		pixels.assign(rows*cols, 0U);
		// 2 fixed points at a distance of 6: 170(2, 0) and 85(8, 0)
		pixels[0] = 170; pixels[cols-1] = 85; // 170 = 85*2, and 170 + 85 = 255

		PixMapSym::computeMcAndAvgPixVal(getSz(), getMaxGlyphSum(), pixels, rows, cols, left, top, 
										 getConsec(), getRevConsec(), mc, apv);
		BOOST_TEST(apv == 1. / getArea(), test_tools::tolerance(1e-4)); // 170 + 85 = 255
		// mc is measured based on a DESCENDING vertical axis
		BOOST_TEST(mc.x == 4. / (getSz() - 1U), test_tools::tolerance(1e-4)); // 4 is at one third the distance 2..8
		BOOST_TEST(mc.y == 0., test_tools::tolerance(1e-4));
	}

	BOOST_AUTO_TEST_CASE(ComputeMassCenterAndGlyphSum_UniformPatch_CenterAndAverage) {
		BOOST_TEST_MESSAGE("Running ComputeMassCenterAndGlyphSum_UniformPatch_CenterAndAverage");
		rows = cols = getSz(); top = getSz()-1U;
		const auto uc = ut::randUnsignedChar(1U);
		cout<<"Checking patch filled with value "<<(unsigned)uc<<endl;
		pixels.assign(rows*cols, uc);

		PixMapSym::computeMcAndAvgPixVal(getSz(), getMaxGlyphSum(), pixels, rows, cols, left, top, 
										 getConsec(), getRevConsec(), mc, apv);
		BOOST_TEST(apv == getArea()*uc / getMaxGlyphSum(), test_tools::tolerance(1e-4));
		// mc is measured based on a DESCENDING vertical axis
		BOOST_TEST(mc.x == .5, test_tools::tolerance(1e-4));
		BOOST_TEST(mc.y == .5, test_tools::tolerance(1e-4));
	}

	BOOST_AUTO_TEST_CASE(ComputeMassCenterAndGlyphSum_SinglePixelNon0_PixelPositionAndPixelValueDiv255) {
		BOOST_TEST_MESSAGE("Running ComputeMassCenterAndGlyphSum_SinglePixelNon0_PixelPositionAndPixelValueDiv255");
		rows = cols = 1U;
		left = ut::randUnsignedChar(0U, getSz()-1U);
		top = ut::randUnsignedChar(0U, getSz()-1U);
		const auto uc = ut::randUnsignedChar(1U);
		pixels.push_back(uc);
		cout<<"Checking patch with a single non-zero pixel at: top="
			<<(unsigned)top<<", left="<<(unsigned)left<<endl;

		PixMapSym::computeMcAndAvgPixVal(getSz(), getMaxGlyphSum(), pixels, rows, cols, left, top,
										 getConsec(), getRevConsec(), mc, apv);
		BOOST_TEST(apv == uc / getMaxGlyphSum(), test_tools::tolerance(1e-4));
		// mc is measured based on a DESCENDING vertical axis
		BOOST_TEST(mc.x == (double)left / (getSz() - 1U), test_tools::tolerance(1e-4));
		BOOST_TEST(mc.y == (double)top / (getSz() - 1U), test_tools::tolerance(1e-4));
	}

	BOOST_AUTO_TEST_CASE(ComputeMassCenterAndGlyphSum_3by3UniformArea_CenterOfAreaAnd9MulPixelValueDiv255) {
		BOOST_TEST_MESSAGE("Running ComputeMassCenterAndGlyphSum_3by3UniformArea_CenterOfAreaAnd9MulPixelValueDiv255");
		rows = cols = 3U;
		left = ut::randUnsignedChar(0U, getSz()-3U);
		top = ut::randUnsignedChar(2U, getSz()-1U);
		const auto uc = ut::randUnsignedChar(1U);
		pixels.assign(rows*cols, uc); // all pixels are 'uc'

		PixMapSym::computeMcAndAvgPixVal(getSz(), getMaxGlyphSum(), pixels, rows, cols, left, top,
										 getConsec(), getRevConsec(), mc, apv);
		BOOST_TEST(apv == rows*cols*uc / getMaxGlyphSum(), test_tools::tolerance(1e-4));
		// mc is measured based on a DESCENDING vertical axis
		BOOST_TEST(mc.x == (double)(left + 1U) / (getSz() - 1U), test_tools::tolerance(1e-4));
		BOOST_TEST(mc.y == (double)(top - 1U) / (getSz() - 1U), test_tools::tolerance(1e-4));
	}

	BOOST_AUTO_TEST_CASE(ComputeMassCenterAndGlyphSum_2RandomChosenPixels_ComputedValues) {
		BOOST_TEST_MESSAGE("Running ComputeMassCenterAndGlyphSum_2RandomChosenPixels_ComputedValues");
		rows = cols = getSz(); top = getSz() - 1U;
		pixels.assign(rows*cols, 0U);
		// random 2 points: p1(x1, y1) and p2(x2, y2)
		const unsigned char p1 = ut::randUnsignedChar(1U), // random value 1..255
			p2 = ut::randUnsignedChar(1U), // random value 1..255
			x1 = ut::randUnsignedChar(0U, getSz()-1U), // random value 0..sz-1
			x2 = ut::randUnsignedChar(0U, getSz()-1U), // random value 0..sz-1
			y1 = ut::randUnsignedChar(0U, getSz()-1U), // random value 0..sz-1
			y2 = ut::randUnsignedChar(0U, getSz()-1U); // random value 0..sz-1
		pixels[x1+y1*cols] = p1; pixels[x2+y2*cols] = p2;
		cout<<"Checking mass-center for 2 pixels: "
			<<(unsigned)p1<<'('<<(unsigned)x1<<','<<(unsigned)y1<<"); "
			<<(unsigned)p2<<'('<<(unsigned)x2<<','<<(unsigned)y2<<')'<<endl;
		
		PixMapSym::computeMcAndAvgPixVal(getSz(), getMaxGlyphSum(), pixels, rows, cols, left, top,
										 getConsec(), getRevConsec(), mc, apv);
		BOOST_TEST(apv == ((double)p1+p2) / getMaxGlyphSum(), test_tools::tolerance(1e-4)); // glyphSum = (p1+p2)/255
		// mc is measured based on a DESCENDING vertical axis
		// ( (x1*p1+x2*p2)/(p1+p2)  ,  sz-1-(y1*p1+y2*p2)/(p1+p2) ) all downscaled by (sz-1)
		BOOST_TEST(mc.x == ((double)x1*p1+x2*p2) / (((double)p1+p2) * (getSz() - 1U)), test_tools::tolerance(1e-4));
		BOOST_TEST(mc.y == 1. - (((double)y1*p1+y2*p2) / (((double)p1+p2) * (getSz() - 1U))), test_tools::tolerance(1e-4));
	}
BOOST_AUTO_TEST_SUITE_END() // FontEngine_Tests_Computations

// Test suite checking FontEngine constraints
BOOST_FIXTURE_TEST_SUITE(FontEngine_Tests_Config, ut::FontEngineFixtConfig)
	BOOST_AUTO_TEST_CASE(IncompleteFontConfig_NoFontFile_logicErrorsForFontOperations) {
		BOOST_TEST_MESSAGE("Running IncompleteFontConfig_NoFontFile_logicErrorsForFontOperations");
		if(nullptr == pfe)
			return;

		FontEngine &fe = *pfe;
		string name;
		BOOST_CHECK_THROW(fe.setFontSz(10U), logic_error);
		BOOST_CHECK_THROW(fe.setEncoding("UNICODE"), logic_error);
		BOOST_CHECK_THROW(fe.setNthUniqueEncoding(0U), logic_error);
		BOOST_CHECK_THROW(fe.symsSet(), logic_error);
		BOOST_CHECK_THROW(fe.smallGlyphsCoverage(), logic_error);
		BOOST_CHECK_THROW(fe.uniqueEncodings(), logic_error);
		BOOST_CHECK_THROW(fe.getEncoding(), logic_error);
		BOOST_CHECK_THROW(fe.getFamily(), logic_error);
		BOOST_CHECK_THROW(fe.getStyle(), logic_error);
			
		BOOST_REQUIRE_NO_THROW(name = fe.fontFileName());
		BOOST_REQUIRE(name.empty());
	}

	BOOST_AUTO_TEST_CASE(FontConfig_IncorrectFontFile_CannotSetFont) {
		BOOST_TEST_MESSAGE("Running FontConfig_IncorrectFontFile_CannotSetFont");
		if(nullptr == pfe)
			return;

		FontEngine &fe = *pfe;
		bool correct;
		BOOST_CHECK_NO_THROW(correct = pfe->newFont("")); // bad font file name
		BOOST_REQUIRE(!correct);
	}

	BOOST_AUTO_TEST_CASE(FontConfig_NonScalableFont_CannotSetFont) {
		BOOST_TEST_MESSAGE("Running FontConfig_NonScalableFont_CannotSetFont");
		if(nullptr == pfe)
			return;

		bool correct;
		BOOST_CHECK_NO_THROW(correct = pfe->newFont("res\\vga855.fon"));
		BOOST_REQUIRE(!correct);
	}

	BOOST_AUTO_TEST_CASE(FontConfig_CorrectFontFile_SettingFontOk) {
		BOOST_TEST_MESSAGE("Running FontConfig_CorrectFontFile_SettingFontOk");
		if(nullptr == pfe)
			return;

		bool correct;
		BOOST_REQUIRE_NO_THROW(correct = pfe->newFont("res\\BPmonoBold.ttf"));
		BOOST_REQUIRE(correct);
	}

	BOOST_AUTO_TEST_CASE(FontConfig_NoFontSize_CannotGetSymsSetNorSmallGlyphCoverage) {
		BOOST_TEST_MESSAGE("Running FontConfig_NoFontSize_CannotGetSymsSetNorSmallGlyphCoverage");
		if(nullptr == pfe)
			return;

		FontEngine &fe = *pfe;
		unsigned unEncs = 0U, encIdx = 0U;
		string enc, fname;
		FT_String *fam = nullptr, *style = nullptr;
		fe.newFont("res\\BPmonoBold.ttf");

		// No font size throws
		BOOST_CHECK_THROW(fe.symsSet(), logic_error);
		BOOST_CHECK_THROW(fe.smallGlyphsCoverage(), logic_error);

		// Everything else ok
		BOOST_REQUIRE_NO_THROW(fname = fe.fontFileName());
		BOOST_CHECK(!fname.empty());

		BOOST_REQUIRE_NO_THROW(unEncs = fe.uniqueEncodings());
		BOOST_REQUIRE(unEncs == 2U);

		BOOST_REQUIRE_NO_THROW(enc = fe.getEncoding(&encIdx));
		BOOST_CHECK(enc.compare("UNICODE") == 0 && encIdx == 0);

		BOOST_REQUIRE_NO_THROW(fam = fe.getFamily());
		BOOST_CHECK(strcmp(fam, "BPmono") == 0);

		BOOST_REQUIRE_NO_THROW(style = fe.getStyle());
		BOOST_CHECK(strcmp(style, "Bold") == 0);

		// Setting Encodings
		BOOST_REQUIRE_NO_THROW(enc = fe.setEncoding(enc)); // same encoding
		BOOST_CHECK(!enc.empty());

		BOOST_REQUIRE_NO_THROW(enc = fe.setEncoding("APPLE_ROMAN")); // new encoding
		BOOST_CHECK(!enc.empty());

		BOOST_CHECK(fe.getEncoding(&encIdx).compare("APPLE_ROMAN") == 0);
		BOOST_CHECK(encIdx == 1U);

		// Recheck that no font size throws for the new encoding
		BOOST_CHECK_THROW(fe.symsSet(), logic_error);
		BOOST_CHECK_THROW(fe.smallGlyphsCoverage(), logic_error);
	}

	BOOST_AUTO_TEST_CASE(FontConfig_CompleteConfig_NoProblemsExpected) {
		BOOST_TEST_MESSAGE("Running FontConfig_CompleteConfig_NoProblemsExpected");
		if(nullptr == pfe)
			return;

		FontEngine &fe = *pfe;
		fe.newFont("res\\BPmonoBold.ttf");

		extern const unsigned Settings_MIN_FONT_SIZE;
		extern const unsigned Settings_MAX_FONT_SIZE;
		BOOST_REQUIRE_THROW(fe.setFontSz(Settings_MIN_FONT_SIZE-1U), invalid_argument);
		BOOST_REQUIRE_NO_THROW(fe.setFontSz(Settings_MIN_FONT_SIZE)); // ok
		BOOST_REQUIRE_THROW(fe.setFontSz(Settings_MAX_FONT_SIZE+1U), invalid_argument);
		BOOST_REQUIRE_NO_THROW(fe.setFontSz(Settings_MAX_FONT_SIZE)); // ok

		BOOST_REQUIRE_NO_THROW(fe.setFontSz(10U)); // ok
		BOOST_CHECK_NO_THROW(fe.symsSet());
		BOOST_TEST(fe.smallGlyphsCoverage() == 0.1201569,
				   test_tools::tolerance(1e-4));

		BOOST_REQUIRE(fe.setEncoding("APPLE_ROMAN")); // APPLE_ROMAN
		BOOST_REQUIRE_NO_THROW(fe.setFontSz(15U));
		BOOST_CHECK_NO_THROW(fe.symsSet());
		BOOST_TEST(fe.smallGlyphsCoverage() == 0.109403,
				   test_tools::tolerance(1e-4));
	}
BOOST_AUTO_TEST_SUITE_END() // FontEngine_Tests_Config