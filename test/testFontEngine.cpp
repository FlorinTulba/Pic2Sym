/**********************************************************
 Project:     UnitTesting
 File:        testFontEngine.cpp

 Author:      Florin Tulba
 Created on:  2016-1-17
 
 Copyright (c) 2016 Florin Tulba
 **********************************************************/

#include "testMain.h"
#include "controller.h"
#include "config.h"
#include "fontEngine.h"

#include <iostream>
#include <numeric>
#include <random>

using namespace std;
using namespace cv;
using namespace boost;

BOOST_FIXTURE_TEST_SUITE(FontEngine_Tests, ut::Fixt)
	BOOST_AUTO_TEST_CASE(ComputeMassCenterAndGlyphSum) {
		BOOST_TEST_MESSAGE("Running ComputeMassCenterAndGlyphSum ...");

		// Glyph data has ASCENDING vertical axis,
		// while the mass-center is considered on a DESCENDING vertical axis

		random_device rd;
		mt19937 gen(rd());
		uniform_int_distribution<unsigned> uid;

		unsigned sz = 10U; // patches of 10x10
		Mat consec(1, sz, CV_64FC1), // 0..9
			revConsec;	// 9..0
		iota(consec.begin<double>(), consec.end<double>(), (double)0.); // 0..9
		flip(consec, revConsec, 1);	// 9..0

		// Test EMPTY PATCH => glyphSum = 0, massCenter = (4.5, 4.5)
		vector<unsigned char> pixels; // uses ASCENDING vertical axis
		unsigned char rows = 0U, cols = 5U, left = 0U, top = sz-1U;
		double gs = PixMapSym::computeGlyphSum(rows, cols, pixels);
		BOOST_REQUIRE(gs == 0.);
		Point2d mc = PixMapSym::computeMc(sz, pixels, rows, cols, left, top, gs, consec, revConsec); // measured based on a DESCENDING vertical axis
		BOOST_TEST(mc.x == 4.5, test_tools::tolerance(1e-4));
		BOOST_TEST(mc.y == 4.5, test_tools::tolerance(1e-4));

		// Empty patch, as well
		rows = 4U; cols = 0U;
		gs = PixMapSym::computeGlyphSum(rows, cols, pixels);
		BOOST_REQUIRE(gs == 0.);
		mc = PixMapSym::computeMc(sz, pixels, rows, cols, left, top, gs, consec, revConsec);
		// mc is measured based on a DESCENDING vertical axis
		BOOST_TEST(mc.x == 4.5, test_tools::tolerance(1e-4));
		BOOST_TEST(mc.y == 4.5, test_tools::tolerance(1e-4));

		// Empty patch, although pixels not empty this time
		rows = 5U; cols = 5U; pixels.assign(rows*cols, 0U);
		gs = PixMapSym::computeGlyphSum(rows, cols, pixels);
		BOOST_TEST(gs == 0., test_tools::tolerance(1e-4));
		mc = PixMapSym::computeMc(sz, pixels, rows, cols, left, top, gs, consec, revConsec);
		// mc is measured based on a DESCENDING vertical axis
		BOOST_TEST(mc.x == 4.5, test_tools::tolerance(1e-4));
		BOOST_TEST(mc.y == 4.5, test_tools::tolerance(1e-4));

		// FULL PATCH => glyphSum = 0, massCenter = (4.5, 4.5)
		unsigned char uc = (unsigned char)uid(gen)&0xFFU; // random value 0..255
		cout<<"Checking patch filled with value "<<(unsigned)uc<<endl;
		rows = sz; cols = sz; pixels.assign(rows*cols, uc); // all pixels are 'uc'
		gs = PixMapSym::computeGlyphSum(rows, cols, pixels);
		BOOST_TEST(gs == sz*sz*uc/255., test_tools::tolerance(1e-4));
		mc = PixMapSym::computeMc(sz, pixels, rows, cols, left, top, gs, consec, revConsec);
		// mc is measured based on a DESCENDING vertical axis
		BOOST_TEST(mc.x == 4.5, test_tools::tolerance(1e-4));
		BOOST_TEST(mc.y == 4.5, test_tools::tolerance(1e-4));

		// Single non-zero pixel => glyphSum = pixelValue/255; massCenter = pixelPosition
		rows = 1U; cols = 1U; pixels.assign(rows*cols, uc); // the pixel has value 'uc'
		left = (unsigned char)uid(gen)%sz; // random value 0..sz-1
		top = (unsigned char)uid(gen)%sz; // random value 0..sz-1
		cout<<"Checking patch with a single non-zero pixel at: top="
			<<(unsigned)top<<", left="<<(unsigned)left<<endl;
		gs = PixMapSym::computeGlyphSum(rows, cols, pixels);
		BOOST_TEST(gs == uc/255., test_tools::tolerance(1e-4));
		mc = PixMapSym::computeMc(sz, pixels, rows, cols, left, top, gs, consec, revConsec);
		// mc is measured based on a DESCENDING vertical axis
		BOOST_TEST(mc.x == left, test_tools::tolerance(1e-4));
		BOOST_TEST(mc.y == (sz - 1U - top), test_tools::tolerance(1e-4));

		// 3x3 subarea of pixels='uc' => glyphSum = 9*uc/255; massCenter = subArea's center
		rows = 3U; cols = 3U; pixels.assign(rows*cols, uc); // all the pixel have the value 'uc'
		left = (unsigned char)uid(gen)%(sz-2U); // random value 0..sz-3
		top = (unsigned char)(2U + uid(gen)%(sz-2U)); // random value 2..sz-1
		cout<<"Checking patch with a 3x3 uniform non-zero subarea at: top="
			<<(unsigned)top<<", left="<<(unsigned)left<<endl;
		gs = PixMapSym::computeGlyphSum(rows, cols, pixels);
		BOOST_TEST(gs == rows*cols*uc/255., test_tools::tolerance(1e-4));
		mc = PixMapSym::computeMc(sz, pixels, rows, cols, left, top, gs, consec, revConsec);
		// mc is measured based on a DESCENDING vertical axis
		BOOST_TEST(mc.x == left+1U, test_tools::tolerance(1e-4));
		BOOST_TEST(mc.y == (sz - 1U - (top - 1U)), test_tools::tolerance(1e-4));

		// 2 fixed points at a distance of 6: 170(2, 0) and 85(8, 0)
		//		glyphSum = 255/255 = 1
		//		massCenter = (4, sz-1),
		// 4 is at one third the distance 2..8
		// 170 = 85*2, and 170 + 85 = 255
		rows = 1; cols = 7; left = 2U; top = 0U; pixels.assign(rows*cols, 0U);
		pixels[0] = 170; pixels[cols-1] = 85;
		gs = PixMapSym::computeGlyphSum(rows, cols, pixels);
		BOOST_TEST(gs == 1., test_tools::tolerance(1e-4));
		mc = PixMapSym::computeMc(sz, pixels, rows, cols, left, top, gs, consec, revConsec);
		// mc is measured based on a DESCENDING vertical axis
		BOOST_TEST(mc.x == 4., test_tools::tolerance(1e-4));
		BOOST_TEST(mc.y == sz-1U, test_tools::tolerance(1e-4));

		// random 2 points: p1(x1, y1) and p2(x2, y2) => glyphSum = (p1+p2)/255 and
		// massCenter = ( (x1*p1+x2*p2)/(p1+p2)  ,  sz-1-(y1*p1+y2*p2)/(p1+p2) )
		rows = sz; cols = sz; left = 0U; top = sz-1U; pixels.assign(rows*cols, 0U);
		unsigned char p1 = 1U+(unsigned char)uid(gen)%255U, // random value 1..255
					p2 = 1U+(unsigned char)uid(gen)%255U, // random value 1..255
			x1 = (unsigned char)uid(gen)%sz, // random value 0..sz-1
			x2 = (unsigned char)uid(gen)%sz, // random value 0..sz-1
			y1 = (unsigned char)uid(gen)%sz, // random value 0..sz-1
			y2 = (unsigned char)uid(gen)%sz; // random value 0..sz-1
		pixels[x1+y1*cols] = p1; pixels[x2+y2*cols] = p2;
		cout<<"Checking mass-center for 2 pixels: "
			<<(unsigned)p1<<'('<<(unsigned)x1<<','<<(unsigned)y1<<"); "
			<<(unsigned)p2<<'('<<(unsigned)x2<<','<<(unsigned)y2<<')'<<endl;
		gs = PixMapSym::computeGlyphSum(rows, cols, pixels);
		BOOST_TEST(gs == ((double)p1+p2)/255., test_tools::tolerance(1e-4));
		mc = PixMapSym::computeMc(sz, pixels, rows, cols, left, top, gs, consec, revConsec);
		// mc is measured based on a DESCENDING vertical axis
		BOOST_TEST(mc.x == ((double)x1*p1+x2*p2)/((double)p1+p2), test_tools::tolerance(1e-4));
		BOOST_TEST(mc.y == sz-1U-((double)y1*p1+y2*p2)/((double)p1+p2), test_tools::tolerance(1e-4));
	}

	BOOST_AUTO_TEST_CASE(IncompleteFontConfig_NoFontFile) {
		BOOST_TEST_MESSAGE("Running IncompleteFontConfig_NoFontFile ...");
		try {
			Config cfg;
			Controller c(cfg);
			string name;

			FontEngine &fe = c.getFontEngine();

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
			BOOST_CHECK(name.empty());
		} catch(runtime_error&) {
			cerr<<"Couldn't create FontEngine"<<endl;
		}
	}

	BOOST_AUTO_TEST_CASE(CorrectFontFile) {
		BOOST_TEST_MESSAGE("Running CorrectFontFile ...");
		try {
			Config cfg;
			Controller c(cfg);
			FontEngine fe(c); // might throw runtime_error

			bool correct = false;
			BOOST_CHECK_NO_THROW(correct = fe.newFont("")); // bad font file name
			BOOST_CHECK(!correct);

			BOOST_REQUIRE_NO_THROW(correct = fe.newFont("res\\vga855.fon")); // non-scalable font
			BOOST_CHECK(!correct);
			
			BOOST_REQUIRE_NO_THROW(correct = fe.newFont("res\\BPmonoBold.ttf")); // CORRECT
			BOOST_CHECK(correct);
		} catch(runtime_error&) {
			cerr<<"Couldn't create FontEngine"<<endl;
		}
	}

	BOOST_AUTO_TEST_CASE(IncompleteFontConfig_NoFontSize) {
		BOOST_TEST_MESSAGE("Running IncompleteFontConfig_NoFontSize ...");
		try {
			Config cfg;
			Controller c(cfg);
			bool correct = false;
			unsigned unEncs = 0U, encIdx = 0U;
			string enc, fname;
			FT_String *fam = nullptr, *style = nullptr;
			FontEngine fe(c); // might throw runtime_error

			BOOST_REQUIRE_NO_THROW(correct = fe.newFont("res\\BPmonoBold.ttf")); // CORRECT
			BOOST_CHECK(correct);

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
			BOOST_REQUIRE_NO_THROW(correct = fe.setEncoding(enc)); // same encoding
			BOOST_CHECK(correct);

			BOOST_REQUIRE_NO_THROW(correct = fe.setEncoding("APPLE_ROMAN")); // new encoding
			BOOST_CHECK(correct);

			BOOST_CHECK(fe.getEncoding(&encIdx).compare("APPLE_ROMAN") == 0);
			BOOST_CHECK(encIdx == 1U);

			// Recheck that no font size throws for the new encoding
			BOOST_CHECK_THROW(fe.symsSet(), logic_error);
			BOOST_CHECK_THROW(fe.smallGlyphsCoverage(), logic_error);
		} catch(runtime_error&) {
			cerr<<"Couldn't create FontEngine"<<endl;
		}
	}

	BOOST_AUTO_TEST_CASE(CompleteFontConfig) {
		BOOST_TEST_MESSAGE("Running CompleteFontConfig ...");
		try {
			Config cfg;
			Controller c(cfg);
			FontEngine fe(c); // might throw runtime_error

			BOOST_REQUIRE_NO_THROW(fe.newFont("res\\BPmonoBold.ttf")); // UNICODE

			BOOST_REQUIRE_THROW(fe.setFontSz(Config::MIN_FONT_SIZE-1U), invalid_argument);
			BOOST_REQUIRE_NO_THROW(fe.setFontSz(Config::MIN_FONT_SIZE)); // ok
			BOOST_REQUIRE_THROW(fe.setFontSz(Config::MAX_FONT_SIZE+1U), invalid_argument);
			BOOST_REQUIRE_NO_THROW(fe.setFontSz(Config::MAX_FONT_SIZE)); // ok

			BOOST_REQUIRE_NO_THROW(fe.setFontSz(10U)); // ok
			BOOST_CHECK_NO_THROW(fe.symsSet());
			BOOST_TEST(fe.smallGlyphsCoverage() == 0.1201569, test_tools::tolerance(1e-4));

			BOOST_REQUIRE(fe.setEncoding("APPLE_ROMAN")); // APPLE_ROMAN
			BOOST_REQUIRE_NO_THROW(fe.setFontSz(15U));
			BOOST_CHECK_NO_THROW(fe.symsSet());
			BOOST_TEST(fe.smallGlyphsCoverage() == 0.109403, test_tools::tolerance(1e-4));
		} catch(runtime_error&) {
			cerr<<"Couldn't create FontEngine"<<endl;
		}
	}
BOOST_AUTO_TEST_SUITE_END() // FontEngine_Tests

