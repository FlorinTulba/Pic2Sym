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

using namespace std;
using namespace boost;
using namespace ut;

BOOST_FIXTURE_TEST_SUITE(FontEngine_Tests, Fixt)
	BOOST_AUTO_TEST_CASE(IncompleteFontConfig_NoFontFile) {
		BOOST_TEST_MESSAGE("Running IncompleteFontConfig_NoFontFile ...");
		try {
			Config cfg;
			Controller c(cfg);
			string name;
			FontEngine fe(c); // might throw runtime_error

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

