/**********************************************************
 Project:     UnitTesting
 File:        testFontEngine.cpp

 Author:      Florin Tulba
 Created on:  2016-1-17
 
 Copyright (c) 2016 Florin Tulba
 **********************************************************/

#include "testMain.h"
#include "fontEngine.cpp" // including CPP, to recompile #if(n)def UNIT_TESTING regions

#include <iostream>

using namespace std;
using namespace boost;

BOOST_AUTO_TEST_SUITE(FontEngine_Tests)
	BOOST_AUTO_TEST_CASE(IncompleteFontConfig_NoFontFile) {
		try {
 			FontEngine fe; // might throw runtime_error
			BOOST_CHECK_THROW(fe.selectEncoding(), logic_error);
			BOOST_CHECK_THROW(fe.setFontSz(10U), logic_error);
			BOOST_CHECK_THROW(fe.getEncoding(), logic_error);
			BOOST_CHECK_THROW(fe.charset(), logic_error);
			BOOST_CHECK_THROW(fe.smallGlyphsCoverage(), logic_error);

			BOOST_CHECK_NO_THROW(fe.fontId());
			BOOST_CHECK_NO_THROW(fe.generateCharmapCharts(""));
		} catch(runtime_error&) {
			cerr<<"Couldn't create FontEngine"<<endl;
		}
	}

	BOOST_AUTO_TEST_CASE(CorrectFontFile) {
		try {
			FontEngine fe; // might throw runtime_error
			FT_Face newFace = nullptr;
			BOOST_CHECK_THROW(fe.setFace(newFace), invalid_argument);

			BOOST_REQUIRE(!fe.checkFontFile("", newFace)); // bad font file name
			BOOST_CHECK_THROW(fe.setFace(newFace), invalid_argument);

			BOOST_REQUIRE(!fe.checkFontFile("res\\vga855.fon", newFace)); // non-scalable font
			BOOST_CHECK(newFace != nullptr);
			
			newFace = nullptr;
			BOOST_REQUIRE(fe.checkFontFile("res\\BPmonoBold.ttf", newFace)); // CORRECT
			BOOST_CHECK_NO_THROW(fe.setFace(newFace));
		} catch(runtime_error&) {
			cerr<<"Couldn't create FontEngine"<<endl;
		}
	}

	BOOST_AUTO_TEST_CASE(IncompleteFontConfig_NoFontSize) {
		try {
			FontEngine fe; // might throw runtime_error
			FT_Face newFace = nullptr;
			BOOST_REQUIRE(fe.checkFontFile("res\\BPmonoBold.ttf", newFace));
			BOOST_REQUIRE_NO_THROW(fe.setFace(newFace));

			BOOST_CHECK_THROW(fe.charset(), logic_error);
			BOOST_CHECK_THROW(fe.smallGlyphsCoverage(), logic_error);
			BOOST_CHECK_NO_THROW(fe.generateCharmapCharts(""));

			BOOST_CHECK(fe.fontId().empty());
			BOOST_CHECK(fe.getEncoding().compare("UNICODE") == 0);

			BOOST_REQUIRE(!fe.setEncoding("UNICODE")); // need to set a different encoding

			BOOST_REQUIRE(fe.setEncoding("APPLE_ROMAN")); // CORRECT

			BOOST_CHECK(fe.fontId().empty());
			BOOST_REQUIRE(fe.getEncoding().compare("APPLE_ROMAN") == 0);

			BOOST_CHECK_THROW(fe.charset(), logic_error);
			BOOST_CHECK_THROW(fe.smallGlyphsCoverage(), logic_error);
			BOOST_CHECK_NO_THROW(fe.generateCharmapCharts(""));
		} catch(runtime_error&) {
			cerr<<"Couldn't create FontEngine"<<endl;
		}
	}


	BOOST_AUTO_TEST_CASE(CompleteFontConfig) {
		try {
			FontEngine fe; // might throw runtime_error
			FT_Face newFace = nullptr;
			BOOST_REQUIRE(fe.checkFontFile("res\\BPmonoBold.ttf", newFace));
			BOOST_REQUIRE_NO_THROW(fe.setFace(newFace)); // UNICODE
			BOOST_REQUIRE_NO_THROW(fe.setFontSz(10U));
			BOOST_CHECK_NO_THROW(fe.charset());
			BOOST_TEST(fe.smallGlyphsCoverage() == 0.1174118, test_tools::tolerance(1e-4));
			BOOST_CHECK(fe.fontId().compare("BPmono_Bold_UNICODE_10") == 0);

			BOOST_REQUIRE(fe.setEncoding("APPLE_ROMAN")); // APPLE_ROMAN
			BOOST_REQUIRE_NO_THROW(fe.setFontSz(15U));
			BOOST_CHECK_NO_THROW(fe.charset());
			BOOST_TEST(fe.smallGlyphsCoverage() == 0.0972026, test_tools::tolerance(1e-4));
			BOOST_CHECK(fe.fontId().compare("BPmono_Bold_APPLE_ROMAN_15") == 0);
		} catch(runtime_error&) {
			cerr<<"Couldn't create FontEngine"<<endl;
		}
	}
BOOST_AUTO_TEST_SUITE_END() // FontEngine_Tests

