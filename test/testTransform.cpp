/**********************************************************
 Project:     UnitTesting
 File:        testTransform.cpp

 Author:      Florin Tulba
 Created on:  2016-1-17
 
 Copyright (c) 2016 Florin Tulba
 **********************************************************/

#include "testMain.h"
#include "transform.cpp" // including CPP, to recompile #if(n)def UNIT_TESTING regions

BOOST_AUTO_TEST_SUITE(Transform_Tests)
BOOST_AUTO_TEST_CASE(___) {
	Config cfg(10U, 500U, 300U, 0U, .3, .3, 3., 1., 1., 1.);
	unsigned sz = cfg.getFontSz();
	const double sz2 = (double)sz*sz;
	FT_Face newFace = nullptr;
	FontEngine fe;
	BOOST_REQUIRE(fe.checkFontFile("res\\BPmonoBold.ttf", newFace));
	BOOST_REQUIRE_NO_THROW(fe.setFace(newFace));
	BOOST_REQUIRE(fe.setEncoding("APPLE_ROMAN"));
	BOOST_REQUIRE_NO_THROW(fe.setFontSz(sz));
	Matcher matcher(sz, fe.smallGlyphsCoverage());
	Mat consec(1, sz, CV_64FC1);
	iota(consec.begin<double>(), consec.end<double>(), 0.);
	auto itFeBegin = fe.charset().cbegin();
	vector<pair<Mat, Mat>> charset;
	charset.clear();
	charset.reserve(fe.charset().size());
	for(auto &pmc : fe.charset()) {
		Mat glyph = toMat(pmc, sz), negGlyph = 1. - glyph;
		charset.emplace_back(glyph, negGlyph);
	}

	Mat patch(sz, sz, CV_64FC1);
	BestMatch best(fe.getEncoding().compare("UNICODE") == 0); // holds the best grayscale match
	findBestMatch(cfg, charset, patch, matcher, best, itFeBegin, sz2, consec);
}
BOOST_AUTO_TEST_SUITE_END() // FontEngine_Tests
