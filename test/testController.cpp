/**********************************************************
 Project:     UnitTesting
 File:        testController.cpp

 Author:      Florin Tulba
 Created on:  2016-1-17
 
 Copyright (c) 2016 Florin Tulba
 **********************************************************/

#include "testMain.h"

#include "controller.h"

using namespace cv;

BOOST_FIXTURE_TEST_SUITE(Controller_Tests, ut::Fixt)
	BOOST_AUTO_TEST_CASE(Check_Controller) {
		BOOST_TEST_MESSAGE("Running Check_Controller ...");

		Settings s(std::move(MatchSettings()));
		Controller c(s);

		BOOST_REQUIRE(!c.performTransformation()); // no font, no image

		BOOST_REQUIRE_NO_THROW(c.newFontFamily("res\\BPmonoBold.ttf"));
		BOOST_REQUIRE_NO_THROW(c.newFontEncoding("APPLE_ROMAN"));
		BOOST_REQUIRE_NO_THROW(c.newFontSize(10U));

		BOOST_REQUIRE(!c.performTransformation()); // no image yet

		BOOST_REQUIRE(!c.newImage(Mat())); // wrong image

		Mat testPatch(c.getFontSize(), c.getFontSize(), CV_8UC1, Scalar(127)),
			testColorPatch(c.getFontSize(), c.getFontSize(), CV_8UC3, Scalar::all(127));

		BOOST_REQUIRE(c.newImage(testPatch)); // image ok
		BOOST_REQUIRE(c.performTransformation()); // ok

		BOOST_REQUIRE(c.newImage(testColorPatch)); // image ok
		BOOST_REQUIRE(c.performTransformation()); // ok
	}
BOOST_AUTO_TEST_SUITE_END() // Controller_Tests
