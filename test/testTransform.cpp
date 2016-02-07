/**********************************************************
 Project:     UnitTesting
 File:        testTransform.cpp

 Author:      Florin Tulba
 Created on:  2016-1-17
 
 Copyright (c) 2016 Florin Tulba
 **********************************************************/

#include "testMain.h"
#include <fstream>

#include "transform.h"
#include "match.h"
#include "ui.h"
#include "misc.h"
#include "controller.h"

using namespace ut;

BOOST_FIXTURE_TEST_SUITE(Transform_Tests, Fixt)
	BOOST_AUTO_TEST_CASE(Check_symbols_set) {
		BOOST_TEST_MESSAGE("Running Check_symbols_set ...");
		Config cfg(10U, 1., 1., 1., 0., 0., 0., 0., 0U, 500U, 300U);
		Controller c(cfg);

		BOOST_REQUIRE_NO_THROW(c.newFontFamily("res\\BPmonoBold.ttf"));
		BOOST_REQUIRE_NO_THROW(c.newFontEncoding(1)); // APPLE_ROMAN
	}
BOOST_AUTO_TEST_SUITE_END() // FontEngine_Tests
