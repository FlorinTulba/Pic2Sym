/**********************************************************
 Project:     UnitTesting
 File:        testMain.h

 Author:      Florin Tulba
 Created on:  2016-1-17
 
 Copyright (c) 2016 Florin Tulba
 **********************************************************/

#ifndef H_TEST_MAIN
#define H_TEST_MAIN

#include <boost/test/unit_test.hpp>

namespace ut { // unit testing namespace

	// Used for a global fixture to reinitialize Controller's fields for each test
	struct Controller {

		/*
		Which Controller's fields to reinitialize.
		The global fixture sets them to true.
		After initialization each is set to false.
		*/
		static bool initImg, initFontEngine, initMatchEngine,
			initTransformer, initComparator, initControlPanel;
	};

	// Used for a global fixture to reinitialize MatchEngine's availAspects in getReady()
	struct MatchEngine {
		static bool initAvailAspects;
	};

	// Fixture to be used before every test
	struct Fixt {
		Fixt();		// set up
		~Fixt();	// tear down
	};
}

#endif