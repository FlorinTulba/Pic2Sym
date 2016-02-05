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

#define UNIT_TESTING

namespace ut {
	extern bool initImg, initFe, initMe, initTr, initComp, initCp;

	struct Fixt {
		Fixt() { initImg = initFe = initMe = initTr = initComp = initCp = true; }
	};
}

#endif