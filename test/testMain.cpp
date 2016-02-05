/**********************************************************
 Project:     UnitTesting
 File:        testMain.cpp

 Author:      Florin Tulba
 Created on:  2016-1-17
 
 Copyright (c) 2016 Florin Tulba
 **********************************************************/

#define BOOST_TEST_MODULE Tests for Pic2Sym project

#include "testMain.h"
#include "misc.cpp"
#include "ui.cpp"
#include "fontEngine.cpp"
#include "match.cpp"
#include "transform.cpp"
#include "controller.cpp"

namespace ut {
	bool initImg = false, initFe = false, initMe = false,
		initTr = false, initComp = false, initCp = false;
}