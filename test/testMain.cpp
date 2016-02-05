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
	bool InitController::initImg = false;
	bool InitController::initFontEngine = false;
	bool InitController::initMatchEngine = false;
	bool InitController::initTransformer = false;
	bool InitController::initComparator = false;
	bool InitController::initControlPanel = false;

	Fixt::Fixt() {
		// reinitialize all these fields
		InitController::initImg = InitController::initFontEngine = InitController::initMatchEngine =
		InitController::initTransformer = InitController::initComparator = InitController::initControlPanel =
			true;
	}

	Fixt::~Fixt() {
	}
}