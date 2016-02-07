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

Config::Config(unsigned fontSz_/* = 0U*/,
			   double kSdevFg_/* = 0.*/, double kSdevEdge_/* = 0.*/, double kSdevBg_/* = 0.*/,
			   double kContrast_/* = 0.*/, double kMCsOffset_/* = 0.*/, double kCosAngleMCs_/* = 0.*/,
			   double kGlyphWeight_/* = 0.*/, unsigned threshold4Blank_/* = 0U*/,
			   unsigned hMaxSyms_/* = 0U*/, unsigned vMaxSyms_/* = 0U*/) :
	   fontSz(fontSz_),
	   kSdevFg(kSdevFg_), kSdevEdge(kSdevEdge_), kSdevBg(kSdevBg_), kContrast(kContrast_),
	   kMCsOffset(kMCsOffset_), kCosAngleMCs(kCosAngleMCs_), kGlyphWeight(kGlyphWeight_),
	   threshold4Blank(threshold4Blank_), hMaxSyms(hMaxSyms_), vMaxSyms(vMaxSyms_) {}

#define GET_FIELD(FieldType, ...) \
	static std::shared_ptr<FieldType> pField; \
	if(ut::InitController::init##FieldType || !pField) { \
		pField = std::make_shared<FieldType>(__VA_ARGS__); \
		ut::InitController::init##FieldType = false; \
			} \
	return *pField;

Img& Controller::getImg() {
	GET_FIELD(Img, nullptr); // Here's useful the hack mentioned at Img's constructor declaration
}

FontEngine& Controller::getFontEngine() const {
	GET_FIELD(FontEngine, *this);
}

MatchEngine& Controller::getMatchEngine(const Config &cfg_) const {
	GET_FIELD(MatchEngine, cfg_, getFontEngine());
}

Transformer& Controller::getTransformer(const Config &cfg_) const {
	GET_FIELD(Transformer, *this, cfg_, getMatchEngine(cfg_), getImg());
}

Comparator& Controller::getComparator() const {
	GET_FIELD(Comparator, *this);
}

ControlPanel& Controller::getControlPanel(Config &cfg_) {
	GET_FIELD(ControlPanel, *this, cfg_);
}

#undef GET_FIELD

void Controller::reportGlyphProgress(double progress) const {}

void Controller::reportTransformationProgress(double progress) const {}
