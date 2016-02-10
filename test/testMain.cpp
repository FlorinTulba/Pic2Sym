/**********************************************************
 Project:     UnitTesting
 File:        testMain.cpp

 Author:      Florin Tulba
 Created on:  2016-1-17
 
 Copyright (c) 2016 Florin Tulba
 **********************************************************/

#define BOOST_TEST_MODULE Tests for Pic2Sym project

#include "testMain.h"

// Including the CPP files allows parsing UNIT_TESTING guarded regions.
// Disadvantage: namespace pollution
#include "misc.cpp"
#include "fontEngine.cpp"
#include "match.cpp"
#include "transform.cpp"
#include "controller.cpp"
#include "img.cpp"
#include "config.cpp"

namespace ut {
	bool Controller::initImg = false;
	bool Controller::initFontEngine = false;
	bool Controller::initMatchEngine = false;
	bool Controller::initTransformer = false;
	bool Controller::initComparator = false;
	bool Controller::initControlPanel = false;
	bool MatchEngine::initAvailAspects = false;

	Fixt::Fixt() {
		// reinitialize all these fields
		Controller::initImg = Controller::initFontEngine = Controller::initMatchEngine =
		Controller::initTransformer = Controller::initComparator = Controller::initControlPanel =
		
		MatchEngine::initAvailAspects =
			true;
	}

	Fixt::~Fixt() {
	}
}

Config::Config(unsigned fontSz_/* = MIN_FONT_SIZE*/,
			   double kSdevFg_/* = 0.*/, double kSdevEdge_/* = 0.*/, double kSdevBg_/* = 0.*/,
			   double kContrast_/* = 0.*/, double kMCsOffset_/* = 0.*/, double kCosAngleMCs_/* = 0.*/,
			   double kGlyphWeight_/* = 0.*/, unsigned threshold4Blank_/* = 0U*/,
			   unsigned hMaxSyms_/* = MAX_H_SYMS*/, unsigned vMaxSyms_/* = MAX_V_SYMS*/) :
	   fontSz(fontSz_),
	   kSdevFg(kSdevFg_), kSdevEdge(kSdevEdge_), kSdevBg(kSdevBg_), kContrast(kContrast_),
	   kMCsOffset(kMCsOffset_), kCosAngleMCs(kCosAngleMCs_), kGlyphWeight(kGlyphWeight_),
	   threshold4Blank(threshold4Blank_), hMaxSyms(hMaxSyms_), vMaxSyms(vMaxSyms_) {
	cout<<"Initial config values:"<<endl<<*this<<endl;
}

const vector<MatchAspect*>& MatchEngine::getAvailAspects() {
	static std::shared_ptr<const vector<MatchAspect*>> pAvailAspects;
	if(ut::MatchEngine::initAvailAspects || !pAvailAspects) {
		pAvailAspects = std::make_shared<const vector<MatchAspect*>>(vector<MatchAspect*>{
			&fgMatch, &bgMatch, &edgeMatch, &conMatch, &grMatch, &dirMatch, &lsMatch
		});
		ut::MatchEngine::initAvailAspects = false;
	}
	return *pAvailAspects;
}

#define GET_FIELD(FieldType, ...) \
	static std::shared_ptr<FieldType> pField; \
	if(ut::Controller::init##FieldType || !pField) { \
		pField = std::make_shared<FieldType>(__VA_ARGS__); \
		ut::Controller::init##FieldType = false; \
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

Controller::~Controller() {}

void Controller::handleRequests() const {}

void Controller::hourGlass(double progress, const string &title/* = ""*/) const {}

void Controller::reportGlyphProgress(double progress) const {}

void Controller::reportTransformationProgress(double progress) const {}

bool Controller::newImage(const cv::Mat &imgMat) {
	bool result = img.reset(imgMat);

	if(result) {
		cout<<"Using Matrix instead of a "<<(img.isColor()?"color":"grayscale")<<" image"<<endl;
		if(!imageOk)
			imageOk = true;

		// For valid matrices of size sz x sz, ignore MIN_H_SYMS & MIN_V_SYMS =>
		// Testing an image containing a single patch
		if(imgMat.cols == cfg.getFontSz() && imgMat.rows == cfg.getFontSz()) {
			if(1U != cfg.getMaxHSyms())
				cfg.setMaxHSyms(1U);
			if(1U != cfg.getMaxVSyms())
				cfg.setMaxVSyms(1U);

			if(!hMaxSymsOk)
				hMaxSymsOk = true;
			if(!vMaxSymsOk)
				vMaxSymsOk = true;
		}
	}

	return result;
}

bool Controller::newFontEncoding(const std::string &encName) {
	bool result = fe.setEncoding(encName);
	if(result) {
		symbolsChanged();
	}
	return result;
}

void Transformer::createOutputFolder() {}