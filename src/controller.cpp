/**********************************************************
 Project:     Pic2Sym
 File:        controller.cpp

 Author:      Florin Tulba
 Created on:  2016-1-22
 
 Copyright (c) 2016 Florin Tulba
 **********************************************************/

#include "controller.h"
#include "misc.h"
#include "dlgs.h"

#include <Windows.h>
#include <sstream>

#include <boost/filesystem/operations.hpp>

#ifndef UNIT_TESTING
#	include <opencv2/highgui.hpp>
#endif

using namespace std;
using namespace std::chrono;
using namespace boost::filesystem;
using namespace boost::archive;
using namespace cv;

Settings::Settings(const MatchSettings &&ms_) :
	ss(DEF_FONT_SIZE), is(MAX_H_SYMS, MAX_V_SYMS), ms(ms_) {}

ostream& operator<<(ostream &os, const Settings &s) {
	os<<s.ss<<s.is<<s.ms<<endl;
	return os;
}

Controller::Controller(Settings &s) :
		img(getImg()), fe(getFontEngine(s.ss)), cfg(s),
		me(getMatchEngine(s)), t(getTransformer(s)),
		comp(getComparator()), cp(getControlPanel(s)),
		hMaxSymsOk(Settings::isHmaxSymsOk(s.is.getMaxHSyms())),
		vMaxSymsOk(Settings::isVmaxSymsOk(s.is.getMaxVSyms())),
		fontSzOk(Settings::isFontSizeOk(s.ss.getFontSz())) {
	comp.setPos(0, 0);
	comp.permitResize(false);
	comp.setTitle("Pic2Sym - (c) 2016 Florin Tulba");
	comp.setStatus("Press Ctrl+P for Control Panel; ESC to Exit");
}

bool Controller::validState(bool imageRequired/* = true*/) const {
	if(((imageOk && hMaxSymsOk && vMaxSymsOk) || !imageRequired) &&
	   fontFamilyOk && fontSzOk)
		return true;

	ostringstream oss;
	oss<<"The problems are:"<<endl<<endl;
	if(imageRequired && !imageOk)
		oss<<"- no image to transform"<<endl;
	if(imageRequired && !hMaxSymsOk)
		oss<<"- max count of symbols horizontally is too small"<<endl;
	if(imageRequired && !vMaxSymsOk)
		oss<<"- max count of symbols vertically is too small"<<endl;
	if(!fontFamilyOk)
		oss<<"- no font family to use during transformation"<<endl;
	if(!fontSzOk)
		oss<<"- selected font size is too small"<<endl;
	errMsg(oss.str(), "Please Correct these errors first!");
	return false;
}

void Controller::newImage(const string &imgPath) {
	if(img.absPath().compare(absolute(imgPath)) == 0)
		return; // same image

	if(!img.reset(imgPath)) {
		ostringstream oss;
		oss<<"Invalid image file: '"<<imgPath<<'\'';
		errMsg(oss.str());
		return;
	}

	// Comparator window is to be placed within 1024x768 top-left part of the screen
	static const double
		HEIGHT_TITLE_TOOLBAR_SLIDER_STATUS = 70,
		WIDTH_LATERAL_BORDERS = 4,
		H_NUMERATOR = 768-HEIGHT_TITLE_TOOLBAR_SLIDER_STATUS, // desired max height - HEIGHT_TITLE_TOOLBAR_SLIDER_STATUS
		W_NUMERATOR = 1024-WIDTH_LATERAL_BORDERS; // desired max width - WIDTH_LATERAL_BORDERS

	ostringstream oss;
	oss<<"Pic2Sym on image: "<<img.absPath();
	comp.setTitle(oss.str());

	if(!imageOk) { // 1st image loaded
		comp.permitResize();
		imageOk = true;
	}

	const Mat &orig = img.original();
	comp.setReference(orig); // displays the image

	// Resize window to preserve the aspect ratio of the loaded image,
	// while not enlarging it, nor exceeding 1024 x 768
	const int height = orig.rows, width = orig.cols;
	const double k = min(1., min(H_NUMERATOR/height, W_NUMERATOR/width));
	const int winHeight = (int)round(k*height+HEIGHT_TITLE_TOOLBAR_SLIDER_STATUS),
			winWidth = (int)round(k*width+WIDTH_LATERAL_BORDERS);
	
	comp.resize(winWidth, winHeight);
}

void Controller::updateCmapStatusBar() const {
	ostringstream oss;
	oss<<"Font type '"<<fe.getFamily()<<' '<<fe.getStyle()
		<<"', size "<<cfg.ss.getFontSz()<<", encoding '"<<fe.getEncoding()<<'\''
		<<" : "<<me.getSymsCount()<<" symbols";
	pCmi->setStatus(oss.str());
}

void Controller::symbolsChanged() {
	fe.setFontSz(cfg.ss.getFontSz());
	me.updateSymbols();

	updateCmapStatusBar();
	pCmi->updatePagesCount((unsigned)fe.symsSet().size());
	pCmi->showPage(0U);
}

bool Controller::_newFontFamily(const string &fontFile, bool forceUpdate/* = false*/) {
	if(fe.fontFileName().compare(fontFile) == 0 && !forceUpdate)
		return false; // same font

	if(!fe.newFont(fontFile)) {
		ostringstream oss;
		oss<<"Invalid font file: '"<<fontFile<<'\'';
		errMsg(oss.str());
		return false;
	}

	cp.updateEncodingsCount(fe.uniqueEncodings());

	if(!fontFamilyOk) {
		fontFamilyOk = true;
		pCmi = std::make_shared<CmapInspect>(*this);
		pCmi->setPos(424, 0);		// Place cmap window on x axis between 424..1064
		pCmi->permitResize(false);	// Ensure the user sees the symbols exactly their size
	}

	return true;
}

void Controller::newFontFamily(const string &fontFile) {
	if(!_newFontFamily(fontFile))
		return;

	symbolsChanged();
}

void Controller::selectedFontFile(const string &fName) const {
	cfg.ss.setFontFile(fName);
}

void Controller::newFontEncoding(int encodingIdx) {
	// Ignore call if no font yet, or just 1 encoding,
	// or if the required hack (mentioned in 'ui.h') provoked this call
	if(!fontFamilyOk || fe.uniqueEncodings() == 1U || cp.encMaxHack())
		return;
	
	unsigned currEncIdx;
	fe.getEncoding(&currEncIdx);
	if(currEncIdx == (unsigned)encodingIdx)
		return;

	fe.setNthUniqueEncoding(encodingIdx);

	symbolsChanged();
}

bool Controller::_newFontEncoding(const string &encName, bool forceUpdate/* = false*/) {
	return fe.setEncoding(encName, forceUpdate);
}

bool Controller::newFontEncoding(const string &encName) {
	bool result = _newFontEncoding(encName);
	if(result)
		symbolsChanged();

	return result;
}

void Controller::selectedEncoding(const string &encName) const {
	cfg.ss.setEncoding(encName);
}

bool Controller::_newFontSize(int fontSz, bool forceUpdate/* = false*/) {
	if(!Settings::isFontSizeOk(fontSz)) {
		fontSzOk = false;
		ostringstream oss;
		oss<<"Invalid font size: "<<fontSz<<". Please set at least "<<Settings::MIN_FONT_SIZE<<'.';
		errMsg(oss.str());
		return false;
	}

	if(!fontSzOk)
		fontSzOk = true;

	if(!fontFamilyOk || ((unsigned)fontSz == cfg.ss.getFontSz() && !forceUpdate))
		return false;

	cfg.ss.setFontSz(fontSz);
	pCmi->updateGrid();

	return true;
}

void Controller::newFontSize(int fontSz) {
	if(!_newFontSize(fontSz))
		return;

	symbolsChanged();
}

void Controller::newHmaxSyms(int maxSymbols) {
	if(!Settings::isHmaxSymsOk(maxSymbols)) {
		hMaxSymsOk = false;
		ostringstream oss;
		oss<<"Invalid max number of horizontal symbols: "<<maxSymbols<<". Please set at least "<<Settings::MIN_H_SYMS<<'.';
		errMsg(oss.str());
		return;
	}

	if(!hMaxSymsOk)
		hMaxSymsOk = true;

	if((unsigned)maxSymbols == cfg.is.getMaxHSyms())
		return;

	cfg.is.setMaxHSyms(maxSymbols);
}

void Controller::newVmaxSyms(int maxSymbols) {
	if(!Settings::isVmaxSymsOk(maxSymbols)) {
		vMaxSymsOk = false;
		ostringstream oss;
		oss<<"Invalid max number of vertical symbols: "<<maxSymbols<<". Please set at least "<<Settings::MIN_V_SYMS<<'.';
		errMsg(oss.str());
		return;
	}

	if(!vMaxSymsOk)
		vMaxSymsOk = true;

	if((unsigned)maxSymbols == cfg.is.getMaxVSyms())
		return;

	cfg.is.setMaxVSyms(maxSymbols);
}

void Controller::newThreshold4BlanksFactor(unsigned threshold) {
	if((unsigned)threshold != cfg.ms.getBlankThreshold())
		cfg.ms.setBlankThreshold(threshold);
}

void Controller::newContrastFactor(double k) {
	if(k != cfg.ms.get_kContrast())
		cfg.ms.set_kContrast(k);
}

void Controller::newStructuralSimilarityFactor(double k) {
	if(k != cfg.ms.get_kSsim())
		cfg.ms.set_kSsim(k);
}

void Controller::newUnderGlyphCorrectnessFactor(double k) {
	if(k != cfg.ms.get_kSdevFg())
		cfg.ms.set_kSdevFg(k);
}

void Controller::newAsideGlyphCorrectnessFactor(double k) {
	if(k != cfg.ms.get_kSdevBg())
		cfg.ms.set_kSdevBg(k);
}

void Controller::newGlyphEdgeCorrectnessFactor(double k) {
	if(k != cfg.ms.get_kSdevEdge())
		cfg.ms.set_kSdevEdge(k);
}

void Controller::newDirectionalSmoothnessFactor(double k) {
	if(k != cfg.ms.get_kCosAngleMCs())
		cfg.ms.set_kCosAngleMCs(k);
}

void Controller::newGravitationalSmoothnessFactor(double k) {
	if(k != cfg.ms.get_kMCsOffset())
		cfg.ms.set_kMCsOffset(k);
}

void Controller::newGlyphWeightFactor(double k) {
	if(k != cfg.ms.get_kSymDensity())
		cfg.ms.set_kSymDensity(k);
}

MatchEngine::VSymDataCItPair Controller::getFontFaces(unsigned from, unsigned maxCount) const {
	return me.getSymsRange(from, maxCount);
}

bool Controller::performTransformation() {
	if(!validState())
		return false;

	t.run();
	return true;
}

void Controller::symsSetUpdate(bool done/* = false*/, double elapsed/* = 0.*/) const {
	if(done) {
		reportGlyphProgress(1.);

		ostringstream oss;
		oss<<"The update of the symbols set took "<<elapsed<<" s!";

		cout<<oss.str()<<endl<<endl;
		if(pCmi)
			pCmi->setOverlay(oss.str(), 3000);

	} else {
		reportGlyphProgress(0.);
	}
}

void Controller::imgTransform(bool done/* = false*/, double elapsed/* = 0.*/) const {
	if(done) {
		reportTransformationProgress(1.);

		ostringstream oss;
		oss<<"The transformation took "<<elapsed<<" s!";

		cout<<oss.str()<<endl<<endl;
		comp.setOverlay(oss.str(), 3000);

	} else {
		reportTransformationProgress(0.);
	}
}

void Controller::restoreUserDefaultMatchSettings() {
	cfg.ms.loadUserDefaults();
	cp.updateMatchSettings(cfg.ms);
}

void Controller::setUserDefaultMatchSettings() const {
	cfg.ms.saveUserDefaults();
}

void Controller::loadSettings() {
	static SettingsSelector ss; // loader
	if(!ss.promptForUserChoice())
		return;
	
	const SymSettings prevSymSettings(cfg.ss); // keep a copy of old SymSettings
	cout<<"Loading settings from '"<<ss.selection()<<'\''<<endl;
	try {
		ifstream ifs(ss.selection(), ios::binary);
		binary_iarchive ia(ifs);
		ia>>cfg;
	} catch(...) {
		cerr<<"Couldn't load these settings"<<endl;
		return;
	}

	cp.updateMatchSettings(cfg.ms);
	cp.updateImgSettings(cfg.is);

	if(prevSymSettings==cfg.ss)
		return;

	bool fontFileChanged = false, encodingChanged = false;
	const auto newEncName = cfg.ss.getEncoding();
	if(prevSymSettings.getFontFile().compare(cfg.ss.getFontFile()) != 0) {
		_newFontFamily(cfg.ss.getFontFile(), true);
		fontFileChanged = true;
	}
	if((!fontFileChanged && prevSymSettings.getEncoding().compare(newEncName) != 0) ||
			(fontFileChanged && cfg.ss.getEncoding().compare(newEncName) != 0)) {
		_newFontEncoding(newEncName, true);
		encodingChanged = true;
	}

	if(prevSymSettings.getFontSz() != cfg.ss.getFontSz()) {
		if(fontFileChanged || encodingChanged) {
			pCmi->updateGrid();
		} else {
			_newFontSize(cfg.ss.getFontSz(), true);
		}
	}

	unsigned currEncIdx;
	fe.getEncoding(&currEncIdx);
	cp.updateSymSettings(currEncIdx, cfg.ss.getFontSz());
	
	symbolsChanged();
}

void Controller::saveSettings() const {
	if(!cfg.ss.ready()) {
		warnMsg("There's no Font yet.\nSave settings only after selecting a font !");
		return;
	}

	static SettingsSelector ss(false); // saver
	if(!ss.promptForUserChoice())
		return;

	cout<<"Saving settings to '"<<ss.selection()<<'\''<<endl;
	try {
		ofstream ofs(ss.selection(), ios::binary);
		binary_oarchive oa(ofs);
		oa<<cfg;
	} catch(...) {
		cerr<<"Couldn't save current settings"<<endl;
		return;
	}
}

Controller::Timer::Timer(const Controller &ctrler_, ComputationType compType_) :
		ctrler(ctrler_), compType(compType_), start(high_resolution_clock::now()) {
	switch(compType) {
		case ComputationType::SYM_SET_UPDATE:
			ctrler.symsSetUpdate();
			break;
		case ComputationType::IMG_TRANSFORM:
			ctrler.imgTransform();
			break;
		default:;
	}
}

Controller::Timer::~Timer() {
	if(!active)
		return;

	duration<double> elapsedS = high_resolution_clock::now() - start;
	switch(compType) {
		case ComputationType::SYM_SET_UPDATE:
			ctrler.symsSetUpdate(true, elapsedS.count());
			break;
		case ComputationType::IMG_TRANSFORM:
			ctrler.imgTransform(true, elapsedS.count());
			break;
		default:;
	}
}

void Controller::Timer::release() {
	active = false;
}

// Methods from below have different definitions for UnitTesting project
#ifndef UNIT_TESTING

Controller::~Controller() {
	destroyAllWindows();
}

void Controller::handleRequests() {
	for(;;) {
		// When pressing ESC, prompt the user if he wants to exit
		if(27 == waitKey() &&
		   IDYES == MessageBox(nullptr, L"Do you want to leave the application?", L"Question",
		   MB_ICONQUESTION | MB_YESNOCANCEL | MB_TASKMODAL | MB_SETFOREGROUND))
		   break;
	}
}

void Controller::hourGlass(double progress, const string &title/* = ""*/) const {
	static const String waitWin = "Please Wait!";
	if(progress == 0.) {
		namedWindow(waitWin);
#ifndef _DEBUG // destroyWindow in Debug mode triggers deallocation of invalid block
	} else if(progress == 1.) {
		destroyWindow(waitWin);
#endif // in Debug mode, leaving the hourGlass window visible, but with 100% as title
	} else {
		ostringstream oss;
		if(title.empty())
			oss<<waitWin;
		else
			oss<<title;
		oss<<" ("<<fixed<<setprecision(0)<<progress*100.<<"%)";
		setWindowTitle(waitWin, oss.str());
	}
}

void Controller::reportGlyphProgress(double progress) const {
	hourGlass(progress, "Processing glyphs. Please wait");
}

void Controller::reportTransformationProgress(double progress) const {
	hourGlass(progress, "Transforming image. Please wait");
	if(0. == progress) {
		comp.setReference(img.getResized()); // display 'original' when starting transformation
	} else if(1. == progress) {
		comp.setResult(t.getResult()); // display the result at the end of the transformation
	}
}

#define GET_FIELD(FieldType, ...) \
	static FieldType field(__VA_ARGS__); \
	return field;

Img& Controller::getImg() {
	GET_FIELD(Img, nullptr); // Here's useful the hack mentioned at Img's constructor declaration
}

Comparator& Controller::getComparator() {
	GET_FIELD(Comparator, nullptr); // Here's useful the hack mentioned at Comparator's constructor declaration
}

FontEngine& Controller::getFontEngine(const SymSettings &ss_) const {
	GET_FIELD(FontEngine, *this, ss_);
}

MatchEngine& Controller::getMatchEngine(const Settings &cfg_) const {
	GET_FIELD(MatchEngine, cfg_, getFontEngine(cfg_.ss));
}

Transformer& Controller::getTransformer(const Settings &cfg_) const {
	GET_FIELD(Transformer, *this, cfg_, getMatchEngine(cfg_), getImg());
}

ControlPanel& Controller::getControlPanel(Settings &cfg_) {
	GET_FIELD(ControlPanel, *this, cfg_);
}

#undef GET_FIELD

#endif // UNIT_TESTING not defined
