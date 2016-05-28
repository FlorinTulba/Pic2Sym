/****************************************************************************************
 The application Pic2Sym approximates images by a
 grid of colored symbols with colored backgrounds.

 This file belongs to the Pic2Sym project.

 Copyrights from the libraries used by the program:
 - (c) 2015 Boost (www.boost.org)
   License: <http://www.boost.org/LICENSE_1_0.txt>
            or doc/licenses/Boost.lic
 - (c) 2015 The FreeType Project (www.freetype.org)
   License: <http://git.savannah.gnu.org/cgit/freetype/freetype2.git/plain/docs/FTL.TXT>
	        or doc/licenses/FTL.txt
 - (c) 2015 OpenCV (www.opencv.org)
   License: <http://opencv.org/license.html>
            or doc/licenses/OpenCV.lic
 - (c) 1997-2002 OpenMP Architecture Review Board (www.openmp.org)
   (c) Microsoft Corporation (Visual C++ implementation for OpenMP C/C++ Version 2.0 March 2002)
   See: <https://msdn.microsoft.com/en-us/library/8y6825x5(v=vs.140).aspx>
 
 (c) 2016 Florin Tulba <florintulba@yahoo.com>

 This program is free software: you can use its results,
 redistribute it and/or modify it under the terms of the GNU
 Affero General Public License version 3 as published by the
 Free Software Foundation.

 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 GNU Affero General Public License for more details.

 You should have received a copy of the GNU Affero General Public License
 along with this program ('agpl-3.0.txt').
 If not, see <http://www.gnu.org/licenses/agpl-3.0.txt>.
 ****************************************************************************************/

#include "controller.h"
#include "misc.h"
#include "dlgs.h"
#include "settings.h"
#include "controlPanel.h"

#include <Windows.h>
#include <sstream>

#include <boost/filesystem/operations.hpp>

#ifndef UNIT_TESTING
#	include "matchSettingsManip.h"
#	include <opencv2/highgui.hpp>
#endif

using namespace std;
using namespace std::chrono;
using namespace boost::filesystem;
using namespace boost::archive;
using namespace cv;

extern const unsigned
	Settings_MAX_THRESHOLD_FOR_BLANKS,
	Settings_MIN_H_SYMS, Settings_MAX_H_SYMS,
	Settings_MIN_V_SYMS, Settings_MAX_V_SYMS,
	Settings_MIN_FONT_SIZE, Settings_MAX_FONT_SIZE,
	Settings_DEF_FONT_SIZE;

bool Settings::isBlanksThresholdOk(unsigned t) {
	return t < Settings_MAX_THRESHOLD_FOR_BLANKS;
}

bool Settings::isHmaxSymsOk(unsigned syms) {
	return syms>=Settings_MIN_H_SYMS && syms<=Settings_MAX_H_SYMS;
}

bool Settings::isVmaxSymsOk(unsigned syms) {
	return syms>=Settings_MIN_V_SYMS && syms<=Settings_MAX_V_SYMS;
}

bool Settings::isFontSizeOk(unsigned fs) {
	return fs>=Settings_MIN_FONT_SIZE && fs<=Settings_MAX_FONT_SIZE;
}

Settings::Settings(const MatchSettings &ms_) :
	ss(Settings_DEF_FONT_SIZE), is(Settings_MAX_H_SYMS, Settings_MAX_V_SYMS), ms(ms_) {}

Settings::Settings() :
	ss(Settings_DEF_FONT_SIZE), is(Settings_MAX_H_SYMS, Settings_MAX_V_SYMS), ms() {}

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

	ostringstream oss;
	oss<<"Pic2Sym on image: "<<img.absPath();
	comp.setTitle(oss.str());

	if(!imageOk) { // 1st image loaded
		comp.permitResize();
		imageOk = true;
	}

	const Mat &orig = img.original();
	comp.setReference(orig); // displays the image

	comp.resize();
}

void Controller::updateCmapStatusBar() const {
	pCmi->setStatus(textForCmapStatusBar());
}

void Controller::symbolsChanged() {
	fe.setFontSz(cfg.ss.getFontSz());
	me.updateSymbols();

	updateCmapStatusBar();
	pCmi->updatePagesCount((unsigned)fe.symsSet().size());
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
		oss<<"Invalid font size: "<<fontSz<<". Please set at least "<<Settings_MIN_FONT_SIZE<<'.';
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
		oss<<"Invalid max number of horizontal symbols: "<<maxSymbols<<". Please set at least "<<Settings_MIN_H_SYMS<<'.';
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
		oss<<"Invalid max number of vertical symbols: "<<maxSymbols<<". Please set at least "<<Settings_MIN_V_SYMS<<'.';
		errMsg(oss.str());
		return;
	}

	if(!vMaxSymsOk)
		vMaxSymsOk = true;

	if((unsigned)maxSymbols == cfg.is.getMaxVSyms())
		return;

	cfg.is.setMaxVSyms(maxSymbols);
}

void Controller::setResultMode(bool hybrid) {
	if(hybrid != cfg.ms.isHybridResult())
		cfg.ms.setResultMode(hybrid);
}

unsigned Controller::getFontSize() const {
	return cfg.ss.getFontSz();
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

void Controller::updateResizedImg(const ResizedImg &resizedImg_) {
	resizedImg = &resizedImg_;
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

		const string &cmapOverlayText = textForCmapOverlay(elapsed);
		cout<<cmapOverlayText<<endl<<endl;
		if(pCmi)
			pCmi->setOverlay(cmapOverlayText, 3000);

	} else {
		reportGlyphProgress(0.);
	}
}

Timer Controller::createTimerForGlyphs() const {
	return std::move(Timer(std::make_shared<Controller::TimerActions_SymSetUpdate>(*this)));
}

void Controller::imgTransform(bool done/* = false*/, double elapsed/* = 0.*/) const {
	if(done) {
		reportTransformationProgress(1.);

		const string &comparatorOverlayText = textForComparatorOverlay(elapsed);
		cout<<comparatorOverlayText <<endl<<endl;
		comp.setOverlay(comparatorOverlayText, 3000);

	} else {
		reportTransformationProgress(0.);
	}
}

Timer Controller::createTimerForImgTransform() const {
	return std::move(Timer(std::make_shared<Controller::TimerActions_ImgTransform>(*this)));
}

void Controller::restoreUserDefaultMatchSettings() {
#ifndef UNIT_TESTING
	MatchSettingsManip::instance().loadUserDefaults(cfg.ms);
#endif
	cp.updateMatchSettings(cfg.ms);
}

void Controller::setUserDefaultMatchSettings() const {
#ifndef UNIT_TESTING
	MatchSettingsManip::instance().saveUserDefaults(cfg.ms);
#endif
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

Controller::TimerActions_Controller::TimerActions_Controller(const Controller &ctrler_) :
	ctrler(ctrler_) {}

Controller::TimerActions_SymSetUpdate::TimerActions_SymSetUpdate(const Controller &ctrler_) :
		TimerActions_Controller(ctrler_) {}

void Controller::TimerActions_SymSetUpdate::onStart() {
	ctrler.symsSetUpdate();
}

void Controller::TimerActions_SymSetUpdate::onRelease(double elapsedS) {
	ctrler.symsSetUpdate(true, elapsedS);
}

Controller::TimerActions_ImgTransform::TimerActions_ImgTransform(const Controller &ctrler_) :
		TimerActions_Controller(ctrler_) {}

void Controller::TimerActions_ImgTransform::onStart() {
	ctrler.imgTransform();
}

void Controller::TimerActions_ImgTransform::onRelease(double elapsedS) {
	ctrler.imgTransform(true, elapsedS);
}

// Methods from below have different definitions for UnitTesting project
#ifndef UNIT_TESTING

Controller::~Controller() {
	destroyAllWindows();
}

void Controller::hourGlass(double progress, const string &title/* = ""*/) const {
	static const String waitWin = "Please Wait!";
	if(progress == 0.) {
		namedWindow(waitWin, CV_GUI_NORMAL); // no status bar, nor toolbar
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
		const string &hourGlassText = textHourGlass(oss.str(), progress);
		setWindowTitle(waitWin, hourGlassText);
	}
}

void Controller::reportGlyphProgress(double progress) const {
	extern const string Controller_PREFIX_GLYPH_PROGRESS;
	hourGlass(progress, Controller_PREFIX_GLYPH_PROGRESS);
}

void Controller::reportTransformationProgress(double progress) const {
	extern const string Controller_PREFIX_TRANSFORMATION_PROGRESS;
	hourGlass(progress, Controller_PREFIX_TRANSFORMATION_PROGRESS);
	if(0. == progress) {
		if(nullptr == resizedImg)
			throw logic_error("Please call Controller::updateResizedImg at the start of transformation!");
		comp.setReference(resizedImg->get()); // display 'original' when starting transformation
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
