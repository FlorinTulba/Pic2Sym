/**********************************************************
 Project:     Pic2Sym
 File:        controller.cpp

 Author:      Florin Tulba
 Created on:  2016-1-22
 
 Copyright (c) 2016 Florin Tulba
 **********************************************************/

#ifndef UNIT_TESTING

#include "controller.h"
#include "misc.h"

#include <Windows.h>
#include <sstream>

#include <boost/filesystem/operations.hpp>
#include <opencv2/highgui.hpp>

using namespace std;
using namespace boost::filesystem;
using namespace cv;

Controller::Controller(const string &cmd) :
		img(*this), fe(*this), cfg(*this, cmd),
		me(*this, cfg, fe), t(*this, cfg, me, img),
		comp(*this), cp(*this),
		hMaxSymsOk(Config::isHmaxSymsOk(cfg.getMaxHSyms())),
		vMaxSymsOk(Config::isVmaxSymsOk(cfg.getMaxVSyms())),
		fontSzOk(Config::isFontSizeOk(cfg.getFontSz())) {
	comp.setPos(0, 0);
	comp.permitResize(false);
	comp.setTitle("Pic2Sym - (c) 2016 Florin Tulba");
	comp.setStatus("Press Ctrl+P for Control Panel; ESC to Exit");
}

Controller::~Controller() {
	destroyAllWindows();
}

void Controller::handleRequests() const {
	for(;;) {
		// When pressing ESC, prompt the user if he wants to exit
		if(27 == waitKey() &&
		   IDYES == MessageBox(nullptr, L"Do you want to leave the application?", L"Question",
		   MB_ICONQUESTION | MB_YESNOCANCEL | MB_TASKMODAL | MB_SETFOREGROUND))
		   break;
	}
}

bool Controller::validState(bool imageReguired/* = true*/) const {
	if(((imageOk && hMaxSymsOk && vMaxSymsOk) || !imageReguired) &&
	   fontFamilyOk && fontSzOk)
		return true;

	ostringstream oss;
	oss<<"The problems are:"<<endl<<endl;
	if(imageReguired && !imageOk)
		oss<<"- no image to transform"<<endl;
	if(imageReguired && !hMaxSymsOk)
		oss<<"- max count of symbols horizontally is too small"<<endl;
	if(imageReguired && !vMaxSymsOk)
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
	oss<<"Font type: '"<<fe.getFamily()<<' '<<fe.getStyle()
		<<"' Size: "<<cfg.getFontSz()<<" Encoding: '"<<fe.getEncoding()<<'\'';
	pCmi->setStatus(oss.str());
}

void Controller::symbolsChanged() {
	fe.setFontSz(cfg.getFontSz());
	me.updateSymbols();

	updateCmapStatusBar();
	pCmi->updatePagesCount((unsigned)fe.symsSet().size());
	pCmi->showPage(0U);
}

void Controller::newFontFamily(const string &fontFile) {
	if(fe.fontFileName().compare(fontFile) == 0)
		return; // same font

	if(!fe.newFont(fontFile)) {
		ostringstream oss;
		oss<<"Invalid font file: '"<<fontFile<<'\'';
		errMsg(oss.str());
		return;
	}

	cp.updateEncodingsCount(fe.uniqueEncodings());

	if(!fontFamilyOk) {
		fontFamilyOk = true;
		pCmi = make_shared<CmapInspect>(*this);
		pCmi->setPos(424, 0);		// Place cmap window on x axis between 424..1064
		pCmi->permitResize(false);	// Ensure the user sees the symbols exactly their size
	}

	symbolsChanged();
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

void Controller::newFontSize(int fontSz) {
	if(!Config::isFontSizeOk(fontSz)) {
		fontSzOk = false;
		ostringstream oss;
		oss<<"Invalid font size: "<<fontSz<<". Please set at least "<<Config::MIN_FONT_SIZE<<'.';
		errMsg(oss.str());
		return;
	}

	if(!fontSzOk)
		fontSzOk = true;

	if(!fontFamilyOk || (unsigned)fontSz == cfg.getFontSz())
		return;

	cfg.setFontSz(fontSz);
	pCmi->updateGrid();
	
	symbolsChanged();
}

void Controller::newHmaxSyms(int maxSymbols) {
	if(!Config::isHmaxSymsOk(maxSymbols)) {
		hMaxSymsOk = false;
		ostringstream oss;
		oss<<"Invalid max number of horizontal symbols: "<<maxSymbols<<". Please set at least "<<Config::MIN_H_SYMS<<'.';
		errMsg(oss.str());
		return;
	}

	if(!hMaxSymsOk)
		hMaxSymsOk = true;

	if((unsigned)maxSymbols == cfg.getMaxHSyms())
		return;

	cfg.setMaxHSyms(maxSymbols);
}

void Controller::newVmaxSyms(int maxSymbols) {
	if(!Config::isVmaxSymsOk(maxSymbols)) {
		vMaxSymsOk = false;
		ostringstream oss;
		oss<<"Invalid max number of vertical symbols: "<<maxSymbols<<". Please set at least "<<Config::MIN_V_SYMS<<'.';
		errMsg(oss.str());
		return;
	}

	if(!vMaxSymsOk)
		vMaxSymsOk = true;

	if((unsigned)maxSymbols == cfg.getMaxVSyms())
		return;

	cfg.setMaxVSyms(maxSymbols);
}

void Controller::newThreshold4BlanksFactor(unsigned threshold) {
	if((unsigned)threshold != cfg.getBlankThreshold())
		cfg.setBlankThreshold(threshold);
}

void Controller::newContrastFactor(double k) {
	if(k != cfg.get_kContrast())
		cfg.set_kContrast(k);
}

void Controller::newUnderGlyphCorrectnessFactor(double k) {
	if(k != cfg.get_kSdevFg())
		cfg.set_kSdevFg(k);
}

void Controller::newAsideGlyphCorrectnessFactor(double k) {
	if(k != cfg.get_kSdevBg())
		cfg.set_kSdevBg(k);
}

void Controller::newGlyphEdgeCorrectnessFactor(double k) {
	if(k != cfg.get_kSdevEdge())
		cfg.set_kSdevEdge(k);
}

void Controller::newDirectionalSmoothnessFactor(double k) {
	if(k != cfg.get_kCosAngleMCs())
		cfg.set_kCosAngleMCs(k);
}

void Controller::newGravitationalSmoothnessFactor(double k) {
	if(k != cfg.get_kMCsOffset())
		cfg.set_kMCsOffset(k);
}

void Controller::newGlyphWeightFactor(double k) {
	if(k != cfg.get_kGlyphWeight())
		cfg.set_kGlyphWeight(k);
}

MatchEngine::VSymDataCItPair Controller::getFontFaces(unsigned from, unsigned maxCount) const {
	return me.getSymsRange(from, maxCount);
}

void Controller::hourGlass(double progress, const string &title/* = ""*/) {
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

void Controller::reportGlyphProgress(double progress) {
	hourGlass(progress, "Processing glyphs. Please wait");
}

void Controller::reportTransformationProgress(double progress) {
	hourGlass(progress, "Transforming image. Please wait");
	if(progress == 0.)
		comp.setReference(img.getResized()); // display 'original' when starting transformation
	else if(progress == 1.)
		comp.setResult(t.getResult()); // display the result at the end of the transformation
}

void Controller::performTransformation() {
	if(!validState())
		return;

	t.run();
}
#endif // UNIT_TESTING not defined