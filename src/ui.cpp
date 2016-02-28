/**********************************************************
 Project:     Pic2Sym
 File:        ui.cpp

 Author:      Florin Tulba
 Created on:  2016-1-22
 
 Copyright (c) 2016 Florin Tulba
 **********************************************************/

#include "controller.h"
#include "dlgs.h"

#include <opencv2/highgui.hpp>

using namespace std;
using namespace cv;

const String Comparator::transpTrackName = "Transparency %";
const double Comparator::defaultTransparency = .25;
const Mat Comparator::noImage = imread("res/NoImage.jpg");

const String CmapInspect::pageTrackName = "Cmap Page:";
const Size CmapInspect::pageSz(640, 480);

const double ControlPanel::Converter::StructuralSim::maxReal = 2.5;
const double ControlPanel::Converter::Correctness::maxReal = 2.5;
const double ControlPanel::Converter::Contrast::maxReal = 2.5;
const double ControlPanel::Converter::Gravity::maxReal = 2.;
const double ControlPanel::Converter::Direction::maxReal = 2.;
const double ControlPanel::Converter::LargerSym::maxReal = 1.;

const String ControlPanel::fontSzTrName = "Font size:";
const String ControlPanel::encodingTrName = "Encoding:";
const String ControlPanel::structuralSimTrName = "Struct Sim:";
const String ControlPanel::underGlyphCorrectnessTrName = "Fit under:";
const String ControlPanel::glyphEdgeCorrectnessTrName = "Fit edge:";
const String ControlPanel::asideGlyphCorrectnessTrName = "Fit aside:";
const String ControlPanel::moreContrastTrName = "Contrast:";
const String ControlPanel::gravityTrName = "Mass Center:";
const String ControlPanel::directionTrName = "Direction:";
const String ControlPanel::largerSymTrName = "Larger Symbols:";
const String ControlPanel::thresh4BlanksTrName = "Blanks below:";
const String ControlPanel::outWTrName = "Max horizontally:";
const String ControlPanel::outHTrName = "Max vertically:";

CvWin::CvWin(const String &winName_) : winName(winName_) {
	namedWindow(winName);
}

void CvWin::setTitle(const std::string &title) const {
	setWindowTitle(winName, title);
}

void CvWin::setOverlay(const std::string &overlay, int timeoutMs/* = 0*/) const {
	displayOverlay(winName, overlay, timeoutMs);
}

void CvWin::setStatus(const std::string &status, int timeoutMs/* = 0*/) const {
	displayStatusBar(winName, status, timeoutMs);
}

void CvWin::setPos(int x, int y) const {
	moveWindow(winName, x, y);
}

void CvWin::permitResize(bool allow/* = true*/) const {
	if(allow) {
		setWindowProperty(winName, CV_WND_PROP_AUTOSIZE, CV_WINDOW_NORMAL);
	} else {
		setWindowProperty(winName, CV_WND_PROP_AUTOSIZE, CV_WINDOW_AUTOSIZE);
		setWindowProperty(winName, CV_WND_PROP_ASPECTRATIO, CV_WINDOW_KEEPRATIO);
	}
	waitKey(1); // makes sure it works
}

void CvWin::resize(int w, int h) const {
	resizeWindow(winName, w, h);
}

Comparator::Comparator(void** /*hackParam = nullptr*/) : CvWin("Pic2Sym") {
	content = noImage;
	createTrackbar(transpTrackName, winName,
				   &trackPos, trackMax,
				   &Comparator::updateTransparency, reinterpret_cast<void*>(this));
	Comparator::updateTransparency(trackPos, reinterpret_cast<void*>(this)); // mandatory
}

void Comparator::setTransparency(double transparency) {
	if(!initial.empty() && !result.empty()) {
		content.release(); // seems to be mandatory
		addWeighted(initial, transparency, result, 1.-transparency, 0., content);
	}
	imshow(winName, content);
}

void Comparator::setReference(const Mat &ref_) {
	if(ref_.empty())
		throw invalid_argument("Please provide a non-empty image to Comparator::setReference()!");
	initial = content = ref_;
	if(!result.empty())
		result.release();
	if(trackPos != trackMax)
		setTrackbarPos(transpTrackName, winName, trackMax);
	else
		setTransparency(1.);
}

void Comparator::setResult(const Mat &res_, int transparency/* = (int)round(defaultTransparency * trackMax)*/) {
	if(initial.empty())
		throw logic_error("Please call Comparator::setResult() after Comparator::setReference()!");
	if(initial.type() != res_.type() || initial.size != res_.size)
		throw invalid_argument("Please provide a resulted image of the same size & type as the original image!");
	result = res_;
	if(trackPos != transparency)
		cv::setTrackbarPos(transpTrackName, winName, transparency);
	else
		setTransparency(defaultTransparency);
}

void Comparator::updateTransparency(int newTransp, void *userdata) {
	Comparator *pComp = reinterpret_cast<Comparator*>(userdata);
	pComp->setTransparency((double)newTransp/trackMax);
}

CmapInspect::CmapInspect(const Controller &ctrler_) : CvWin("Charmap View"), ctrler(ctrler_),
		grid(content = createGrid()), symsPerPage(computeSymsPerPage()) {
	createTrackbar(pageTrackName, winName, &page, 1, &CmapInspect::updatePageIdx,
				   reinterpret_cast<void*>(this));
	CmapInspect::updatePageIdx(page, reinterpret_cast<void*>(this)); // mandatory
}

unsigned CmapInspect::computeSymsPerPage() const {
	const int cellSide = 1+ctrler.getFontSize();
	return (pageSz.width / cellSide) * (pageSz.height / cellSide);
}

void CmapInspect::updateGrid() {
	grid = createGrid();
}

Mat CmapInspect::createGrid() const {
	Mat result(pageSz, CV_8UC1, Scalar(255U));

	// Draws horizontal & vertical cell borders
	const int cellSide = 1+ctrler.getFontSize();
	for(int i = cellSide-1; i < pageSz.width; i += cellSide)
		line(result, Point(i, 0), Point(i, pageSz.height-1), 200);
	for(int i = cellSide-1; i < pageSz.height; i += cellSide)
		line(result, Point(0, i), Point(pageSz.width-1, i), 200);
	return result;
}

void CmapInspect::updatePagesCount(unsigned cmapSize) {
	updatingPageMax = true;
	symsPerPage = computeSymsPerPage();
	pagesCount = (unsigned)ceil(cmapSize / (double)symsPerPage);
	setTrackbarMax(pageTrackName, winName, max(1, (int)pagesCount-1));

	// Sequence from below is required to really update the trackbar max & pos
	// The controller should prevent them to trigger CmapInspect::updatePageIdx
	setTrackbarPos(pageTrackName, winName, 1);
	updatingPageMax = false;
	setTrackbarPos(pageTrackName, winName, 0); // => page = 0
}

void CmapInspect::showPage(unsigned pageIdx) {
	// Ignore call if pageIdx isn't valid,
	// or if the required hack (mentioned in 'ui.h') provoked this call
	if(updatingPageMax || pageIdx >= pagesCount)
		return;

	if((unsigned)page != pageIdx)
		setTrackbarPos(pageTrackName, winName, pageIdx); // => page = pageIdx

	populateGrid(ctrler.getFontFaces(symsPerPage*pageIdx, symsPerPage));
	imshow(winName, content);
}

void CmapInspect::populateGrid(const MatchEngine::VSymDataCItPair &itPair) {
	MatchEngine::VSymDataCIt it = itPair.first, itEnd = itPair.second;
	content = grid.clone();
	const int fontSz = ctrler.getFontSize(),
		fontSzM1 = fontSz - 1,
		cellSide = 1 + fontSz,
		height = pageSz.height,
		width = pageSz.width;

	// Convert each 'negative' glyph to 0..255 and place it within the grid
	for(int r = fontSzM1; it!=itEnd && r < height; r += cellSide) {
		Range rowRange(r - fontSzM1, r + 1);
		for(int c = fontSzM1; it!=itEnd && c < width; c += cellSide, ++it) {
			Mat region(content, rowRange, Range(c - fontSzM1, c + 1));
			it->symAndMasks[SymData::NEG_SYM_IDX].copyTo(region);
		}
	}
}

void CmapInspect::updatePageIdx(int newPage, void *userdata) {
	CmapInspect *pCmi = reinterpret_cast<CmapInspect*>(userdata);
	pCmi->showPage(newPage);
}

double ControlPanel::Converter::proportionRule(double x, double xMax, double yMax) {
	return x * yMax / xMax;
}

int ControlPanel::Converter::StructuralSim::toSlider(double ssim) {
	return (int)round(proportionRule(ssim, maxReal, maxSlider));
}

double ControlPanel::Converter::StructuralSim::fromSlider(int ssim) {
	return proportionRule(ssim, maxSlider, maxReal);
}

int ControlPanel::Converter::Contrast::toSlider(double contrast) {
	return (int)round(proportionRule(contrast, maxReal, maxSlider));
}

double ControlPanel::Converter::Contrast::fromSlider(int contrast) {
	return proportionRule(contrast, maxSlider, maxReal);
}

int ControlPanel::Converter::Correctness::toSlider(double correctness) {
	return (int)round(proportionRule(correctness, maxReal, maxSlider));
}

double ControlPanel::Converter::Correctness::fromSlider(int correctness) {
	return proportionRule(correctness, maxSlider, maxReal);
}

int ControlPanel::Converter::Direction::toSlider(double direction) {
	return (int)round(proportionRule(direction, maxReal, maxSlider));
}

double ControlPanel::Converter::Direction::fromSlider(int direction) {
	return proportionRule(direction, maxSlider, maxReal);
}

int ControlPanel::Converter::Gravity::toSlider(double gravity) {
	return (int)round(proportionRule(gravity, maxReal, maxSlider));
}

double ControlPanel::Converter::Gravity::fromSlider(int gravity) {
	return proportionRule(gravity, maxSlider, maxReal);
}

int ControlPanel::Converter::LargerSym::toSlider(double largerSym) {
	return (int)round(proportionRule(largerSym, maxReal, maxSlider));
}

double ControlPanel::Converter::LargerSym::fromSlider(int largerSym) {
	return proportionRule(largerSym, maxSlider, maxReal);
}

ControlPanel::ControlPanel(Controller &ctrler_, const Settings &cfg) :
		ctrler(ctrler_),
		maxHSyms(cfg.imgSettings().getMaxHSyms()), maxVSyms(cfg.imgSettings().getMaxVSyms()),
		encoding(0U), fontSz(cfg.symSettings().getFontSz()),
		structuralSim(Converter::StructuralSim::toSlider(cfg.matchSettings().get_kSsim())),
		underGlyphCorrectness(Converter::Correctness::toSlider(cfg.matchSettings().get_kSdevFg())),
		glyphEdgeCorrectness(Converter::Correctness::toSlider(cfg.matchSettings().get_kSdevEdge())),
		asideGlyphCorrectness(Converter::Correctness::toSlider(cfg.matchSettings().get_kSdevBg())),
		moreContrast(Converter::Contrast::toSlider(cfg.matchSettings().get_kContrast())),
		gravity(Converter::Gravity::toSlider(cfg.matchSettings().get_kMCsOffset())),
		direction(Converter::Direction::toSlider(cfg.matchSettings().get_kCosAngleMCs())),
		largerSym(Converter::LargerSym::toSlider(cfg.matchSettings().get_kSymDensity())),
		thresh4Blanks(cfg.matchSettings().getBlankThreshold()) {

	createButton("Select an Image",
				 [] (int, void *userdata) {
		Controller *pCtrler = reinterpret_cast<Controller*>(userdata);
		static ImgSelector is;
		if(is.promptForUserChoice())
			pCtrler->newImage(is.selection());
	}, reinterpret_cast<void*>(&ctrler));
	createButton("Transform the Image",
				 [] (int, void *userdata) {
		Controller *pCtrler = reinterpret_cast<Controller*>(userdata);
		pCtrler->performTransformation();
	}, reinterpret_cast<void*>(&ctrler));

	createTrackbar(outWTrName, nullptr, &maxHSyms, Settings::MAX_H_SYMS,
				   [] (int val, void *userdata) {
		Controller *pCtrler = reinterpret_cast<Controller*>(userdata);
		pCtrler->newHmaxSyms(val);
	}, reinterpret_cast<void*>(&ctrler));
	createTrackbar(outHTrName, nullptr, &maxVSyms, Settings::MAX_V_SYMS,
				   [] (int val, void *userdata) {
		Controller *pCtrler = reinterpret_cast<Controller*>(userdata);
		pCtrler->newVmaxSyms(val);
	}, reinterpret_cast<void*>(&ctrler));

	createButton("Select a Scalable, preferably also Monospaced Font Family",
				 [] (int, void *userdata) {
		Controller *pCtrler = reinterpret_cast<Controller*>(userdata);
		static SelectFont sf;
		if(sf.promptForUserChoice())
			pCtrler->newFontFamily(sf.selection());
	}, reinterpret_cast<void*>(&ctrler));

	createTrackbar(encodingTrName, nullptr, &encoding, 1,
				   [] (int val, void *userdata) {
		Controller *pCtrler = reinterpret_cast<Controller*>(userdata);
		pCtrler->newFontEncoding(val);
	}, reinterpret_cast<void*>(&ctrler));
	createTrackbar(fontSzTrName, nullptr, &fontSz, Settings::MAX_FONT_SIZE,
				   [] (int val, void *userdata) {
		Controller *pCtrler = reinterpret_cast<Controller*>(userdata);
		pCtrler->newFontSize(val);
	}, reinterpret_cast<void*>(&ctrler));

	createButton("Restore defaults for values below", [] (int, void *userdata) {
		Controller *pCtrler = reinterpret_cast<Controller*>(userdata);
		pCtrler->restoreUserDefaultMatchSettings();
	}, reinterpret_cast<void*>(&ctrler));
	createButton("Set as defaults the values below", [] (int, void *userdata) {
		Controller *pCtrler = reinterpret_cast<Controller*>(userdata);
		pCtrler->setUserDefaultMatchSettings();
	}, reinterpret_cast<void*>(&ctrler));

	createTrackbar(structuralSimTrName, nullptr, &structuralSim, Converter::StructuralSim::maxSlider,
				   [] (int val, void *userdata) {
		Controller *pCtrler = reinterpret_cast<Controller*>(userdata);
		pCtrler->newStructuralSimilarityFactor(Converter::StructuralSim::fromSlider(val));
	}, reinterpret_cast<void*>(&ctrler));
	createTrackbar(underGlyphCorrectnessTrName, nullptr, &underGlyphCorrectness, Converter::Correctness::maxSlider,
				   [] (int val, void *userdata) {
		Controller *pCtrler = reinterpret_cast<Controller*>(userdata);
		pCtrler->newUnderGlyphCorrectnessFactor(Converter::Correctness::fromSlider(val));
	}, reinterpret_cast<void*>(&ctrler));
	createTrackbar(glyphEdgeCorrectnessTrName, nullptr, &glyphEdgeCorrectness, Converter::Correctness::maxSlider,
				   [] (int val, void *userdata) {
		Controller *pCtrler = reinterpret_cast<Controller*>(userdata);
		pCtrler->newGlyphEdgeCorrectnessFactor(Converter::Correctness::fromSlider(val));
	}, reinterpret_cast<void*>(&ctrler));
	createTrackbar(asideGlyphCorrectnessTrName, nullptr, &asideGlyphCorrectness, Converter::Correctness::maxSlider,
				   [] (int val, void *userdata) {
		Controller *pCtrler = reinterpret_cast<Controller*>(userdata);
		pCtrler->newAsideGlyphCorrectnessFactor(Converter::Correctness::fromSlider(val));
	}, reinterpret_cast<void*>(&ctrler));

	createTrackbar(moreContrastTrName, nullptr, &moreContrast, Converter::Contrast::maxSlider,
				   [] (int val, void *userdata) {
		Controller *pCtrler = reinterpret_cast<Controller*>(userdata);
		pCtrler->newContrastFactor(Converter::Contrast::fromSlider(val));
	}, reinterpret_cast<void*>(&ctrler));

	createTrackbar(gravityTrName, nullptr, &gravity, Converter::Gravity::maxSlider,
				   [] (int val, void *userdata) {
		Controller *pCtrler = reinterpret_cast<Controller*>(userdata);
		pCtrler->newGravitationalSmoothnessFactor(Converter::Gravity::fromSlider(val));
	}, reinterpret_cast<void*>(&ctrler));
	createTrackbar(directionTrName, nullptr, &direction, Converter::Direction::maxSlider,
				   [] (int val, void *userdata) {
		Controller *pCtrler = reinterpret_cast<Controller*>(userdata);
		pCtrler->newDirectionalSmoothnessFactor(Converter::Direction::fromSlider(val));
	}, reinterpret_cast<void*>(&ctrler));

	createTrackbar(largerSymTrName, nullptr, &largerSym, Converter::LargerSym::maxSlider,
				   [] (int val, void *userdata) {
		Controller *pCtrler = reinterpret_cast<Controller*>(userdata);
		pCtrler->newGlyphWeightFactor(Converter::LargerSym::fromSlider(val));
	}, reinterpret_cast<void*>(&ctrler));

	createTrackbar(thresh4BlanksTrName, nullptr, &thresh4Blanks, Settings::MAX_THRESHOLD_FOR_BLANKS,
				   [] (int val, void *userdata) {
		Controller *pCtrler = reinterpret_cast<Controller*>(userdata);
		pCtrler->newThreshold4BlanksFactor(val);
	}, reinterpret_cast<void*>(&ctrler));

	createButton("Instructions", [] (int, void*) {
		MessageBox(nullptr, L"\tPic2Sym (v1.0) - by Florin Tulba\n\n" \
					L"\tThe Control Panel allows setting:\n\n" \
					L"- which image to be approximated by symbols from a charset\n" \
					L"- maximum number of symbols used horizontally & vertically\n" \
					L"- which font family provides these symbols\n" \
					L"- the desired encoding within the selected font family\n" \
					L"- the size of these symbols\n" \
					L"- a factor to encourage structural similarity\n" \
					L"- factors to favor better correspondence of foreground\n" \
					L"   (under glyph) / contours (edges of the glyph) /\n" \
					L"   background (around, aside glyph)\n" \
					L"- a factor to encourage matching symbols with large contrast\n" \
					L"- a factor to encourage 'gravitational' smoothness\n" \
					L"   (match patch 'mass center')\n" \
					L"- a factor to enhance directionality (match patch 'gradient')\n" \
					L"- a factor to favor selecting larger symbols over small ones\n" \
					L"- the threshold contrast of determined matching glyphs below\n" \
					L"   which to replace these barely visible matches with Blanks\n\n" \
					L"The rudimentary sliders used here won't always show valid ranges.\n" \
					L"They all must be integer, start from 0, end at least on 1.\n" \
					L"When their labels is truncated, clicking on them will help.\n\n" \
					L"There's more information in ReadMe.md.\n",
					L"Instructions", MB_ICONINFORMATION | MB_OK | MB_TASKMODAL | MB_SETFOREGROUND);
	});
	createButton("Load Settings", [] (int, void *userdata) {
		Controller *pCtrler = reinterpret_cast<Controller*>(userdata);
		pCtrler->loadSettings();
	}, reinterpret_cast<void*>(&ctrler));
	createButton("Save Settings", [] (int, void *userdata) {
		Controller *pCtrler = reinterpret_cast<Controller*>(userdata);
		pCtrler->saveSettings();
	}, reinterpret_cast<void*>(&ctrler));
}

void ControlPanel::updateMatchSettings(const MatchSettings &ms) {
	int newVal = Converter::StructuralSim::toSlider(ms.get_kSsim());
	while(structuralSim != newVal)
		setTrackbarPos(structuralSimTrName, nullptr, newVal);

	newVal = Converter::Correctness::toSlider(ms.get_kSdevFg());
	while(underGlyphCorrectness != newVal)
		setTrackbarPos(underGlyphCorrectnessTrName, nullptr, newVal);

	newVal = Converter::Correctness::toSlider(ms.get_kSdevEdge());
	while(glyphEdgeCorrectness != newVal)
		setTrackbarPos(glyphEdgeCorrectnessTrName, nullptr, newVal);

	newVal = Converter::Correctness::toSlider(ms.get_kSdevBg());
	while(asideGlyphCorrectness != newVal)
		setTrackbarPos(asideGlyphCorrectnessTrName, nullptr, newVal);

	newVal = Converter::Contrast::toSlider(ms.get_kContrast());
	while(moreContrast != newVal)
		setTrackbarPos(moreContrastTrName, nullptr, newVal);

	newVal = Converter::Gravity::toSlider(ms.get_kMCsOffset());
	while(gravity != newVal)
		setTrackbarPos(gravityTrName, nullptr, newVal);

	newVal = Converter::Direction::toSlider(ms.get_kCosAngleMCs());
	while(direction != newVal)
		setTrackbarPos(directionTrName, nullptr, newVal);

	newVal = Converter::LargerSym::toSlider(ms.get_kSymDensity());
	while(largerSym != newVal)
		setTrackbarPos(largerSymTrName, nullptr, newVal);

	newVal = ms.getBlankThreshold();
	while(thresh4Blanks != newVal)
		setTrackbarPos(thresh4BlanksTrName, nullptr, newVal);
}

void ControlPanel::updateImgSettings(const ImgSettings &is) {
	int newVal = is.getMaxHSyms();
	while(maxHSyms != newVal)
		setTrackbarPos(outWTrName, nullptr, newVal);

	newVal = is.getMaxVSyms();
	while(maxVSyms != newVal)
		setTrackbarPos(outHTrName, nullptr, newVal);
}

void ControlPanel::updateSymSettings(unsigned encIdx, unsigned fontSz_) {
	while(encoding != encIdx)
		setTrackbarPos(encodingTrName, nullptr, encIdx);

	while(fontSz != fontSz_)
		setTrackbarPos(fontSzTrName, nullptr, fontSz_);
}

void ControlPanel::updateEncodingsCount(unsigned uniqueEncodings) {
	updatingEncMax = true;
	setTrackbarMax(encodingTrName, nullptr, max(1, int(uniqueEncodings-1U)));

	// Sequence from below is required to really update the trackbar max & pos
	// The controller should prevent them to trigger Controller::newFontEncoding
	setTrackbarPos(encodingTrName, nullptr, 1);
	updatingEncMax = false;
	setTrackbarPos(encodingTrName, nullptr, 0);
}
