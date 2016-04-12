/****************************************************************************************
 The application Pic2Sym approximates images by a
 grid of colored symbols with colored backgrounds.

 This file was created on 2016-4-9
 and belongs to the Pic2Sym project.

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
#include "settings.h"
#include "matchParams.h"
#include "controlPanel.h"
#include "views.h"
#include "dlgs.h"
#include "misc.h"

#include <iomanip>

#include <opencv2/highgui.hpp>

using namespace std;
using namespace boost;
using namespace cv;

void pauseAfterError() {
	string line;
	cout<<endl<<"Press Enter to leave"<<endl;
	getline(cin, line);
}

void showUsage() {
	cout<<"Usage:"<<endl;
	cout<<"There are 2 launch modes:"<<endl;
	cout<<"A) Normal launch mode (no parameters)"<<endl;
	cout<<"		Pic2Sym.exe"<<endl<<endl;
	cout<<"B) View mismatches launch mode (Support for Unit Testing, using 1 parameters)"<<endl;
	cout<<"		Pic2Sym.exe \"<testTitle>\""<<endl<<endl;
	pauseAfterError();
}

ostream& operator<<(ostream &os, const Settings &s) {
	os<<s.ss<<s.is<<s.ms<<endl;
	return os;
}

ostream& operator<<(ostream &os, const ImgSettings &is) {
	os<<"hMaxSyms"<<" : "<<is.hMaxSyms<<endl;
	os<<"vMaxSyms"<<" : "<<is.vMaxSyms<<endl;
	return os;
}

ostream& operator<<(ostream &os, const SymSettings &ss) {
	os<<"fontFile"<<" : "<<ss.fontFile<<endl;
	os<<"encoding"<<" : "<<ss.encoding<<endl;
	os<<"fontSz"<<" : "<<ss.fontSz<<endl;
	return os;
}

ostream& operator<<(ostream &os, const MatchSettings &c) {
	if(c.hybridResultMode)
		os<<"hybridResultMode"<<" : "<<boolalpha<<c.hybridResultMode<<endl;
	if(c.kSsim > 0.)
		os<<"kSsim"<<" : "<<c.kSsim<<endl;
	if(c.kSdevFg > 0.)
		os<<"kSdevFg"<<" : "<<c.kSdevFg<<endl;
	if(c.kSdevEdge > 0.)
		os<<"kSdevEdge"<<" : "<<c.kSdevEdge<<endl;
	if(c.kSdevBg > 0.)
		os<<"kSdevBg"<<" : "<<c.kSdevBg<<endl;
	if(c.kContrast > 0.)
		os<<"kContrast"<<" : "<<c.kContrast<<endl;
	if(c.kMCsOffset > 0.)
		os<<"kMCsOffset"<<" : "<<c.kMCsOffset<<endl;
	if(c.kCosAngleMCs > 0.)
		os<<"kCosAngleMCs"<<" : "<<c.kCosAngleMCs<<endl;
	if(c.kSymDensity > 0.)
		os<<"kSymDensity"<<" : "<<c.kSymDensity<<endl;
	if(c.threshold4Blank > 0.)
		os<<"threshold4Blank"<<" : "<<c.threshold4Blank<<endl;
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
	extern const string Comparator_initial_title;
	comp.setTitle(Comparator_initial_title);
	extern const string Comparator_statusBar;
	comp.setStatus(Comparator_statusBar);
}

const string Controller::textForCmapOverlay(double elapsed) const {
	ostringstream oss;
	oss<<"The update of the symbols set took "<<elapsed<<" s!";
	return oss.str();
}

const string Controller::textForComparatorOverlay(double elapsed) const {
	ostringstream oss;
	oss<<"The transformation took "<<elapsed<<" s!";
	return oss.str();
}

const string Controller::textForCmapStatusBar() const {
	ostringstream oss;
	oss<<"Font type '"<<fe.getFamily()<<' '<<fe.getStyle()
		<<"', size "<<cfg.ss.getFontSz()<<", encoding '"<<fe.getEncoding()<<'\''
		<<" : "<<me.getSymsCount()<<" symbols";
	return oss.str();
}

const string Controller::textHourGlass(const std::string &prefix, double progress) const {
	ostringstream oss;
	oss<<prefix<<" ("<<fixed<<setprecision(0)<<progress*100.<<"%)";
	return oss.str();
}

#ifndef UNIT_TESTING
void Controller::handleRequests() {
	for(;;) {
		// When pressing ESC, prompt the user if he wants to exit
		if(27 == waitKey() &&
		   IDYES == MessageBox(nullptr, L"Do you want to leave the application?", L"Question",
		   MB_ICONQUESTION | MB_YESNOCANCEL | MB_TASKMODAL | MB_SETFOREGROUND))
		   break;
	}
}
#endif

string MatchEngine::getIdForSymsToUse() {
	const unsigned sz = cfg.symSettings().getFontSz();
	if(!Settings::isFontSizeOk(sz)) {
		cerr<<"Invalid font size to use: "<<sz<<endl;
		throw logic_error("Invalid font size in " __FUNCTION__);
	}

	ostringstream oss;
	oss<<fe.getFamily()<<'_'<<fe.getStyle()<<'_'<<fe.getEncoding()<<'_'<<sz;
	// this also throws logic_error if no family/style

	return oss.str();
}

const string Transformer::textStudiedCase(int rows, int cols) const {
	const auto &ss = cfg.matchSettings();
	ostringstream oss;
	oss<<img.name()<<'_'
		<<me.getIdForSymsToUse()<<'_'
		<<ss.isHybridResult()<<'_'
		<<ss.get_kSsim()<<'_'
		<<ss.get_kSdevFg()<<'_'<<ss.get_kSdevEdge()<<'_'<<ss.get_kSdevBg()<<'_'
		<<ss.get_kContrast()<<'_'<<ss.get_kMCsOffset()<<'_'<<ss.get_kCosAngleMCs()<<'_'
		<<ss.get_kSymDensity()<<'_'<<ss.getBlankThreshold()<<'_'
		<<cols<<'_'<<rows; // no extension yet
	return oss.str(); // this text is included in the result & trace file names
}

#ifndef UNIT_TESTING
Comparator::Comparator(void** /*hackParam = nullptr*/) : CvWin("Pic2Sym") {
	content = noImage;
	createTrackbar(transpTrackName, winName,
				   &trackPos, trackMax,
				   &Comparator::updateTransparency, reinterpret_cast<void*>(this));
	Comparator::updateTransparency(trackPos, reinterpret_cast<void*>(this)); // mandatory
}

void Comparator::resize() const {
	// Comparator window is to be placed within 1024x768 top-left part of the screen
	static const double
		HEIGHT_TITLE_TOOLBAR_SLIDER_STATUS = 70,
		WIDTH_LATERAL_BORDERS = 4,
		H_NUMERATOR = 768-HEIGHT_TITLE_TOOLBAR_SLIDER_STATUS, // desired max height - HEIGHT_TITLE_TOOLBAR_SLIDER_STATUS
		W_NUMERATOR = 1024-WIDTH_LATERAL_BORDERS; // desired max width - WIDTH_LATERAL_BORDERS

	// Resize window to preserve the aspect ratio of the loaded image,
	// while not enlarging it, nor exceeding 1024 x 768
	const int h = initial.rows, w = initial.cols;
	const double k = min(1., min(H_NUMERATOR/h, W_NUMERATOR/w));
	const int winHeight = (int)round(k*h+HEIGHT_TITLE_TOOLBAR_SLIDER_STATUS),
			winWidth = (int)round(k*w+WIDTH_LATERAL_BORDERS);

	CvWin::resize(winWidth, winHeight);
}

CmapInspect::CmapInspect(const Controller &ctrler_) :
		CvWin("Charmap View"), ctrler(ctrler_),
		grid(content = createGrid()), symsPerPage(computeSymsPerPage()) {
	createTrackbar(pageTrackName, winName, &page, 1, &CmapInspect::updatePageIdx,
				   reinterpret_cast<void*>(this));
	CmapInspect::updatePageIdx(page, reinterpret_cast<void*>(this)); // mandatory
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

ControlPanel::ControlPanel(Controller &ctrler_, const Settings &cfg) :
		ctrler(ctrler_),
		maxHSyms(cfg.imgSettings().getMaxHSyms()), maxVSyms(cfg.imgSettings().getMaxVSyms()),
		encoding(0U), fontSz(cfg.symSettings().getFontSz()),
		hybridResult(cfg.matchSettings().isHybridResult() ? 1 : 0),
		structuralSim(Converter::StructuralSim::toSlider(cfg.matchSettings().get_kSsim())),
		underGlyphCorrectness(Converter::Correctness::toSlider(cfg.matchSettings().get_kSdevFg())),
		glyphEdgeCorrectness(Converter::Correctness::toSlider(cfg.matchSettings().get_kSdevEdge())),
		asideGlyphCorrectness(Converter::Correctness::toSlider(cfg.matchSettings().get_kSdevBg())),
		moreContrast(Converter::Contrast::toSlider(cfg.matchSettings().get_kContrast())),
		gravity(Converter::Gravity::toSlider(cfg.matchSettings().get_kMCsOffset())),
		direction(Converter::Direction::toSlider(cfg.matchSettings().get_kCosAngleMCs())),
		largerSym(Converter::LargerSym::toSlider(cfg.matchSettings().get_kSymDensity())),
		thresh4Blanks(cfg.matchSettings().getBlankThreshold()) {
	createButton(selectImgLabel,
				 [] (int, void *userdata) {
		Controller *pCtrler = reinterpret_cast<Controller*>(userdata);
		static ImgSelector is;
		if(is.promptForUserChoice())
			pCtrler->newImage(is.selection());
	}, reinterpret_cast<void*>(&ctrler));
	createButton(transformImgLabel,
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

	createButton(selectFontLabel,
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

	createButton(restoreDefaultsLabel, [] (int, void *userdata) {
		Controller *pCtrler = reinterpret_cast<Controller*>(userdata);
		pCtrler->restoreUserDefaultMatchSettings();
	}, reinterpret_cast<void*>(&ctrler));
	createButton(saveAsDefaultsLabel, [] (int, void *userdata) {
		Controller *pCtrler = reinterpret_cast<Controller*>(userdata);
		pCtrler->setUserDefaultMatchSettings();
	}, reinterpret_cast<void*>(&ctrler));

	createTrackbar(hybridResultTrName, nullptr, &hybridResult, 1,
				   [] (int state, void *userdata) {
		Controller *pCtrler = reinterpret_cast<Controller*>(userdata);
		pCtrler->setResultMode(state != 0);
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

	createButton(aboutLabel, [] (int, void*) {
		MessageBox(nullptr, aboutText.c_str(),
				   str2wstr(aboutLabel).c_str(), MB_ICONINFORMATION | MB_OK | MB_TASKMODAL | MB_SETFOREGROUND);
	});
	createButton(instructionsLabel, [] (int, void*) {
		MessageBox(nullptr, instructionsText.c_str(),
				   str2wstr(instructionsLabel).c_str(), MB_ICONINFORMATION | MB_OK | MB_TASKMODAL | MB_SETFOREGROUND);
	});
	createButton(loadSettingsLabel, [] (int, void *userdata) {
		Controller *pCtrler = reinterpret_cast<Controller*>(userdata);
		pCtrler->loadSettings();
	}, reinterpret_cast<void*>(&ctrler));
	createButton(saveSettingsLabel, [] (int, void *userdata) {
		Controller *pCtrler = reinterpret_cast<Controller*>(userdata);
		pCtrler->saveSettings();
	}, reinterpret_cast<void*>(&ctrler));
}
#endif // UNIT_TESTING not defined