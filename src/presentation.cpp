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

void Controller::showAboutDlg(const string &title, const wstring &content) {
	MessageBox(nullptr, content.c_str(),
			   str2wstr(title).c_str(), MB_ICONINFORMATION | MB_OK | MB_TASKMODAL | MB_SETFOREGROUND);
}

void Controller::showInstructionsDlg(const string &title, const wstring &content) {
	MessageBox(nullptr, content.c_str(),
			   str2wstr(title).c_str(), MB_ICONINFORMATION | MB_OK | MB_TASKMODAL | MB_SETFOREGROUND);
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

	extern const int Comparator_trackMax;
	extern const String Comparator_transpTrackName;

	createTrackbar(Comparator_transpTrackName, winName,
				   &trackPos, Comparator_trackMax,
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

extern const Size CmapInspect_pageSz;
extern const String CmapInspect_pageTrackName;

CmapInspect::CmapInspect(const IPresentCmap &cmapPresenter_) :
		CvWin("Charmap View"), cmapPresenter(cmapPresenter_),
		grid(content = createGrid()), symsPerPage(computeSymsPerPage()) {
	createTrackbar(CmapInspect_pageTrackName, winName, &page, 1, &CmapInspect::updatePageIdx,
				   reinterpret_cast<void*>(this));
	CmapInspect::updatePageIdx(page, reinterpret_cast<void*>(this)); // mandatory
}

Mat CmapInspect::createGrid() const {
	Mat result(CmapInspect_pageSz, CV_8UC1, Scalar(255U));

	// Draws horizontal & vertical cell borders
	const int cellSide = 1+cmapPresenter.getFontSize();
	for(int i = cellSide-1; i < CmapInspect_pageSz.width; i += cellSide)
		line(result, Point(i, 0), Point(i, CmapInspect_pageSz.height-1), 200);
	for(int i = cellSide-1; i < CmapInspect_pageSz.height; i += cellSide)
		line(result, Point(0, i), Point(CmapInspect_pageSz.width-1, i), 200);
	return result;
}

void CmapInspect::populateGrid(const MatchEngine::VSymDataCItPair &itPair) {
	MatchEngine::VSymDataCIt it = itPair.first, itEnd = itPair.second;
	content = grid.clone();
	const int fontSz = cmapPresenter.getFontSize(),
			fontSzM1 = fontSz - 1,
			cellSide = 1 + fontSz,
			height = CmapInspect_pageSz.height,
			width = CmapInspect_pageSz.width;

	// Convert each 'negative' glyph to 0..255 and place it within the grid
	for(int r = fontSzM1; it!=itEnd && r < height; r += cellSide) {
		Range rowRange(r - fontSzM1, r + 1);
		for(int c = fontSzM1; it!=itEnd && c < width; c += cellSide, ++it) {
			Mat region(content, rowRange, Range(c - fontSzM1, c + 1));
			it->symAndMasks[SymData::NEG_SYM_IDX].copyTo(region);
		}
	}
}

extern const wstring ControlPanel_aboutText;
extern const wstring ControlPanel_instructionsText;
extern const String ControlPanel_aboutLabel;
extern const String ControlPanel_instructionsLabel;

ControlPanel::ControlPanel(IControlPanelActions &actions_, const Settings &cfg) :
		actions(actions_),
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
	extern const unsigned Settings_MAX_THRESHOLD_FOR_BLANKS;
	extern const unsigned Settings_MAX_H_SYMS;
	extern const unsigned Settings_MAX_V_SYMS;
	extern const unsigned Settings_MAX_FONT_SIZE;
	extern const int ControlPanel_Converter_StructuralSim_maxSlider;
	extern const int ControlPanel_Converter_Contrast_maxSlider;
	extern const int ControlPanel_Converter_Correctness_maxSlider;
	extern const int ControlPanel_Converter_Direction_maxSlider;
	extern const int ControlPanel_Converter_Gravity_maxSlider;
	extern const int ControlPanel_Converter_LargerSym_maxSlider;

	extern const String ControlPanel_selectImgLabel;
	extern const String ControlPanel_transformImgLabel;
	extern const String ControlPanel_selectFontLabel;
	extern const String ControlPanel_restoreDefaultsLabel;
	extern const String ControlPanel_saveAsDefaultsLabel;
	extern const String ControlPanel_loadSettingsLabel;
	extern const String ControlPanel_saveSettingsLabel;
	extern const String ControlPanel_fontSzTrName;
	extern const String ControlPanel_encodingTrName;
	extern const String ControlPanel_hybridResultTrName;
	extern const String ControlPanel_structuralSimTrName;
	extern const String ControlPanel_underGlyphCorrectnessTrName;
	extern const String ControlPanel_glyphEdgeCorrectnessTrName;
	extern const String ControlPanel_asideGlyphCorrectnessTrName;
	extern const String ControlPanel_moreContrastTrName;
	extern const String ControlPanel_gravityTrName;
	extern const String ControlPanel_directionTrName;
	extern const String ControlPanel_largerSymTrName;
	extern const String ControlPanel_thresh4BlanksTrName;
	extern const String ControlPanel_outWTrName;
	extern const String ControlPanel_outHTrName;

	createButton(ControlPanel_selectImgLabel,
				 [] (int, void *userdata) {
		IControlPanelActions *pActions = reinterpret_cast<IControlPanelActions*>(userdata);
		static ImgSelector is;
		if(is.promptForUserChoice())
			pActions->newImage(is.selection());
	}, reinterpret_cast<void*>(&actions));
	createButton(ControlPanel_transformImgLabel,
				 [] (int, void *userdata) {
		IControlPanelActions *pActions = reinterpret_cast<IControlPanelActions*>(userdata);
		pActions->performTransformation();
	}, reinterpret_cast<void*>(&actions));

	createTrackbar(ControlPanel_outWTrName, nullptr, &maxHSyms, Settings_MAX_H_SYMS,
				   [] (int val, void *userdata) {
		IControlPanelActions *pActions = reinterpret_cast<IControlPanelActions*>(userdata);
		pActions->newHmaxSyms(val);
	}, reinterpret_cast<void*>(&actions));
	createTrackbar(ControlPanel_outHTrName, nullptr, &maxVSyms, Settings_MAX_V_SYMS,
				   [] (int val, void *userdata) {
		IControlPanelActions *pActions = reinterpret_cast<IControlPanelActions*>(userdata);
		pActions->newVmaxSyms(val);
	}, reinterpret_cast<void*>(&actions));

	createButton(ControlPanel_selectFontLabel,
				 [] (int, void *userdata) {
		IControlPanelActions *pActions = reinterpret_cast<IControlPanelActions*>(userdata);
		static SelectFont sf;
		if(sf.promptForUserChoice())
			pActions->newFontFamily(sf.selection());
	}, reinterpret_cast<void*>(&actions));

	createTrackbar(ControlPanel_encodingTrName, nullptr, &encoding, 1,
				   [] (int val, void *userdata) {
		IControlPanelActions *pActions = reinterpret_cast<IControlPanelActions*>(userdata);
		pActions->newFontEncoding(val);
	}, reinterpret_cast<void*>(&actions));
	createTrackbar(ControlPanel_fontSzTrName, nullptr, &fontSz, Settings_MAX_FONT_SIZE,
				   [] (int val, void *userdata) {
		IControlPanelActions *pActions = reinterpret_cast<IControlPanelActions*>(userdata);
		pActions->newFontSize(val);
	}, reinterpret_cast<void*>(&actions));

	createButton(ControlPanel_restoreDefaultsLabel, [] (int, void *userdata) {
		IControlPanelActions *pActions = reinterpret_cast<IControlPanelActions*>(userdata);
		pActions->restoreUserDefaultMatchSettings();
	}, reinterpret_cast<void*>(&actions));
	createButton(ControlPanel_saveAsDefaultsLabel, [] (int, void *userdata) {
		IControlPanelActions *pActions = reinterpret_cast<IControlPanelActions*>(userdata);
		pActions->setUserDefaultMatchSettings();
	}, reinterpret_cast<void*>(&actions));

	createTrackbar(ControlPanel_hybridResultTrName, nullptr, &hybridResult, 1,
				   [] (int state, void *userdata) {
		IControlPanelActions *pActions = reinterpret_cast<IControlPanelActions*>(userdata);
		pActions->setResultMode(state != 0);
	}, reinterpret_cast<void*>(&actions));

	createTrackbar(ControlPanel_structuralSimTrName, nullptr, &structuralSim, ControlPanel_Converter_StructuralSim_maxSlider,
				   [] (int val, void *userdata) {
		IControlPanelActions *pActions = reinterpret_cast<IControlPanelActions*>(userdata);
		pActions->newStructuralSimilarityFactor(Converter::StructuralSim::fromSlider(val));
	}, reinterpret_cast<void*>(&actions));
	createTrackbar(ControlPanel_underGlyphCorrectnessTrName, nullptr, &underGlyphCorrectness, ControlPanel_Converter_Correctness_maxSlider,
				   [] (int val, void *userdata) {
		IControlPanelActions *pActions = reinterpret_cast<IControlPanelActions*>(userdata);
		pActions->newUnderGlyphCorrectnessFactor(Converter::Correctness::fromSlider(val));
	}, reinterpret_cast<void*>(&actions));
	createTrackbar(ControlPanel_glyphEdgeCorrectnessTrName, nullptr, &glyphEdgeCorrectness, ControlPanel_Converter_Correctness_maxSlider,
				   [] (int val, void *userdata) {
		IControlPanelActions *pActions = reinterpret_cast<IControlPanelActions*>(userdata);
		pActions->newGlyphEdgeCorrectnessFactor(Converter::Correctness::fromSlider(val));
	}, reinterpret_cast<void*>(&actions));
	createTrackbar(ControlPanel_asideGlyphCorrectnessTrName, nullptr, &asideGlyphCorrectness, ControlPanel_Converter_Correctness_maxSlider,
				   [] (int val, void *userdata) {
		IControlPanelActions *pActions = reinterpret_cast<IControlPanelActions*>(userdata);
		pActions->newAsideGlyphCorrectnessFactor(Converter::Correctness::fromSlider(val));
	}, reinterpret_cast<void*>(&actions));

	createTrackbar(ControlPanel_moreContrastTrName, nullptr, &moreContrast, ControlPanel_Converter_Contrast_maxSlider,
				   [] (int val, void *userdata) {
		IControlPanelActions *pActions = reinterpret_cast<IControlPanelActions*>(userdata);
		pActions->newContrastFactor(Converter::Contrast::fromSlider(val));
	}, reinterpret_cast<void*>(&actions));

	createTrackbar(ControlPanel_gravityTrName, nullptr, &gravity, ControlPanel_Converter_Gravity_maxSlider,
				   [] (int val, void *userdata) {
		IControlPanelActions *pActions = reinterpret_cast<IControlPanelActions*>(userdata);
		pActions->newGravitationalSmoothnessFactor(Converter::Gravity::fromSlider(val));
	}, reinterpret_cast<void*>(&actions));
	createTrackbar(ControlPanel_directionTrName, nullptr, &direction, ControlPanel_Converter_Direction_maxSlider,
				   [] (int val, void *userdata) {
		IControlPanelActions *pActions = reinterpret_cast<IControlPanelActions*>(userdata);
		pActions->newDirectionalSmoothnessFactor(Converter::Direction::fromSlider(val));
	}, reinterpret_cast<void*>(&actions));

	createTrackbar(ControlPanel_largerSymTrName, nullptr, &largerSym, ControlPanel_Converter_LargerSym_maxSlider,
				   [] (int val, void *userdata) {
		IControlPanelActions *pActions = reinterpret_cast<IControlPanelActions*>(userdata);
		pActions->newGlyphWeightFactor(Converter::LargerSym::fromSlider(val));
	}, reinterpret_cast<void*>(&actions));

	createTrackbar(ControlPanel_thresh4BlanksTrName, nullptr, &thresh4Blanks, Settings_MAX_THRESHOLD_FOR_BLANKS,
				   [] (int val, void *userdata) {
		IControlPanelActions *pActions = reinterpret_cast<IControlPanelActions*>(userdata);
		pActions->newThreshold4BlanksFactor(val);
	}, reinterpret_cast<void*>(&actions));

	createButton(ControlPanel_aboutLabel, [] (int, void *userdata) {
		IControlPanelActions *pActions = reinterpret_cast<IControlPanelActions*>(userdata);
		pActions->showAboutDlg(ControlPanel_aboutLabel, ControlPanel_aboutText);
	}, reinterpret_cast<void*>(&actions));
	createButton(ControlPanel_instructionsLabel, [] (int, void *userdata) {
		IControlPanelActions *pActions = reinterpret_cast<IControlPanelActions*>(userdata);
		pActions->showInstructionsDlg(ControlPanel_instructionsLabel, ControlPanel_instructionsText);
	}, reinterpret_cast<void*>(&actions));
	createButton(ControlPanel_loadSettingsLabel, [] (int, void *userdata) {
		IControlPanelActions *pActions = reinterpret_cast<IControlPanelActions*>(userdata);
		pActions->loadSettings();
	}, reinterpret_cast<void*>(&actions));
	createButton(ControlPanel_saveSettingsLabel, [] (int, void *userdata) {
		IControlPanelActions *pActions = reinterpret_cast<IControlPanelActions*>(userdata);
		pActions->saveSettings();
	}, reinterpret_cast<void*>(&actions));
}
#endif // UNIT_TESTING not defined