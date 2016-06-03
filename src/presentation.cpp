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
#include "settings.h"
#include "matchParams.h"
#include "controlPanel.h"
#include "updateSymsActions.h"
#include "views.h"
#include "dlgs.h"
#include "misc.h"

#include <iomanip>
#include <functional>
#include <thread>

#include <opencv2/highgui.hpp>

using namespace std;
using namespace std::chrono;
using namespace boost;
using namespace boost::lockfree;
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

namespace {
	/// Common realization of IUpdateSymsAction
	struct UpdateSymsAction : IUpdateSymsAction {
	protected:
		std::function<void()> fn; ///< the function to be called by perform, that has access to private fields & methods

	public:
		/// Creating an action object that performs the tasks described in fn_
		UpdateSymsAction(std::function<void()> fn_) : fn(fn_) {}

		void perform() override {
			fn();
		}
	};
}

extern const string Controller_PREFIX_GLYPH_PROGRESS;

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

const string Controller::textForCmapStatusBar(unsigned upperSymsCount/* = 0U*/) const {
	ostringstream oss;
	oss<<"Font type '"<<fe.getFamily()<<' '<<fe.getStyle()
		<<"', size "<<cfg.ss.getFontSz()<<", encoding '"<<fe.getEncoding()<<'\''
		<<" : "<<((upperSymsCount != 0U) ? upperSymsCount : (unsigned)fe.symsSet().size())<<" symbols";
	return oss.str();
}

const string Controller::textHourGlass(const std::string &prefix, double progress) const {
	ostringstream oss;
	oss<<prefix<<" ("<<fixed<<setprecision(0)<<progress*100.<<"%)";
	return oss.str();
}

void Controller::resetCmapView() {
	if(pCmi)
		pCmi->reset();
}

MatchEngine::VSymDataCItPair Controller::getFontFaces(unsigned from, unsigned maxCount) const {
	return me.getSymsRange(from, maxCount);
}

void Controller::showUnofficialSymDetails(unsigned symsCount) const {
	cout<<"The current charmap contains "<<symsCount<<" symbols"<<endl;

	// placing a task in the queue for the GUI updating thread
	updateSymsActionsQueue.push(new UpdateSymsAction([&, symsCount] {
		pCmi->setStatus(textForCmapStatusBar(symsCount));
	}));
}

void Controller::reportSymsUpdateDuration(double elapsed) const {
	const string cmapOverlayText = textForCmapOverlay(elapsed);
	cout<<cmapOverlayText<<endl<<endl;
	pCmi->setOverlay(cmapOverlayText, 3000);
}

void Controller::symbolsChanged() {
	// Timing the update.
	// timer's destructor will report duration and will close the hourglass window.
	// These actions will be launched in this GUI updating thread,
	// after processing all previously registered actions.
	Timer timer = createTimerForGlyphs();

	updatingSymbols.test_and_set();
	updating1stCmapPage.clear();

	// Starting a thread to perform the actual change of the symbols,
	// while preserving this thread for GUI updating
	thread([&] {
		fe.setFontSz(cfg.ss.getFontSz());
		me.updateSymbols();

		// Symbols have been changed in the model. Only GUI must be updated
		// The GUI will display the 1st cmap page.
		// This must happen after an eventual early preview of it:
		// - we need to wait for the preview to finish if it started before this point
		// - we have to prevent an available preview to be displayed after the official version
		while(updating1stCmapPage.test_and_set())
			this_thread::sleep_for(milliseconds(1));
		updatingSymbols.clear(); // signal that the work has finished
	}).detach(); // termination captured by updatingSymbols flag

	IUpdateSymsAction *evt = nullptr;
	auto performRegisteredActions = [&] { // lambda used twice below
		while(updateSymsActionsQueue.pop(evt)) { // perform available actions
			evt->perform();
			delete evt;
			waitKey(1);
		}
	};

	while(updatingSymbols.test_and_set()) // loop while work is carried out
		performRegisteredActions();
	performRegisteredActions(); // perform any remaining actions

	// Official versions of the status bar and the 1st cmap page
	pCmi->setStatus(textForCmapStatusBar());
	pCmi->updatePagesCount((unsigned)fe.symsSet().size());
}

void Controller::display1stPageIfFull(const vector<const PixMapSym> &syms) {
	if((unsigned)syms.size() != pCmi->getSymsPerPage())
		return;

	// Starting the thread that builds the 'pre-release' version of the 1st cmap page
	thread([&, syms] {
		vector<const Mat> matSyms;
		const auto fontSz = getFontSize();
		for(const auto &pms : syms)
			matSyms.emplace_back(pms.toMat(fontSz, true));
		
		const_cast<vector<const PixMapSym>&>(syms).clear(); // discard local copy of the vector

		pCmi->showUnofficial1stPage(matSyms, updating1stCmapPage, updateSymsActionsQueue);
	}).detach(); // termination doesn't matter
}

Timer Controller::createTimerForGlyphs() const {
	return Timer(std::make_shared<Controller::TimerActions_SymSetUpdate>(*this)); // RVO
}

Timer Controller::createTimerForImgTransform() const {
	return Timer(std::make_shared<Controller::TimerActions_ImgTransform>(*this)); // RVO
}

Controller::TimerActions_Controller::TimerActions_Controller(const Controller &ctrler_) :
		ctrler(ctrler_) {}

Controller::TimerActions_SymSetUpdate::TimerActions_SymSetUpdate(const Controller &ctrler_) :
		TimerActions_Controller(ctrler_) {}

void Controller::TimerActions_SymSetUpdate::onStart() {
	ctrler.reportGlyphProgress(0.);
}

void Controller::TimerActions_SymSetUpdate::onRelease(double elapsedS) {
	ctrler.updateSymsDone(elapsedS);
}

Controller::TimerActions_ImgTransform::TimerActions_ImgTransform(const Controller &ctrler_) :
		TimerActions_Controller(ctrler_) {}

void Controller::TimerActions_ImgTransform::onStart() {
	ctrler.reportTransformationProgress(0.);
}

void Controller::TimerActions_ImgTransform::onRelease(double elapsedS) {
	ctrler.reportTransformationProgress(1.);
	ctrler.presentTransformationResults(elapsedS);
}

void Controller::TimerActions_ImgTransform::onCancel(const string &reason/* = ""*/) {
	ctrler.reportTransformationProgress(1., true);
	infoMsg(reason);
}

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
		namedWindow(waitWin, CV_GUI_NORMAL); // no status bar, nor toolbar
		moveWindow(waitWin, 0, 400);
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
		const string hourGlassText = textHourGlass(oss.str(), progress);
		setWindowTitle(waitWin, hourGlassText);
	}
}

void Controller::reportGlyphProgress(double progress) const {
	updateSymsActionsQueue.push(new UpdateSymsAction([&, progress] {
		hourGlass(progress, Controller_PREFIX_GLYPH_PROGRESS);
	}));
}

void Controller::updateSymsDone(double durationS) const {
	hourGlass(1., Controller_PREFIX_GLYPH_PROGRESS);
	reportSymsUpdateDuration(durationS);
}

void Controller::reportTransformationProgress(double progress, bool showDraft/* = false*/) const {
	extern const string Controller_PREFIX_TRANSFORMATION_PROGRESS;
	hourGlass(progress, Controller_PREFIX_TRANSFORMATION_PROGRESS);

	if(showDraft)
		presentTransformationResults();
}

void Controller::presentTransformationResults(double completionDurationS/* = -1.*/) const {
	comp.setResult(t.getResult()); // display the result at the end of the transformation
	if(completionDurationS > 0.) {
		const string comparatorOverlayText = textForComparatorOverlay(completionDurationS);
		cout<<comparatorOverlayText <<endl<<endl;
		comp.setOverlay(comparatorOverlayText, 3000);
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

void Transformer::updateStudiedCase(int rows, int cols) {
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
	studiedCase = oss.str(); // this text is included in the result & trace file names
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

namespace {

	/// type of a function extracting the negative mask from an iterator
	template<typename Iterator>
	using NegSymExtractor = std::function<const Mat(const typename Iterator&)>;

	/**
	Creates a page from the cmap to be displayed within the Cmap View.
	@param it iterator to the first element to appear on the page
	@param itEnd iterator after the last element to appear on the page
	@param negSymExtractor function extracting the negative mask from each iterator
	@param content the resulted page as a matrix (image)
	@param grid the 'hive' for the glyphs to be displayed
	@param cmapPresenter provides the font size
	*/
	template<typename Iterator>
	void populateGrid(Iterator it, Iterator itEnd,
					  NegSymExtractor<Iterator> negSymExtractor,
					  Mat &content, const Mat &grid, const IPresentCmap &cmapPresenter) {
		content = grid.clone();
		const int fontSz = cmapPresenter.getFontSize(),
			fontSzM1 = fontSz - 1,
			cellSide = 1 + fontSz,
			height = CmapInspect_pageSz.height,
			width = CmapInspect_pageSz.width;

		// Place each 'negative' glyph within the grid
		for(int r = fontSzM1; it!=itEnd && r < height; r += cellSide) {
			Range rowRange(r - fontSzM1, r + 1);
			for(int c = fontSzM1; it!=itEnd && c < width; c += cellSide, ++it) {
				Mat region(content, rowRange, Range(c - fontSzM1, c + 1));
				vector<Mat> glyphChannels(3, negSymExtractor(it));
				Mat glyphAsIfColor;
				merge(glyphChannels, glyphAsIfColor);
				glyphAsIfColor.copyTo(region);
			}
		}
	}
} // anonymous namespace

void CmapInspect::populateGrid(const MatchEngine::VSymDataCItPair &itPair) {
	MatchEngine::VSymDataCIt it = itPair.first, itEnd = itPair.second;
	::populateGrid(it, itEnd,
				   (NegSymExtractor<MatchEngine::VSymDataCIt>) // conversion
				   [](const MatchEngine::VSymDataCIt &iter) { return iter->symAndMasks[SymData::NEG_SYM_IDX]; },
				   content, grid, cmapPresenter);
}

void CmapInspect::showUnofficial1stPage(vector<const Mat> &symsOn1stPage,
										atomic_flag &updating1stCmapPage,
										LockFreeQueueSz22 &updateSymsActionsQueue) {
	std::shared_ptr<Mat> unofficial = std::make_shared<Mat>();
	::populateGrid(CBOUNDS(symsOn1stPage),
				   (NegSymExtractor<vector<const Mat>::const_iterator>) // conversion
				   [](const vector<const Mat>::const_iterator &iter) { return *iter; },
				   *unofficial, grid, cmapPresenter);

	symsOn1stPage.clear(); // discard values now

	// If the main worker didn't register its intention to render already the official 1st cmap page
	// display the unofficial early version.
	// Otherwise just leave
	if(!updating1stCmapPage.test_and_set()) { // holds the value on true after acquiring it
		// Creating local copies that can be passed by value to Unofficial1stPageCmap's parameter
		auto *pUpdating1stCmapPage = &updating1stCmapPage;
		const auto winNameCopy = winName;

		updateSymsActionsQueue.push(new UpdateSymsAction([pUpdating1stCmapPage, winNameCopy, unofficial] {
			imshow(winNameCopy, *unofficial);

			pUpdating1stCmapPage->clear(); // puts its value on false, to be acquired by the official version publisher
		}));
	}
}

CmapInspect::CmapInspect(const IPresentCmap &cmapPresenter_) :
		CvWin("Charmap View"), cmapPresenter(cmapPresenter_), grid(content = createGrid()) {
	reset();

	extern const String CmapInspect_pageTrackName;
	createTrackbar(CmapInspect_pageTrackName, winName, &page, 1, &CmapInspect::updatePageIdx,
				   reinterpret_cast<void*>(this));
	CmapInspect::updatePageIdx(page, reinterpret_cast<void*>(this)); // mandatory

	setPos(424, 0);			// Place cmap window on x axis between 424..1064
	permitResize(false);	// Ensure the user sees the symbols exactly their size
}

void CmapInspect::reset() {
	symsPerPage = computeSymsPerPage();
}

Mat CmapInspect::createGrid() const {
	Mat result(CmapInspect_pageSz, CV_8UC3, Scalar::all(255U));

	// Draws horizontal & vertical cell borders
	const int cellSide = 1+cmapPresenter.getFontSize();
	for(int i = cellSide-1; i < CmapInspect_pageSz.width; i += cellSide)
		line(result, Point(i, 0), Point(i, CmapInspect_pageSz.height-1), Scalar(255U, 200U, 200U));
	for(int i = cellSide-1; i < CmapInspect_pageSz.height; i += cellSide)
		line(result, Point(0, i), Point(CmapInspect_pageSz.width-1, i), Scalar(255U, 200U, 200U));
	return result;
}

extern const wstring ControlPanel_aboutText;
extern const wstring ControlPanel_instructionsText;
extern const String ControlPanel_aboutLabel;
extern const String ControlPanel_instructionsLabel;
extern const unsigned SymsBatch_defaultSz;

ControlPanel::ControlPanel(IControlPanelActions &actions_, const Settings &cfg) :
		actions(actions_),
		maxHSyms(cfg.imgSettings().getMaxHSyms()), maxVSyms(cfg.imgSettings().getMaxVSyms()),
		encoding(0U), fontSz(cfg.symSettings().getFontSz()),
		symsBatchSz((int)SymsBatch_defaultSz),
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
	extern const unsigned SymsBatch_trackMax;
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
	extern const String ControlPanel_symsBatchSzTrName;
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

	createTrackbar(ControlPanel_outWTrName, nullptr, &maxHSyms, (int)Settings_MAX_H_SYMS,
				   [] (int val, void *userdata) {
		IControlPanelActions *pActions = reinterpret_cast<IControlPanelActions*>(userdata);
		pActions->newHmaxSyms(val);
	}, reinterpret_cast<void*>(&actions));
	createTrackbar(ControlPanel_outHTrName, nullptr, &maxVSyms, (int)Settings_MAX_V_SYMS,
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
	createTrackbar(ControlPanel_fontSzTrName, nullptr, &fontSz, (int)Settings_MAX_FONT_SIZE,
				   [] (int val, void *userdata) {
		IControlPanelActions *pActions = reinterpret_cast<IControlPanelActions*>(userdata);
		pActions->newFontSize(val);
	}, reinterpret_cast<void*>(&actions));

	createTrackbar(ControlPanel_symsBatchSzTrName, nullptr, &symsBatchSz, (int)SymsBatch_trackMax,
				   [] (int val, void *userdata) {
		IControlPanelActions *pActions = reinterpret_cast<IControlPanelActions*>(userdata);
		pActions->newSymsBatchSize(val);
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

	createTrackbar(ControlPanel_thresh4BlanksTrName, nullptr, &thresh4Blanks, (int)Settings_MAX_THRESHOLD_FOR_BLANKS,
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