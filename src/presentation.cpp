/************************************************************************************************
 The application Pic2Sym approximates images by a
 grid of colored symbols with colored backgrounds.

 Copyrights from the libraries used by the program:
 - (c) 2016 Boost (www.boost.org)
		License: <http://www.boost.org/LICENSE_1_0.txt>
			or doc/licenses/Boost.lic
 - (c) 2015 OpenCV (www.opencv.org)
		License: <http://opencv.org/license.html>
            or doc/licenses/OpenCV.lic
 - (c) 2015 The FreeType Project (www.freetype.org)
		License: <http://git.savannah.gnu.org/cgit/freetype/freetype2.git/plain/docs/FTL.TXT>
	        or doc/licenses/FTL.txt
 - (c) 1997-2002 OpenMP Architecture Review Board (www.openmp.org)
   (c) Microsoft Corporation (Visual C++ implementation for OpenMP C/C++ Version 2.0 March 2002)
		See: <https://msdn.microsoft.com/en-us/library/8y6825x5(v=vs.140).aspx>
 - (c) 1995-2013 zlib software (Jean-loup Gailly and Mark Adler - see: www.zlib.net)
		License: <http://www.zlib.net/zlib_license.html>
            or doc/licenses/zlib.lic
 
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
 ***********************************************************************************************/

#include "controller.h"
#include "settings.h"
#include "jobMonitor.h"
#include "progressNotifier.h"
#include "matchParams.h"
#include "matchAssessment.h"
#include "structuralSimilarity.h"
#include "appStart.h"
#include "controlPanel.h"
#include "updateSymsActions.h"
#include "symsSerialization.h"
#include "views.h"
#include "dlgs.h"

#pragma warning ( push, 0 )

#include <Windows.h>

#include <functional>
#include <thread>

#include <boost/filesystem/operations.hpp>
#include <opencv2/highgui/highgui.hpp>

#pragma warning ( pop )

#ifndef UNIT_TESTING

#include "cmapPerspective.h"

#pragma warning ( push, 0 )

#include <numeric>

#pragma warning ( pop )

#endif // UNIT_TESTING

using namespace std;
using namespace std::chrono;
using namespace boost;
using namespace boost::filesystem;
using namespace boost::lockfree;
using namespace cv;

// UnitTesting project launches an instance of Pic2Sym to visualize any discovered issues,
// so it won't refer viewMismatches and viewMisfiltered from below.
// But it will start a Pic2Sym instance that will do such calls.
#ifndef UNIT_TESTING

void viewMismatches(const string &testTitle, const Mat &mismatches) {
	const int twiceTheRows = mismatches.rows, rows = twiceTheRows>>1, cols = mismatches.cols;
	const Mat reference = mismatches.rowRange(0, rows), // upper half is the reference
		result = mismatches.rowRange(rows, twiceTheRows); // lower half is the result

	// Comparator window size should stay within ~ 800x600
	// Enlarge up to 3 times if resulting rows < 600.
	// Enlarge also when resulted width would be less than 140 (width when the slider is visible)
	const double resizeFactor = max(140./cols, min(600./rows, 3.));

	ostringstream oss;
	oss<<"View mismatches for "<<testTitle;
	const string title(oss.str());

	Comparator comp;
	comp.setPos(0, 0);
	comp.permitResize();
	comp.resize(4+(int)ceil(cols*resizeFactor), 70+(int)ceil(rows*resizeFactor));
	comp.setTitle(title.c_str());
	comp.setStatus("Press Esc to close this window");
	comp.setReference(reference);
	comp.setResult(result, 90); // Emphasize the references 

	Controller::handleRequests();
}

void viewMisfiltered(const string &testTitle, const Mat &misfiltered) {
	const String winName = testTitle;
	namedWindow(winName);
	setWindowProperty(winName, CV_WND_PROP_AUTOSIZE, CV_WINDOW_NORMAL);
	imshow(winName, misfiltered);
	displayStatusBar(winName, "Press Esc to close this window");
	waitKey();
}

#endif // UNIT_TESTING not defined

void pauseAfterError() {
	string line;
	cout<<endl<<"Press Enter to leave"<<endl;
	getline(cin, line);
}

void showUsage() {
	cout<<"Usage:"<<endl;
	cout<<"There are 3 launch modes:"<<endl;
	cout<<"A) Normal launch mode (no parameters)"<<endl;
	cout<<"		Pic2Sym.exe"<<endl<<endl;
	cout<<"B) View mismatches launch mode (Support for Unit Testing, using 2 parameters)"<<endl;
	cout<<"		Pic2Sym.exe mismatches \"<testTitle>\""<<endl<<endl;
	cout<<"C) View misfiltered symbols launch mode (Support for Unit Testing, using 2 parameters)"<<endl;
	cout<<"		Pic2Sym.exe misfiltered \"<testTitle>\""<<endl<<endl;
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
	/// Adapter from IProgressNotifier to IGlyphsProgressTracker
	struct SymsUpdateProgressNotifier : IProgressNotifier {
		IGlyphsProgressTracker &performer;

		SymsUpdateProgressNotifier(IGlyphsProgressTracker &performer_) : performer(performer_) {}
		void operator=(const SymsUpdateProgressNotifier&) = delete;

		void notifyUser(const std::string&, double progress) override {
			performer.reportGlyphProgress(progress);
		}
	};

	/// Adapter from IProgressNotifier to IPicTransformProgressTracker
	struct PicTransformProgressNotifier : IProgressNotifier {
		IPicTransformProgressTracker &performer;

		PicTransformProgressNotifier(IPicTransformProgressTracker &performer_) : performer(performer_) {}
		void operator=(const PicTransformProgressNotifier&) = delete;

		void notifyUser(const std::string&, double progress) override {
			performer.reportTransformationProgress(progress);
		}
	};

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

	/// Displays a histogram with the distribution of the weights of the symbols from the charmap
	void viewSymWeightsHistogram(const vector<const PixMapSym> &theSyms) {
#ifndef UNIT_TESTING
		vector<double> symSums;
		for(const auto &pms : theSyms)
			symSums.push_back(pms.avgPixVal);

		static const size_t MaxBinHeight = 256ULL;
		const size_t binsCount = min(256ULL, symSums.size());
		const double smallestSum = symSums.front(), largestSum = symSums.back(),
			sumsSpan = largestSum - smallestSum;
		vector<size_t> hist(binsCount, 0U);
		const auto itBegin = symSums.cbegin();
		ptrdiff_t prevCount = 0LL;
		for(size_t bin = 0ULL; bin < binsCount; ++bin) {
			const auto it = upper_bound(CBOUNDS(symSums), smallestSum + sumsSpan*(bin+1.)/binsCount);
			const auto curCount = distance(itBegin, it);
			hist[bin] = size_t(curCount - prevCount);
			prevCount = curCount;
		}
		const double maxBinValue = (double)*max_element(CBOUNDS(hist));
		for(size_t &binValue : hist)
			binValue = (size_t)round(binValue * MaxBinHeight / maxBinValue);
		Mat histImg((int)MaxBinHeight, (int)binsCount, CV_8UC1, Scalar(255U));
		for(size_t bin = 0U; bin < binsCount; ++bin)
			if(hist[bin] > 0U)
				histImg.rowRange(int(MaxBinHeight-hist[bin]), (int)MaxBinHeight).col((int)bin) = 0U;
		imshow("histogram", histImg);
		waitKey(1);
#endif // UNIT_TESTING not defined
	}
} // anonymous namespace

string FontEngine::getFontType() {
	ostringstream oss;
	oss<<getFamily()<<'_'<<getStyle()<<'_'<<getEncoding();
	// throws logic_error if no family/style

	return oss.str();
}

string MatchEngine::getIdForSymsToUse() {
	const unsigned sz = cfg.symSettings().getFontSz();
	assert(Settings::isFontSizeOk(sz));

	ostringstream oss;
	oss<<fe.getFontType()<<'_'<<sz;

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


#ifdef MONITOR_SKIPPED_MATCHING_ASPECTS

void MatchAssessorSkip::reportSkippedAspects() const {
	cout<<endl<<"Transformation finished. Reporting skipped aspects from a total of "
		<<totalIsBetterMatchCalls<<" isBetterMatch calls:"<<endl;
	for(size_t i = 0ULL; i < enabledAspectsCount; ++i) {
		if(skippedAspects[i] == 0ULL)
			continue;
		cout<<"\t\t"<<setw(25)<<left<<enabledAspects[i]->name()
			<<" : "<<setw(10)<<right<<skippedAspects[i]<<" times"
			<<" (Complexity : "<<setw(8)<<fixed<<setprecision(3)<<right
			<<enabledAspects[i]->relativeComplexity()<<")"
			<<" ["<<setw(5)<<fixed<<setprecision(2)<<right
			<<(100. * skippedAspects[i] / totalIsBetterMatchCalls)
			<<"% of the calls]"<<endl;
	}
	cout<<endl;
}

#endif // MONITOR_SKIPPED_MATCHING_ASPECTS

extern const string Controller_PREFIX_GLYPH_PROGRESS;
extern const String ControlPanel_aboutLabel;
extern const String ControlPanel_instructionsLabel;
extern const double Transform_ProgressReportsIncrement;
extern const double SymbolsProcessing_ProgressReportsIncrement;

#pragma warning( disable : WARN_BASE_INIT_USING_THIS )
Controller::Controller(Settings &s) :
		glyphsUpdateMonitor(std::make_shared<JobMonitor>("Processing glyphs",
			std::make_shared<SymsUpdateProgressNotifier>(*this),
			SymbolsProcessing_ProgressReportsIncrement)),
		imgTransformMonitor(std::make_shared<JobMonitor>("Transforming image",
			std::make_shared<PicTransformProgressNotifier>(*this),
			Transform_ProgressReportsIncrement)),
		img(getImg()), fe(getFontEngine(s.ss).useSymsMonitor(*glyphsUpdateMonitor)), cfg(s),
		me(getMatchEngine(s).useSymsMonitor(*glyphsUpdateMonitor)),
		t(getTransformer(s).useTransformMonitor(*imgTransformMonitor)),
		pm(getPreselManager(s)),
		comp(getComparator()), cp(getControlPanel(s)) {
	me.usePreselManager(pm);
	t.usePreselManager(pm);

	comp.setPos(0, 0);
	comp.permitResize(false);

	extern const string Comparator_initial_title, Comparator_statusBar;
	comp.setTitle(Comparator_initial_title);
	comp.setStatus(Comparator_statusBar);
}
#pragma warning( default : WARN_BASE_INIT_USING_THIS )

void Controller::showAboutDlg(const string &title, const wstring &content) {
	const auto permit = cp.actionDemand(ControlPanel_aboutLabel);
	if(nullptr == permit)
		return;

	MessageBox(nullptr, content.c_str(),
			   str2wstr(title).c_str(), MB_ICONINFORMATION | MB_OK | MB_TASKMODAL | MB_SETFOREGROUND);
}

void Controller::showInstructionsDlg(const string &title, const wstring &content) {
	const auto permit = cp.actionDemand(ControlPanel_instructionsLabel);
	if(nullptr == permit)
		return;

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

bool Controller::markClustersAsNotUsed() const {
	return !me.ce.worthGrouping();
}

void Controller::showUnofficialSymDetails(unsigned symsCount) const {
	cout<<endl<<"The current charmap contains "<<symsCount<<" symbols"<<endl;

	// placing a task in the queue for the GUI updating thread
	updateSymsActionsQueue.push(new UpdateSymsAction([&, symsCount] {
		pCmi->setStatus(textForCmapStatusBar(symsCount));
	}));
}

void Controller::reportSymsUpdateDuration(double elapsed) const {
	const string cmapOverlayText = textForCmapOverlay(elapsed);
	cout<<endl<<cmapOverlayText<<endl<<endl;
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
	pCmi->setBrowsable(false);

	// Starting a thread to perform the actual change of the symbols,
	// while preserving this thread for GUI updating
	thread([&] {
		glyphsUpdateMonitor->setTasksDetails({
			.01,	// determine optimal square-fitting for the symbols
			.185,	// load & filter symbols that fit the square
			.01,	// load & filter extra-squeezed symbols
			.005,	// determine coverageOfSmallGlyphs
			.17,	// computing specific symbol-related values
			.61,	// clustering the small symbols
			.01		// reorders clusters
		}, timer);
		fe.setFontSz(cfg.ss.getFontSz());
		me.updateSymbols();

		// Symbols have been changed in the model. Only GUI must be updated
		// The GUI will display the 1st cmap page.
		// This must happen after an eventual early preview of it:
		// - we need to wait for the preview to finish if it started before this point
		// - we have to prevent an available preview to be displayed after the official version
		while(updating1stCmapPage.test_and_set())
			this_thread::sleep_for(milliseconds(1));
		updatingSymbols.clear(); // signal task completion
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
	pCmi->setBrowsable();

	extern const bool ViewSymWeightsHistogram;
	if(ViewSymWeightsHistogram)
		viewSymWeightsHistogram(fe.symsSet());
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
#pragma warning ( disable : WARN_THREAD_UNSAFE )
	static const String waitWin = "Please Wait!";
#pragma warning ( default : WARN_THREAD_UNSAFE )

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
		cout<<endl<<comparatorOverlayText <<endl<<endl;
		comp.setOverlay(comparatorOverlayText, 3000);
	}
}

const SymData* Controller::pointedSymbol(int x, int y) const {
	if(!pCmi->isBrowsable())
		return nullptr;

	const unsigned cellSide = pCmi->getCellSide(),
		r = (unsigned)y / cellSide, c = (unsigned)x / cellSide,
		symIdx = pCmi->getPageIdx()*pCmi->getSymsPerPage() + r*pCmi->getSymsPerRow() + c;

	if(symIdx >= me.getSymsCount())
		return nullptr;

	return *cmP.getSymsRange(symIdx, 1U).first;
}

void Controller::displaySymCode(unsigned long symCode) const {
	ostringstream oss;
	oss<<textForCmapStatusBar()<<" [symbol "<<symCode<<']';
	pCmi->setStatus(oss.str());
}

void Controller::enlistSymbolForInvestigation(const SymData &sd) const {
	cout<<"Appending symbol "<<sd.code<<" to the list needed for further investigations"<<endl;
	symsToInvestigate.push_back(255U - sd.negSym); // enlist actual symbol, not its negative
}

void Controller::symbolsReadyToInvestigate() const {
	if(symsToInvestigate.empty()) {
		cout<<"The list of symbols for further investigations was empty, so there's nothing to save."<<endl;
		return;
	}

	path destFile = AppStart::dir();
	if(!exists(destFile.append("SymsSelections")))
		create_directory(destFile);
	destFile.append(to_string(time(nullptr))).concat(".txt");
	cout<<"The list of "<<symsToInvestigate.size()<<" symbols for further investigations will be saved to file "
		<<destFile<<" and then cleared."<<endl;

	ut::saveSymsSelection(destFile.string(), symsToInvestigate);
	symsToInvestigate.clear();
}

CmapPerspective::VPSymDataCItPair Controller::getFontFaces(unsigned from, unsigned maxCount) const {
	return cmP.getSymsRange(from, maxCount);
}

const set<unsigned>& Controller::getClusterOffsets() const {
	return cmP.getClusterOffsets();
}

Comparator::Comparator() : CvWin("Pic2Sym") {
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
#pragma warning ( disable : WARN_THREAD_UNSAFE )
	static const double
		HEIGHT_TITLE_TOOLBAR_SLIDER_STATUS = 70,
		WIDTH_LATERAL_BORDERS = 4,
		H_NUMERATOR = 768-HEIGHT_TITLE_TOOLBAR_SLIDER_STATUS, // desired max height - HEIGHT_TITLE_TOOLBAR_SLIDER_STATUS
		W_NUMERATOR = 1024-WIDTH_LATERAL_BORDERS; // desired max width - WIDTH_LATERAL_BORDERS
#pragma warning ( default : WARN_THREAD_UNSAFE )

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
	const String CmapInspectWinName = "Charmap View";

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
	@param clusterOffsets where does each cluster start
	@param idxOfFirstSymFromPage index of the first symbol to be displayed on the page
	*/
	template<typename Iterator>
	void populateGrid(Iterator it, Iterator itEnd,
					  NegSymExtractor<Iterator> negSymExtractor,
					  Mat &content, const Mat &grid,
					  const IPresentCmap &cmapPresenter,
					  const set<unsigned> &clusterOffsets = {},
					  unsigned idxOfFirstSymFromPage = UINT_MAX) {
		content = grid.clone();
		const unsigned symsToShow = (unsigned)distance(it, itEnd);
		const int fontSz = (int)cmapPresenter.getFontSize(),
				cellSide = 1 + fontSz,
				height = CmapInspect_pageSz.height,
				width = CmapInspect_pageSz.width;

		// Place each 'negative' glyph within the grid
		for(int r = cellSide; it!=itEnd && r < height; r += cellSide) {
			Range rowRange(r - fontSz, r);
			for(int c = cellSide; it!=itEnd && c < width; c += cellSide, ++it) {
				const vector<Mat> glyphChannels(3, negSymExtractor(it));
				Mat glyphAsIfColor, region(content, rowRange, Range(c - fontSz, c));
				merge(glyphChannels, glyphAsIfColor);
				glyphAsIfColor.copyTo(region);
			}
		}

		if(clusterOffsets.empty() || UINT_MAX == idxOfFirstSymFromPage)
			return;

		// Display cluster limits if last 2 parameters provide this information
		auto showMark = [&] (unsigned offsetNewCluster, bool endMark) {
#pragma warning ( disable : WARN_THREAD_UNSAFE )
			static const Vec3b ClusterMarkColor(0U, 0U, 255U),
							ClustersEndMarkColor(128U, 0U, 64U);
#pragma warning ( default : WARN_THREAD_UNSAFE )

			const unsigned symsInArow = (unsigned)((width - 1) / cellSide);
			const div_t pos = div((int)offsetNewCluster, (int)symsInArow);
			const int r = pos.quot * cellSide + 1,
						c = pos.rem * cellSide;
			const Vec3b splitColor = endMark ? ClustersEndMarkColor : ClusterMarkColor;
			if(cmapPresenter.markClustersAsNotUsed()) { // use dashed lines as splits
				const Point up(c, r), down(c, r + fontSz - 1);
				const int dashLen = (fontSz + 4) / 5; // ceiling(fontSz/5) - at most 3 dashes with 2 breaks in between
				LineIterator lit(content, up, down, 4);
				for(int idx = 0, lim = lit.count; idx < lim; ++idx, ++lit) {
					if(0 == (1 & (idx / dashLen)))
						content.at<Vec3b>(lit.pos()) = splitColor;
				}

			} else { // use filled lines as splits
				const Mat clusterMark(fontSz, 1, CV_8UC3, splitColor);
				clusterMark.copyTo((const Mat&)content.col(c).rowRange(r, r + fontSz));
			}
		};

		const auto itBegin = clusterOffsets.cbegin();
		const unsigned firstClusterSz = *std::next(itBegin) - *itBegin;
		if(firstClusterSz < 2U) {
			// show the end mark before the 1st symbol to signal there are no non-trivial clusters
			showMark(0U, true);
			return;
		}

		auto itCo = clusterOffsets.lower_bound(idxOfFirstSymFromPage);
		assert(itCo != clusterOffsets.cend());
		if(itCo != itBegin) { // true if the required page is not the 1st one
			const unsigned prevClustSz = *itCo - *std::prev(itCo);
			if(prevClustSz < 2U)
				// When even the last cluster on the previous page was trivial,
				// this page and following ones will contain only trivial clusters.
				// So, nothing to mark, therefore just leave.
				return;
		}

		// Here the size of previous cluster is >= 2		
		for(unsigned offsetNewCluster = *itCo - idxOfFirstSymFromPage; offsetNewCluster < symsToShow; ++itCo) {
			const unsigned curClustSz = *std::next(itCo) - *itCo;
			const bool firstTrivialCluster = curClustSz < 2U;

			// mark cluster beginning or the end of non-trivial clusters
			showMark(offsetNewCluster, firstTrivialCluster);
			if(firstTrivialCluster)
				break; // stop marking trivial clusters

			offsetNewCluster += curClustSz;
		}
	}
} // anonymous namespace

void CmapInspect::populateGrid(const CmapPerspective::VPSymDataCItPair &itPair,
							   const set<unsigned> &clusterOffsets,
							   unsigned idxOfFirstSymFromPage) {
	CmapPerspective::VPSymDataCIt it = itPair.first, itEnd = itPair.second;
	::populateGrid(it, itEnd,
				   (NegSymExtractor<CmapPerspective::VPSymDataCIt>) // conversion
				   [](const CmapPerspective::VPSymDataCIt &iter) -> Mat {
						if((*iter)->removable)
							return 255U - (*iter)->negSym;
						return (*iter)->negSym;
					},
				   content, grid, cmapPresenter, clusterOffsets, idxOfFirstSymFromPage);
}

void CmapInspect::showUnofficial1stPage(vector<const Mat> &symsOn1stPage,
										atomic_flag &updating1stCmapPage,
										LockFreeQueue &updateSymsActionsQueue) {
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
		CvWin(CmapInspectWinName), cmapPresenter(cmapPresenter_) {
	content = grid = createGrid();
	extern const String CmapInspect_pageTrackName;
	createTrackbar(CmapInspect_pageTrackName, winName, &page, 1, &CmapInspect::updatePageIdx,
				   reinterpret_cast<void*>(this));
	CmapInspect::updatePageIdx(page, reinterpret_cast<void*>(this)); // mandatory

	setPos(424, 0);			// Place cmap window on x axis between 424..1064
	permitResize(false);	// Ensure the user sees the symbols exactly their size

	// Support for investigating selected symbols from one or more charmaps:
	// - mouse moves over the Charmap will display in the status bar the code of the pointed symbol
	// - Ctrl + left mouse click will append the pointed symbol to the current list
	// - left mouse double-click will save the current list and then it will clear it
	setMouseCallback(CmapInspectWinName, [] (int event, int x, int y, int flags, void* userdata) {
		const IPresentCmap *ppc = reinterpret_cast<IPresentCmap*>(userdata);
		if(event == EVENT_MOUSEMOVE) { // Mouse move
			const SymData *psd = ppc->pointedSymbol(x, y);
			if(nullptr != psd)
				ppc->displaySymCode(psd->code);
		} else if((event == EVENT_LBUTTONDBLCLK) && ((flags & EVENT_FLAG_CTRLKEY) == 0)) { // Ctrl key not pressed and left mouse double-click
			ppc->symbolsReadyToInvestigate();
		} else if((event == EVENT_LBUTTONUP) && ((flags & EVENT_FLAG_CTRLKEY) != 0)) { // Ctrl key pressed and left mouse click
			const SymData *psd = ppc->pointedSymbol(x, y);
			if(nullptr != psd)
				ppc->enlistSymbolForInvestigation(*psd);
		}
	}, reinterpret_cast<void*>(const_cast<IPresentCmap*>(&cmapPresenter)));
}

Mat CmapInspect::createGrid() {
#pragma warning ( disable : WARN_THREAD_UNSAFE )
	static const Scalar GridColor(255U, 200U, 200U);
#pragma warning ( default : WARN_THREAD_UNSAFE )

	Mat emptyGrid(CmapInspect_pageSz, CV_8UC3, Scalar::all(255U));

	cellSide = 1U + cmapPresenter.getFontSize();
	symsPerRow = (((unsigned)CmapInspect_pageSz.width - 1U) / cellSide);
	symsPerPage = symsPerRow * (((unsigned)CmapInspect_pageSz.height - 1U) / cellSide);

	// Draws horizontal & vertical cell borders
	for(int i = 0; i < CmapInspect_pageSz.width; i += cellSide)
		emptyGrid.col(i).setTo(GridColor);
	for(int i = 0; i < CmapInspect_pageSz.height; i += cellSide)
		emptyGrid.row(i).setTo(GridColor);
	return emptyGrid;
}

void CmapPerspective::reset(const VSymData &symsSet,
							const vector<vector<unsigned>> &symsIndicesPerCluster_) {
	assert(!symsSet.empty());
	assert(!symsIndicesPerCluster_.empty());

	const auto symsCount = symsSet.size(),
				clustersCount = symsIndicesPerCluster_.size();

	vector<const vector<unsigned> *> symsIndicesPerCluster(clustersCount);
	size_t clustIdx = 0ULL;
	for(const auto &clusterMembers : symsIndicesPerCluster_)
		symsIndicesPerCluster[clustIdx++] = &clusterMembers;

	// View the clusters in descending order of their size

	// Typically, there are only a few clusters larger than 1 element.
	// This partition separates the actual formed clusters from one-of-a-kind elements
	// leaving less work to perform to the sort executed afterwards.
	// Using the stable algorithm version, to preserve avgPixVal sorting
	// set by ClusterEngine::process
	auto itFirstClusterWithOneItem = stable_partition(BOUNDS(symsIndicesPerCluster),
													  [] (const vector<unsigned> *a) {
		return a->size() > 1ULL; // place actual clusters at the beginning of the vector
	});

	// Sort non-trivial clusters in descending order of their size.
	// Using the stable algorithm version, to preserve avgPixVal sorting
	// set by ClusterEngine::process
	stable_sort(begin(symsIndicesPerCluster), itFirstClusterWithOneItem,
				[] (const vector<unsigned> *a, const vector<unsigned> *b) {
		return a->size() > b->size();
	});

	pSyms.resize(symsCount);

	size_t offset = 0ULL;
	clusterOffsets.clear();
	clusterOffsets.insert((unsigned)offset);
	for(auto clusterMembers : symsIndicesPerCluster) {
		const auto prevOffset = offset;
		offset += clusterMembers->size();
		clusterOffsets.emplace_hint(end(clusterOffsets), (unsigned)offset);
		for(size_t idxPSyms = prevOffset, idxCluster = 0ULL; idxPSyms < offset; ++idxPSyms, ++idxCluster)
			pSyms[idxPSyms] = &symsSet[(size_t)(*clusterMembers)[idxCluster]];
	}
}

CmapPerspective::VPSymDataCItPair CmapPerspective::getSymsRange(unsigned from, unsigned count) const {
	const auto sz = pSyms.size();
	const VPSymDataCIt itEnd = pSyms.cend();
	if((size_t)from >= sz)
		return make_pair(itEnd, itEnd);

	const VPSymDataCIt itStart = next(pSyms.cbegin(), from);
	const auto maxCount = sz - (size_t)from;
	if((size_t)count >= maxCount)
		return make_pair(itStart, itEnd);

	return make_pair(itStart, next(itStart, count));
}

const set<unsigned>& CmapPerspective::getClusterOffsets() const {
	return clusterOffsets;
}

extern const wstring ControlPanel_aboutText;
extern const wstring ControlPanel_instructionsText;
extern const unsigned SymsBatch_defaultSz;

ControlPanel::ControlPanel(IControlPanelActions &performer_, const Settings &cfg_) :
		performer(performer_), cfg(cfg_),
		maxHSyms((int)cfg_.imgSettings().getMaxHSyms()), maxVSyms((int)cfg_.imgSettings().getMaxVSyms()),
		encoding(0), fontSz((int)cfg_.symSettings().getFontSz()),
		symsBatchSz((int)SymsBatch_defaultSz),
		hybridResult(cfg_.matchSettings().isHybridResult() ? 1 : 0),
		structuralSim(Converter::StructuralSim::toSlider(cfg_.matchSettings().get_kSsim())),
		underGlyphCorrectness(Converter::Correctness::toSlider(cfg_.matchSettings().get_kSdevFg())),
		glyphEdgeCorrectness(Converter::Correctness::toSlider(cfg_.matchSettings().get_kSdevEdge())),
		asideGlyphCorrectness(Converter::Correctness::toSlider(cfg_.matchSettings().get_kSdevBg())),
		moreContrast(Converter::Contrast::toSlider(cfg_.matchSettings().get_kContrast())),
		gravity(Converter::Gravity::toSlider(cfg_.matchSettings().get_kMCsOffset())),
		direction(Converter::Direction::toSlider(cfg_.matchSettings().get_kCosAngleMCs())),
		largerSym(Converter::LargerSym::toSlider(cfg_.matchSettings().get_kSymDensity())),
		thresh4Blanks((int)cfg_.matchSettings().getBlankThreshold()) {
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
#pragma warning ( disable : WARN_THREAD_UNSAFE )
		static ImgSelector is;
#pragma warning ( default : WARN_THREAD_UNSAFE )

		if(is.promptForUserChoice())
			pActions->newImage(is.selection());
	}, reinterpret_cast<void*>(&performer));
	createButton(ControlPanel_transformImgLabel,
				 [] (int, void *userdata) {
		IControlPanelActions *pActions = reinterpret_cast<IControlPanelActions*>(userdata);
		pActions->performTransformation();
	}, reinterpret_cast<void*>(&performer));

	createTrackbar(ControlPanel_outWTrName, nullptr, &maxHSyms, (int)Settings_MAX_H_SYMS,
				   [] (int val, void *userdata) {
		IControlPanelActions *pActions = reinterpret_cast<IControlPanelActions*>(userdata);
		pActions->newHmaxSyms(val);
	}, reinterpret_cast<void*>(&performer));
	createTrackbar(ControlPanel_outHTrName, nullptr, &maxVSyms, (int)Settings_MAX_V_SYMS,
				   [] (int val, void *userdata) {
		IControlPanelActions *pActions = reinterpret_cast<IControlPanelActions*>(userdata);
		pActions->newVmaxSyms(val);
	}, reinterpret_cast<void*>(&performer));

	createButton(ControlPanel_selectFontLabel,
				 [] (int, void *userdata) {
		IControlPanelActions *pActions = reinterpret_cast<IControlPanelActions*>(userdata);

#pragma warning ( disable : WARN_THREAD_UNSAFE )
		static SelectFont sf;
#pragma warning ( default : WARN_THREAD_UNSAFE )

		if(sf.promptForUserChoice())
			pActions->newFontFamily(sf.selection());
	}, reinterpret_cast<void*>(&performer));

	createTrackbar(ControlPanel_encodingTrName, nullptr, &encoding, 1,
				   [] (int val, void *userdata) {
		IControlPanelActions *pActions = reinterpret_cast<IControlPanelActions*>(userdata);
		pActions->newFontEncoding(val);
	}, reinterpret_cast<void*>(&performer));
	createTrackbar(ControlPanel_fontSzTrName, nullptr, &fontSz, (int)Settings_MAX_FONT_SIZE,
				   [] (int val, void *userdata) {
		IControlPanelActions *pActions = reinterpret_cast<IControlPanelActions*>(userdata);
		pActions->newFontSize(val);
	}, reinterpret_cast<void*>(&performer));

	createTrackbar(ControlPanel_symsBatchSzTrName, nullptr, &symsBatchSz, (int)SymsBatch_trackMax,
				   [] (int val, void *userdata) {
		IControlPanelActions *pActions = reinterpret_cast<IControlPanelActions*>(userdata);
		pActions->newSymsBatchSize(val);
	}, reinterpret_cast<void*>(&performer));

	createButton(ControlPanel_restoreDefaultsLabel, [] (int, void *userdata) {
		IControlPanelActions *pActions = reinterpret_cast<IControlPanelActions*>(userdata);
		pActions->restoreUserDefaultMatchSettings();
	}, reinterpret_cast<void*>(&performer));
	createButton(ControlPanel_saveAsDefaultsLabel, [] (int, void *userdata) {
		IControlPanelActions *pActions = reinterpret_cast<IControlPanelActions*>(userdata);
		pActions->setUserDefaultMatchSettings();
	}, reinterpret_cast<void*>(&performer));

	createTrackbar(ControlPanel_hybridResultTrName, nullptr, &hybridResult, 1,
				   [] (int state, void *userdata) {
		IControlPanelActions *pActions = reinterpret_cast<IControlPanelActions*>(userdata);
		pActions->setResultMode(state != 0);
	}, reinterpret_cast<void*>(&performer));

	createTrackbar(ControlPanel_structuralSimTrName, nullptr, &structuralSim, ControlPanel_Converter_StructuralSim_maxSlider,
				   [] (int val, void *userdata) {
		IControlPanelActions *pActions = reinterpret_cast<IControlPanelActions*>(userdata);
		pActions->newStructuralSimilarityFactor(Converter::StructuralSim::fromSlider(val));
	}, reinterpret_cast<void*>(&performer));
	createTrackbar(ControlPanel_underGlyphCorrectnessTrName, nullptr, &underGlyphCorrectness, ControlPanel_Converter_Correctness_maxSlider,
				   [] (int val, void *userdata) {
		IControlPanelActions *pActions = reinterpret_cast<IControlPanelActions*>(userdata);
		pActions->newUnderGlyphCorrectnessFactor(Converter::Correctness::fromSlider(val));
	}, reinterpret_cast<void*>(&performer));
	createTrackbar(ControlPanel_glyphEdgeCorrectnessTrName, nullptr, &glyphEdgeCorrectness, ControlPanel_Converter_Correctness_maxSlider,
				   [] (int val, void *userdata) {
		IControlPanelActions *pActions = reinterpret_cast<IControlPanelActions*>(userdata);
		pActions->newGlyphEdgeCorrectnessFactor(Converter::Correctness::fromSlider(val));
	}, reinterpret_cast<void*>(&performer));
	createTrackbar(ControlPanel_asideGlyphCorrectnessTrName, nullptr, &asideGlyphCorrectness, ControlPanel_Converter_Correctness_maxSlider,
				   [] (int val, void *userdata) {
		IControlPanelActions *pActions = reinterpret_cast<IControlPanelActions*>(userdata);
		pActions->newAsideGlyphCorrectnessFactor(Converter::Correctness::fromSlider(val));
	}, reinterpret_cast<void*>(&performer));

	createTrackbar(ControlPanel_moreContrastTrName, nullptr, &moreContrast, ControlPanel_Converter_Contrast_maxSlider,
				   [] (int val, void *userdata) {
		IControlPanelActions *pActions = reinterpret_cast<IControlPanelActions*>(userdata);
		pActions->newContrastFactor(Converter::Contrast::fromSlider(val));
	}, reinterpret_cast<void*>(&performer));

	createTrackbar(ControlPanel_gravityTrName, nullptr, &gravity, ControlPanel_Converter_Gravity_maxSlider,
				   [] (int val, void *userdata) {
		IControlPanelActions *pActions = reinterpret_cast<IControlPanelActions*>(userdata);
		pActions->newGravitationalSmoothnessFactor(Converter::Gravity::fromSlider(val));
	}, reinterpret_cast<void*>(&performer));
	createTrackbar(ControlPanel_directionTrName, nullptr, &direction, ControlPanel_Converter_Direction_maxSlider,
				   [] (int val, void *userdata) {
		IControlPanelActions *pActions = reinterpret_cast<IControlPanelActions*>(userdata);
		pActions->newDirectionalSmoothnessFactor(Converter::Direction::fromSlider(val));
	}, reinterpret_cast<void*>(&performer));

	createTrackbar(ControlPanel_largerSymTrName, nullptr, &largerSym, ControlPanel_Converter_LargerSym_maxSlider,
				   [] (int val, void *userdata) {
		IControlPanelActions *pActions = reinterpret_cast<IControlPanelActions*>(userdata);
		pActions->newGlyphWeightFactor(Converter::LargerSym::fromSlider(val));
	}, reinterpret_cast<void*>(&performer));

	createTrackbar(ControlPanel_thresh4BlanksTrName, nullptr, &thresh4Blanks, (int)Settings_MAX_THRESHOLD_FOR_BLANKS,
				   [] (int val, void *userdata) {
		IControlPanelActions *pActions = reinterpret_cast<IControlPanelActions*>(userdata);
		pActions->newThreshold4BlanksFactor((unsigned)val);
	}, reinterpret_cast<void*>(&performer));

	createButton(ControlPanel_aboutLabel, [] (int, void *userdata) {
		IControlPanelActions *pActions = reinterpret_cast<IControlPanelActions*>(userdata);
		pActions->showAboutDlg(ControlPanel_aboutLabel, ControlPanel_aboutText);
	}, reinterpret_cast<void*>(&performer));
	createButton(ControlPanel_instructionsLabel, [] (int, void *userdata) {
		IControlPanelActions *pActions = reinterpret_cast<IControlPanelActions*>(userdata);
		pActions->showInstructionsDlg(ControlPanel_instructionsLabel, ControlPanel_instructionsText);
	}, reinterpret_cast<void*>(&performer));
	createButton(ControlPanel_loadSettingsLabel, [] (int, void *userdata) {
		IControlPanelActions *pActions = reinterpret_cast<IControlPanelActions*>(userdata);
		pActions->loadSettings();
	}, reinterpret_cast<void*>(&performer));
	createButton(ControlPanel_saveSettingsLabel, [] (int, void *userdata) {
		IControlPanelActions *pActions = reinterpret_cast<IControlPanelActions*>(userdata);
		pActions->saveSettings();
	}, reinterpret_cast<void*>(&performer));
}

#endif // UNIT_TESTING not defined