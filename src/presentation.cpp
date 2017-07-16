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
 
 (c) 2016, 2017 Florin Tulba <florintulba@yahoo.com>

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
#include "fontEngine.h"
#include "transform.h"
#include "updateSymSettings.h"
#include "glyphsProgressTracker.h"
#include "picTransformProgressTracker.h"
#include "selectSymbols.h"
#include "controlPanelActions.h"
#include "settingsBase.h"
#include "symSettings.h"
#include "imgSettings.h"
#include "jobMonitor.h"
#include "progressNotifier.h"
#include "matchParams.h"
#include "matchAssessment.h"
#include "structuralSimilarity.h"
#include "controlPanel.h"
#include "controlPanelActionsBase.h"
#include "updateSymsActions.h"
#include "views.h"
#include "presentCmap.h"
#include "dlgs.h"

#pragma warning ( push, 0 )

#include <thread>

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
	cout<<"B) Timing a certain scenario (5 parameters)"<<endl;
	cout<<"		Pic2Sym.exe timing \"<caseName>\" \"<settingsPath>\" \"<imagePath>\" \"<reportFilePath>\""<<endl<<endl;
	cout<<"C) View mismatches launch mode (Support for Unit Testing, using 2 parameters)"<<endl;
	cout<<"		Pic2Sym.exe mismatches \"<testTitle>\""<<endl<<endl;
	cout<<"D) View misfiltered symbols launch mode (Support for Unit Testing, using 2 parameters)"<<endl;
	cout<<"		Pic2Sym.exe misfiltered \"<testTitle>\""<<endl<<endl;
	pauseAfterError();
}

extern const string Controller_PREFIX_GLYPH_PROGRESS;

namespace {
	/// Adapter from IProgressNotifier to IGlyphsProgressTracker
	struct SymsUpdateProgressNotifier : IProgressNotifier {
		const IController &performer;

		SymsUpdateProgressNotifier(const IController &performer_) : performer(performer_) {}
		void operator=(const SymsUpdateProgressNotifier&) = delete;

		void notifyUser(const std::string&, double progress) override {
			performer.hourGlass(progress, Controller_PREFIX_GLYPH_PROGRESS, true); // async call
		}
	};

	/// Adapter from IProgressNotifier to IPicTransformProgressTracker
	struct PicTransformProgressNotifier : IProgressNotifier {
		std::shared_ptr<const IPicTransformProgressTracker> performer;

		PicTransformProgressNotifier(std::shared_ptr<const IPicTransformProgressTracker>performer_) : performer(performer_) {}
		void operator=(const PicTransformProgressNotifier&) = delete;

		void notifyUser(const std::string&, double progress) override {
			performer->reportTransformationProgress(progress);
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

	return oss.str();
}

string MatchEngine::getIdForSymsToUse() {
	const unsigned sz = cfg.getSS().getFontSz();
	assert(ISettings::isFontSizeOk(sz));

	ostringstream oss;
	oss<<fe.getFontType()<<'_'<<sz;

	return oss.str();
}

void Transformer::updateStudiedCase(int rows, int cols) {
	const auto &ss = cfg.getMS();
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

extern const String ControlPanel_aboutLabel;
extern const String ControlPanel_instructionsLabel;
extern const double Transform_ProgressReportsIncrement;
extern const double SymbolsProcessing_ProgressReportsIncrement;

#pragma warning( disable : WARN_BASE_INIT_USING_THIS )
Controller::Controller(ISettingsRW &s) :
		updateSymSettings(std::make_shared<const UpdateSymSettings>(s.SS())),
		glyphsProgressTracker(std::make_shared<const GlyphsProgressTracker>(*this)),
		picTransformProgressTracker(std::make_shared<PicTransformProgressTracker>(*this)),
		glyphsUpdateMonitor(std::make_shared<JobMonitor>("Processing glyphs",
			std::make_shared<SymsUpdateProgressNotifier>(*this),
			SymbolsProcessing_ProgressReportsIncrement)),
		imgTransformMonitor(std::make_shared<JobMonitor>("Transforming image",
			std::make_shared<PicTransformProgressNotifier>(getPicTransformProgressTracker()),
			Transform_ProgressReportsIncrement)),
		cmP(),
		presentCmap(std::make_shared<const PresentCmap>(*this, cmP)),
		fe(getFontEngine(s.getSS()).useSymsMonitor(*glyphsUpdateMonitor)), cfg(s),
		me(getMatchEngine(s).useSymsMonitor(*glyphsUpdateMonitor)),
		t(getTransformer(s).useTransformMonitor(*imgTransformMonitor)),
		pm(getPreselManager(s)),
		comp(getComparator()),
		pCmi(),
		selectSymbols(std::make_shared<const SelectSymbols>(*this, getMatchEngine(s), cmP, pCmi)),
		controlPanelActions(std::make_shared<ControlPanelActions>(*this, s,
			getFontEngine(s.getSS()), getMatchEngine(s).assessor(), getTransformer(s), getComparator(), pCmi)) {
	me.usePreselManager(pm);
	t.usePreselManager(pm);
	const_cast<IPresentCmap*>(presentCmap.get())->markClustersAsUsed(&me.isClusteringUseful());

	comp.setPos(0, 0);
	comp.permitResize(false);

	extern const string Comparator_initial_title, Comparator_statusBar;
	comp.setTitle(Comparator_initial_title);
	comp.setStatus(Comparator_statusBar);
}
#pragma warning( default : WARN_BASE_INIT_USING_THIS )

const string Controller::textForCmapStatusBar(unsigned upperSymsCount/* = 0U*/) const {
	assert(nullptr != fe.getFamily() && 0ULL < strlen(fe.getFamily()));
	assert(nullptr != fe.getStyle() && 0ULL < strlen(fe.getStyle()));
	assert(!fe.getEncoding().empty());
	ostringstream oss;
	oss<<"Font type '"<<fe.getFamily()<<' '<<fe.getStyle()
		<<"', size "<<cfg.getSS().getFontSz()<<", encoding '"<<fe.getEncoding()<<'\''
		<<" : "<<((upperSymsCount != 0U) ? upperSymsCount : (unsigned)fe.symsSet().size())<<" symbols";
	return oss.str();
}

const string Controller::textHourGlass(const std::string &prefix, double progress) const {
	ostringstream oss;
	oss<<prefix<<" ("<<fixed<<setprecision(0)<<progress*100.<<"%)";
	return oss.str();
}

void Controller::symbolsChanged() {
	// Timing the update.
	// timer's destructor will report duration and will close the hourglass window.
	// These actions will be launched in this GUI updating thread,
	// after processing all previously registered actions.
	Timer timer = glyphsProgressTracker->createTimerForGlyphs();

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

		try {
			fe.setFontSz(cfg.getSS().getFontSz());
			me.updateSymbols();

		} catch(NormalSymsLoadingFailure &excObj) { // capture it in one thread, then pack it for the other thread
			const string errText(excObj.what());
			// An exception with the same errText will be thrown in the main thread when executing next action
			updateSymsActionsQueue.push(new BasicUpdateSymsAction([errText] {
				throw NormalSymsLoadingFailure(errText);
			}));

			updatingSymbols.clear(); // signal task completion
			return;

		} catch(TinySymsLoadingFailure &excObj) { // capture it in one thread, then pack it for the other thread
			const string errText(excObj.what());
			// An exception with the same errText will be thrown in the main thread when executing next action
			updateSymsActionsQueue.push(new BasicUpdateSymsAction([errText] {
				throw TinySymsLoadingFailure(errText);
			}));

			updatingSymbols.clear(); // signal task completion
			return;
		}

		// Symbols have been changed in the model. Only GUI must be updated
		// The GUI will display the 1st cmap page.
		// This must happen after an eventual early preview of it:
		// - we need to wait for the preview to finish if it started before this point
		// - we have to prevent an available preview to be displayed after the official version
		while(updating1stCmapPage.test_and_set())
			this_thread::sleep_for(milliseconds(1));
		updatingSymbols.clear(); // signal task completion
	}).detach(); // termination captured by updatingSymbols flag

	IUpdateSymsAction *action = nullptr;
	auto performRegisteredActions = [&] { // lambda used twice below
		while(updateSymsActionsQueue.pop(action)) { // perform available actions
#pragma warning ( disable : WARN_SEH_NOT_CAUGHT )
			try {
				action->perform();
			} catch(...) { // making sure action gets deleted even when exceptions are thrown
				delete action;
				
				// Discarding all remaining actions
				while(updateSymsActionsQueue.pop(action))
					delete action;

				// Postponing the processing of the exception
				// See above NormalSymsLoadingFailure & TinySymsLoadingFailure being thrown
				throw;
			}
#pragma warning ( default : WARN_SEH_NOT_CAUGHT )

			delete action;
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

void Controller::hourGlass(double progress, const string &title/* = ""*/, bool async/* = false*/) const {
#pragma warning ( disable : WARN_THREAD_UNSAFE )
	static const String waitWin = "Please Wait!";
#pragma warning ( default : WARN_THREAD_UNSAFE )

	if(async) { // enqueue the call
		updateSymsActionsQueue.push(new BasicUpdateSymsAction([&, progress] {
			hourGlass(progress, title);
		}));

	} else { // direct call or one async call due now
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
}

void Controller::updateStatusBarCmapInspect(unsigned upperSymsCount/* = 0U*/,
											const string &suffix/* = ""*/,
											bool async/* = false*/) const {
	const string newStatusBarText(textForCmapStatusBar(upperSymsCount) + suffix);
	if(async) { // placing a task in the queue for the GUI updating thread
		updateSymsActionsQueue.push(new BasicUpdateSymsAction([&, newStatusBarText] {
			pCmi->setStatus(newStatusBarText);
		}));
	} else { // direct call or one async call due now
		pCmi->setStatus(newStatusBarText);
	}
}

void Controller::reportDuration(const std::string &text, double durationS) const {
	ostringstream oss;
	oss<<text<<' '<<durationS<<" s!";
	const string cmapOverlayText(oss.str());
	cout<<endl<<cmapOverlayText<<endl<<endl;
	pCmi->setOverlay(cmapOverlayText, 3000);
}

bool Controller::updateResizedImg(std::shared_ptr<const ResizedImg> resizedImg_) {
	if(!resizedImg_)
		THROW_WITH_CONST_MSG("Provided nullptr param to " __FUNCTION__, invalid_argument);

	const bool result = !resizedImg || (*resizedImg != *resizedImg_);

	if(result)
		resizedImg = resizedImg_;
	comp.setReference(resizedImg->get());
	if(result)
		comp.resize();

	return result;
}

void Controller::showResultedImage(double completionDurationS) {
	comp.setResult(t.getResult()); // display the result at the end of the transformation
	if(completionDurationS > 0.) {
		reportDuration("The transformation took", completionDurationS);
		t.setDuration(completionDurationS);
	}
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
	@param fontSz provides the font size
	@param areClustersIgnored adapts the display of clusters depending on their importance
	@param clusterOffsets where does each cluster start
	@param idxOfFirstSymFromPage index of the first symbol to be displayed on the page
	*/
	template<typename Iterator>
	void populateGrid(Iterator it, Iterator itEnd,
					  NegSymExtractor<Iterator> negSymExtractor,
					  Mat &content, const Mat &grid,
					  int fontSz, bool areClustersIgnored,
					  const set<unsigned> &clusterOffsets = {},
					  unsigned idxOfFirstSymFromPage = UINT_MAX) {
		content = grid.clone();
		const unsigned symsToShow = (unsigned)distance(it, itEnd);
		const int cellSide = 1 + fontSz,
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
			if(areClustersIgnored) { // use dashed lines as splits
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

#ifdef AI_REVIEWER_CHECK
	template<class It>
	const Mat fnNegSymExtractor(const typename It &) { return Mat(); }
#endif // AI_REVIEWER_CHECK defined

} // anonymous namespace

void CmapInspect::populateGrid(const CmapPerspective::VPSymDataCItPair &itPair,
							   const set<unsigned> &clusterOffsets,
							   unsigned idxOfFirstSymFromPage) {
	CmapPerspective::VPSymDataCIt it = itPair.first, itEnd = itPair.second;
	::populateGrid(it, itEnd,
#ifndef AI_REVIEWER_CHECK
				   (NegSymExtractor<CmapPerspective::VPSymDataCIt>) // conversion
				   [](const CmapPerspective::VPSymDataCIt &iter) -> Mat {
						if((*iter)->removable)
							return 255U - (*iter)->negSym;
						return (*iter)->negSym;
					},
#else // AI_REVIEWER_CHECK defined
				   fnNegSymExtractor,
#endif // AI_REVIEWER_CHECK
				   content, grid, (int)fontSz,
				   !const_cast<IPresentCmap*>(cmapPresenter.get())->markClustersAsUsed(),
				   clusterOffsets, idxOfFirstSymFromPage);
}

void CmapInspect::showUnofficial1stPage(vector<const Mat> &symsOn1stPage,
										atomic_flag &updating1stCmapPage,
										LockFreeQueue &updateSymsActionsQueue) {
	std::shared_ptr<Mat> unofficial = std::make_shared<Mat>();
	::populateGrid(CBOUNDS(symsOn1stPage),
#ifndef AI_REVIEWER_CHECK
				   (NegSymExtractor<vector<const Mat>::const_iterator>) // conversion
				   [](const vector<const Mat>::const_iterator &iter) { return *iter; },
#else // AI_REVIEWER_CHECK defined
				   fnNegSymExtractor,
#endif // AI_REVIEWER_CHECK
				   *unofficial, grid, (int)fontSz,
				   !const_cast<IPresentCmap*>(cmapPresenter.get())->markClustersAsUsed());

	symsOn1stPage.clear(); // discard values now

	// If the main worker didn't register its intention to render already the official 1st cmap page
	// display the unofficial early version.
	// Otherwise just leave
	if(!updating1stCmapPage.test_and_set()) { // holds the value on true after acquiring it
		// Creating local copies that can be passed by value to Unofficial1stPageCmap's parameter
		auto *pUpdating1stCmapPage = &updating1stCmapPage;
		const auto winNameCopy = winName;

		updateSymsActionsQueue.push(new BasicUpdateSymsAction([pUpdating1stCmapPage, winNameCopy, unofficial] {
			imshow(winNameCopy, *unofficial);

			pUpdating1stCmapPage->clear(); // puts its value on false, to be acquired by the official version publisher
		}));
	}
}

CmapInspect::CmapInspect(std::shared_ptr<const IPresentCmap> cmapPresenter_,
						 std::shared_ptr<const ISelectSymbols> symsSelector_,
						 const unsigned &fontSz_) :
		CvWin(CmapInspectWinName),
		cmapPresenter(cmapPresenter_), symsSelector(symsSelector_), fontSz(fontSz_) {
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
		const ISelectSymbols *pss = reinterpret_cast<ISelectSymbols*>(userdata);
		if(event == EVENT_MOUSEMOVE) { // Mouse move
			const SymData *psd = pss->pointedSymbol(x, y);
			if(nullptr != psd)
				pss->displaySymCode(psd->code);
		} else if((event == EVENT_LBUTTONDBLCLK) && ((flags & EVENT_FLAG_CTRLKEY) == 0)) { // Ctrl key not pressed and left mouse double-click
			pss->symbolsReadyToInvestigate();
		} else if((event == EVENT_LBUTTONUP) && ((flags & EVENT_FLAG_CTRLKEY) != 0)) { // Ctrl key pressed and left mouse click
			const SymData *psd = pss->pointedSymbol(x, y);
			if(nullptr != psd)
				pss->enlistSymbolForInvestigation(*psd);
		}
	}, reinterpret_cast<void*>(const_cast<ISelectSymbols*>(symsSelector.get())));
}

Mat CmapInspect::createGrid() {
#pragma warning ( disable : WARN_THREAD_UNSAFE )
	static const Scalar GridColor(255U, 200U, 200U);
#pragma warning ( default : WARN_THREAD_UNSAFE )

	Mat emptyGrid(CmapInspect_pageSz, CV_8UC3, Scalar::all(255U));

	cellSide = 1U + fontSz;
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

ControlPanel::ControlPanel(IControlPanelActions &performer_, const ISettings &cfg_) :
		performer(performer_), cfg(cfg_),
		maxHSyms((int)cfg_.getIS().getMaxHSyms()), maxVSyms((int)cfg_.getIS().getMaxVSyms()),
		encoding(0), fontSz((int)cfg_.getSS().getFontSz()),
		symsBatchSz((int)SymsBatch_defaultSz),
		hybridResult(cfg_.getMS().isHybridResult() ? 1 : 0),
		structuralSim(Converter::StructuralSim::toSlider(cfg_.getMS().get_kSsim())),
		underGlyphCorrectness(Converter::Correctness::toSlider(cfg_.getMS().get_kSdevFg())),
		glyphEdgeCorrectness(Converter::Correctness::toSlider(cfg_.getMS().get_kSdevEdge())),
		asideGlyphCorrectness(Converter::Correctness::toSlider(cfg_.getMS().get_kSdevBg())),
		moreContrast(Converter::Contrast::toSlider(cfg_.getMS().get_kContrast())),
		gravity(Converter::Gravity::toSlider(cfg_.getMS().get_kMCsOffset())),
		direction(Converter::Direction::toSlider(cfg_.getMS().get_kCosAngleMCs())),
		largerSym(Converter::LargerSym::toSlider(cfg_.getMS().get_kSymDensity())),
		thresh4Blanks((int)cfg_.getMS().getBlankThreshold()) {
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

		try {
			if(sf.promptForUserChoice())
				pActions->newFontFamily(sf.selection());
		} catch(FontLocationFailure&) {
			pActions->invalidateFont();
			extern const string CannotLoadFontErrSuffix;
			infoMsg("Couldn't locate the selected font!" + CannotLoadFontErrSuffix, "Manageable Error");
		}
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