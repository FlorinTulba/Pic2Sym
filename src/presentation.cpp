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
 
 (c) 2016-2019 Florin Tulba <florintulba@yahoo.com>

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

#include "comparatorBase.h"
#include "cmapInspectBase.h"
#include "controller.h"
#include "fontEngine.h"
#include "transform.h"
#include "imgBasicData.h"
#include "resizedImg.h"
#include "updateSymSettingsBase.h"
#include "glyphsProgressTracker.h"
#include "picTransformProgressTracker.h"
#include "selectSymbols.h"
#include "controlPanelActions.h"
#include "settingsBase.h"
#include "imgSettingsBase.h"
#include "symSettingsBase.h"
#include "matchSettings.h"
#include "jobMonitorBase.h"
#include "matchEngine.h"
#include "matchParamsBase.h"
#include "matchAssessment.h"
#include "structuralSimilarity.h"
#include "updateSymsActions.h"
#include "views.h"
#include "symsLoadingFailure.h"
#include "presentCmapBase.h"

#pragma warning ( push, 0 )

#include <Windows.h>

// Using <thread> in VS2013 might trigger this warning:
// https://connect.microsoft.com/VisualStudio/feedback/details/809540/c-warnings-in-stl-thread
#pragma warning( disable : WARN_VIRT_DESTRUCT_EXPECTED )
#include <thread>
#pragma warning( default : WARN_VIRT_DESTRUCT_EXPECTED )

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

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

void viewMismatches(const stringType &testTitle, const Mat &mismatches) {
	const int twiceTheRows = mismatches.rows, rows = twiceTheRows>>1, cols = mismatches.cols;
	const Mat reference = mismatches.rowRange(0, rows), // upper half is the reference
		result = mismatches.rowRange(rows, twiceTheRows); // lower half is the result

	// Comparator window size should stay within ~ 800x600
	// Enlarge up to 3 times if resulting rows < 600.
	// Enlarge also when resulted width would be less than 140 (width when the slider is visible)
	const double resizeFactor = max(140./cols, min(600./rows, 3.));

	ostringstream oss;
	oss<<"View mismatches for "<<testTitle;
	const stringType title(oss.str());

	Comparator comp;
	comp.permitResize();
	comp.resize(4+(int)ceil(cols*resizeFactor), 70+(int)ceil(rows*resizeFactor));
	comp.setTitle(title.c_str());
	comp.setStatus("Press Esc to close this window");
	comp.setReference(reference);
	comp.setResult(result, 90); // Emphasize the references 

	Controller::handleRequests();
}

void viewMisfiltered(const stringType &testTitle, const Mat &misfiltered) {
	const String winName = testTitle;
	namedWindow(winName);
	setWindowProperty(winName, CV_WND_PROP_AUTOSIZE, CV_WINDOW_NORMAL);
	imshow(winName, misfiltered);
	displayStatusBar(winName, "Press Esc to close this window");
	waitKey();
}

#endif // UNIT_TESTING not defined

void pauseAfterError() {
	stringType line;
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

/// Displays a histogram with the distribution of the weights of the symbols from the charmap
static void viewSymWeightsHistogram(const VPixMapSym &theSyms) {
#ifndef UNIT_TESTING
	vector<double> symSums;
	for(const uniquePtr<const IPixMapSym> &pms : theSyms)
		symSums.push_back(pms->getAvgPixVal());

	static const size_t MaxBinHeight = 256ULL;
	const size_t binsCount = min(256ULL, symSums.size());
	const double smallestSum = symSums.front(), largestSum = symSums.back(),
		sumsSpan = largestSum - smallestSum;
	vector<size_t> hist(binsCount, 0ULL);
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
	for(size_t bin = 0ULL; bin < binsCount; ++bin)
		if(hist[bin] > 0ULL)
			histImg.rowRange(int(MaxBinHeight-hist[bin]), (int)MaxBinHeight).col((int)bin) = 0U;
	imshow("histogram", histImg);
	waitKey(1);
#endif // UNIT_TESTING not defined
}

stringType FontEngine::getFontType() {
	ostringstream oss;
	oss<<getFamily()<<'_'<<getStyle()<<'_'<<getEncoding();

	return oss.str();
}

stringType MatchEngine::getIdForSymsToUse() {
	const unsigned sz = cfg.getSS().getFontSz();
	assert(ISettings::isFontSizeOk(sz));

	ostringstream oss;
	oss<<fe.getFontType()<<'_'<<sz;

	return oss.str();
}

const stringType MatchSettings::toString(bool verbose) const {
	ostringstream oss;
	if(verbose) {
		if(hybridResultMode)
			oss<<"hybridResultMode : true"<<endl;

#define REPORT_POSITIVE_PARAM(paramName) \
		if(paramName > 0.) \
			oss<<#paramName " : "<<paramName<<endl

		REPORT_POSITIVE_PARAM(kSsim);
		REPORT_POSITIVE_PARAM(kCorrel);
		REPORT_POSITIVE_PARAM(kSdevFg);
		REPORT_POSITIVE_PARAM(kSdevEdge);
		REPORT_POSITIVE_PARAM(kSdevBg);
		REPORT_POSITIVE_PARAM(kContrast);
		REPORT_POSITIVE_PARAM(kMCsOffset);
		REPORT_POSITIVE_PARAM(kCosAngleMCs);
		REPORT_POSITIVE_PARAM(kSymDensity);

#undef REPORT_POSITIVE_PARAM

		if(threshold4Blank > 0.)
			oss<<"threshold4Blank : "<<threshold4Blank<<endl;

	} else {
		oss<<hybridResultMode<<'_'
			<<kSsim<<'_'<<kCorrel<<'_'
			<<kSdevFg<<'_'<<kSdevEdge<<'_'<<kSdevBg<<'_'
			<<kContrast<<'_'<<kMCsOffset<<'_'<<kCosAngleMCs<<'_'
			<<kSymDensity<<'_'<<threshold4Blank;
	}

	return std::move(oss.str());
}

void Transformer::updateStudiedCase(int rows, int cols) {
	const IMatchSettings &ms = cfg.getMS();
	ostringstream oss;

	// no extension yet
	oss<<img.name()<<'_'<<me.getIdForSymsToUse()<<'_'<<ms.toString(false)<<'_'<<cols<<'_'<<rows;
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

const stringType Controller::textForCmapStatusBar(unsigned upperSymsCount/* = 0U*/) const {
	assert(nullptr != fe.getFamily() && 0ULL < strlen(fe.getFamily()));
	assert(nullptr != fe.getStyle() && 0ULL < strlen(fe.getStyle()));
	assert(!fe.getEncoding().empty());
	ostringstream oss;
	oss<<"Font type '"<<fe.getFamily()<<' '<<fe.getStyle()
		<<"', size "<<cfg.getSS().getFontSz()<<", encoding '"<<fe.getEncoding()<<'\''
		<<" : "<<((upperSymsCount != 0U) ? upperSymsCount : (unsigned)fe.symsSet().size())<<" symbols";
	return oss.str();
}

const stringType Controller::textHourGlass(const std::stringType &prefix, double progress) const {
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
			const stringType errText(excObj.what());
			// An exception with the same errText will be thrown in the main thread when executing next action
			updateSymsActionsQueue.push(new BasicUpdateSymsAction([errText] {
				throw NormalSymsLoadingFailure(errText);
			}));

			updatingSymbols.clear(); // signal task completion
			return;

		} catch(TinySymsLoadingFailure &excObj) { // capture it in one thread, then pack it for the other thread
			const stringType errText(excObj.what());
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

#ifndef AI_REVIEWER_CHECK // AI Reviewer might not tackle the following lambda as expected
	auto performRegisteredActions = [&] { // lambda used twice below
		IUpdateSymsAction *action = nullptr;
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

#else // AI_REVIEWER_CHECK defined
	// AI Reviewer needs to be aware of the methods called within previous lambda
	IUpdateSymsAction *action = nullptr;
	updateSymsActionsQueue.pop(action);
	action->perform();

	// Define a placeholder for the lambda above
#define performRegisteredActions()

#endif // AI_REVIEWER_CHECK

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

#ifndef UNIT_TESTING

Controller::~Controller() {
	destroyAllWindows();
}

void Controller::display1stPageIfFull(const VPixMapSym &syms) {
	if((unsigned)syms.size() != pCmi->getSymsPerPage())
		return;

	// Copy all available symbols from the 1st page in current thread
	// to ensure they get presented in the original order by the other thread
	vector<const Mat> *matSyms = new vector<const Mat>;
	matSyms->reserve(syms.size());
	const auto fontSz = getFontSize();
	for(const uniquePtr<const IPixMapSym> &pms : syms)
		matSyms->emplace_back(pms->toMat(fontSz, true));

	// Starting the thread that builds the 'pre-release' version of the 1st cmap page
	thread([&, matSyms] {
				pCmi->showUnofficial1stPage(*matSyms, updating1stCmapPage, updateSymsActionsQueue);
				delete matSyms;
			}
		).detach(); // termination doesn't matter
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

void Controller::hourGlass(double progress, const stringType &title/* = ""*/, bool async/* = false*/) const {
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
			const stringType hourGlassText = textHourGlass(oss.str(), progress);
			setWindowTitle(waitWin, hourGlassText);
		}
	}
}

void Controller::updateStatusBarCmapInspect(unsigned upperSymsCount/* = 0U*/,
											const stringType &suffix/* = ""*/,
											bool async/* = false*/) const {
	const stringType newStatusBarText(textForCmapStatusBar(upperSymsCount) + suffix);
	if(async) { // placing a task in the queue for the GUI updating thread
		updateSymsActionsQueue.push(new BasicUpdateSymsAction([&, newStatusBarText] {
			pCmi->setStatus(newStatusBarText);
		}));
	} else { // direct call or one async call due now
		pCmi->setStatus(newStatusBarText);
	}
}

void Controller::reportDuration(const std::stringType &text, double durationS) const {
	ostringstream oss;
	oss<<text<<' '<<durationS<<" s!";
	const stringType cmapOverlayText(oss.str());
	cout<<endl<<cmapOverlayText<<endl<<endl;
	pCmi->setOverlay(cmapOverlayText, 3000);
}

bool Controller::updateResizedImg(const IResizedImg &resizedImg_) {
	const ResizedImg &paramVal = dynamic_cast<const ResizedImg&>(resizedImg_);
	const bool result = !resizedImg ||
		(dynamic_cast<const ResizedImg&>(*resizedImg) != paramVal);

	if(result)
		resizedImg = makeUnique<const ResizedImg>(paramVal);
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
	extern const stringType Comparator_initial_title, Comparator_statusBar;

	createTrackbar(Comparator_transpTrackName, winName,
				   &trackPos, Comparator_trackMax,
				   &Comparator::updateTransparency, reinterpret_cast<void*>(this));
	Comparator::updateTransparency(trackPos, reinterpret_cast<void*>(this)); // mandatory

	// Default initial configuration
	setPos(0, 0);
	permitResize(false);
	setTitle(Comparator_initial_title);
	setStatus(Comparator_statusBar);
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
	const Mat fnNegSymExtractor(const typename It &it) {
		// Let AI Reviewer know that ISymData::isRemovable() and ISymData::getNegSym()
		// were used in the actual lambda function (see below, where fnNegSymExtractor is called)
		const ISymData &symData = **it;
		symData.isRemovable();
		return symData.getNegSym();
	}
#endif // AI_REVIEWER_CHECK defined

} // anonymous namespace

void CmapInspect::populateGrid(const ICmapPerspective::VPSymDataCItPair &itPair,
							   const set<unsigned> &clusterOffsets,
							   unsigned idxOfFirstSymFromPage) {
	ICmapPerspective::VPSymDataCIt it = itPair.first, itEnd = itPair.second;
	::populateGrid(it, itEnd,
#ifndef AI_REVIEWER_CHECK
				   (NegSymExtractor<ICmapPerspective::VPSymDataCIt>) // conversion
				   [](const ICmapPerspective::VPSymDataCIt &iter) -> Mat {
						if((*iter)->isRemovable())
							return 255U - (*iter)->getNegSym();
						return (*iter)->getNegSym();
					},
#else // AI_REVIEWER_CHECK defined
				   fnNegSymExtractor,
#endif // AI_REVIEWER_CHECK
				   content, grid, (int)fontSz,
				   !cmapPresenter.areClustersUsed(),
				   clusterOffsets, idxOfFirstSymFromPage);
}

void CmapInspect::showUnofficial1stPage(vector<const Mat> &symsOn1stPage,
										atomic_flag &updating1stCmapPage,
										LockFreeQueue &updateSymsActionsQueue) {
	std::sharedPtr<Mat> unofficial = std::makeShared<Mat>();
	::populateGrid(CBOUNDS(symsOn1stPage),
#ifndef AI_REVIEWER_CHECK
				   (NegSymExtractor<vector<const Mat>::const_iterator>) // conversion
				   [](const vector<const Mat>::const_iterator &iter) { return *iter; },
#else // AI_REVIEWER_CHECK defined
				   fnNegSymExtractor,
#endif // AI_REVIEWER_CHECK
				   *unofficial, grid, (int)fontSz,
				   !cmapPresenter.areClustersUsed());

	symsOn1stPage.clear(); // discard values now

	// If the main worker didn't register its intention to render already the official 1st cmap page
	// display the unofficial early version.
	// Otherwise just leave
	if(!updating1stCmapPage.test_and_set()) { // holds the value on true after acquiring it
		// Creating local copies that can be passed by value to Unofficial1stPageCmap's parameter
		atomic_flag *pUpdating1stCmapPage = &updating1stCmapPage;
		const String winNameCopy = winName;

		updateSymsActionsQueue.push(new BasicUpdateSymsAction([pUpdating1stCmapPage, winNameCopy, unofficial] {
			imshow(winNameCopy, *unofficial);

			pUpdating1stCmapPage->clear(); // puts its value on false, to be acquired by the official version publisher
		}));
	}
}

CmapInspect::CmapInspect(const IPresentCmap &cmapPresenter_,
						 const ISelectSymbols &symsSelector_,
						 const unsigned &fontSz_) :
		CvWin(CmapInspectWinName),
		cmapPresenter(cmapPresenter_), symsSelector(symsSelector_), fontSz(fontSz_) {
	extern const String CmapInspect_pageTrackName;
	content = grid = createGrid();

	/*
	`CmapInspect::updatePageIdx` from below needs to work with a `void*` of actual type `ICmapInspect*`.
	But, passing `reinterpret_cast<void*>(this)` will fail, since `this` is a `CmapInspect*`
	and the actual address of `(ICmapInspect*)this` is different from `this`
	because `ICmapInspect` isn't the top inherited interface.
	*/
	void * const thisAsICmapInspectPtr = reinterpret_cast<void*>((ICmapInspect*)this);

	createTrackbar(CmapInspect_pageTrackName, winName, &page, 1, &CmapInspect::updatePageIdx,
				   thisAsICmapInspectPtr);
	CmapInspect::updatePageIdx(page, thisAsICmapInspectPtr); // mandatory call

	setPos(424, 0);			// Place cmap window on x axis between 424..1064
	permitResize(false);	// Ensure the user sees the symbols exactly their size

	// Support for investigating selected symbols from one or more charmaps:
	// - mouse moves over the Charmap will display in the status bar the code of the pointed symbol
	// - Ctrl + left mouse click will append the pointed symbol to the current list
	// - left mouse double-click will save the current list and then it will clear it
#ifndef AI_REVIEWER_CHECK // AI Reviewer might not parse correctly such lambda-s
	setMouseCallback(CmapInspectWinName, [] (int event, int x, int y, int flags, void* userdata) {
		ISelectSymbols *pss = reinterpret_cast<ISelectSymbols*>(userdata);
		if(event == EVENT_MOUSEMOVE) { // Mouse move
			const ISymData *psd = pss->pointedSymbol(x, y);
			if(nullptr != psd)
				pss->displaySymCode(psd->getCode());
		} else if((event == EVENT_LBUTTONDBLCLK) && ((flags & EVENT_FLAG_CTRLKEY) == 0)) { // Ctrl key not pressed and left mouse double-click
			pss->symbolsReadyToInvestigate();
		} else if((event == EVENT_LBUTTONUP) && ((flags & EVENT_FLAG_CTRLKEY) != 0)) { // Ctrl key pressed and left mouse click
			const ISymData *psd = pss->pointedSymbol(x, y);
			if(nullptr != psd)
				pss->enlistSymbolForInvestigation(*psd);
		}
	}, reinterpret_cast<void*>(const_cast<ISelectSymbols*>(&symsSelector)));

#else // AI_REVIEWER_CHECK defined
	// Let AI Reviewer know that following methods were used within the lambda above
	ISelectSymbols *pss = const_cast<ISelectSymbols*>(&symsSelector);
	const ISymData *psd = pss->pointedSymbol(0, 0);
	pss->displaySymCode(psd->getCode());
	pss->symbolsReadyToInvestigate();
	pss->enlistSymbolForInvestigation(*psd);
#endif // AI_REVIEWER_CHECK
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
	for(const vector<unsigned> &clusterMembers : symsIndicesPerCluster_)
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
	for(const vector<unsigned> *clusterMembers : symsIndicesPerCluster) {
		const auto prevOffset = offset;
		offset += clusterMembers->size();
		clusterOffsets.emplace_hint(end(clusterOffsets), (unsigned)offset);
		for(size_t idxPSyms = prevOffset, idxCluster = 0ULL; idxPSyms < offset; ++idxPSyms, ++idxCluster)
			pSyms[idxPSyms] = symsSet[(size_t)(*clusterMembers)[idxCluster]].get();
	}
}

ICmapPerspective::VPSymDataCItPair CmapPerspective::getSymsRange(unsigned from, unsigned count) const {
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

#endif // UNIT_TESTING not defined
