/******************************************************************************
 The application Pic2Sym approximates images by a
 grid of colored symbols with colored backgrounds.

 Copyrights from the libraries used by the program:
 - (c) 2003 Boost (www.boost.org)
     License: doc/licenses/Boost.lic
     http://www.boost.org/LICENSE_1_0.txt
 - (c) 2015-2016 OpenCV (www.opencv.org)
     License: doc/licenses/OpenCV.lic
     http://opencv.org/license/
 - (c) 1996-2002, 2006 The FreeType Project (www.freetype.org)
     License: doc/licenses/FTL.txt
     http://git.savannah.gnu.org/cgit/freetype/freetype2.git/plain/docs/FTL.TXT
 - (c) 1997-2002 OpenMP Architecture Review Board (www.openmp.org)
   (c) Microsoft Corporation (implementation for OpenMP C/C++ v2.0 March 2002)
     See: https://msdn.microsoft.com/en-us/library/8y6825x5.aspx
 - (c) 1995-2017 zlib software (Jean-loup Gailly and Mark Adler - www.zlib.net)
     License: doc/licenses/zlib.lic
     http://www.zlib.net/zlib_license.html


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
 If not, see: http://www.gnu.org/licenses/agpl-3.0.txt .
 *****************************************************************************/

#include "precompiled.h"

#include "cmapInspectBase.h"
#include "comparatorBase.h"
#include "controlPanelActions.h"
#include "controller.h"
#include "fontEngine.h"
#include "glyphsProgressTracker.h"
#include "imgBasicData.h"
#include "imgSettingsBase.h"
#include "jobMonitorBase.h"
#include "matchAssessment.h"
#include "matchEngine.h"
#include "matchParamsBase.h"
#include "matchSettings.h"
#include "picTransformProgressTracker.h"
#include "presentCmapBase.h"
#include "resizedImg.h"
#include "selectSymbols.h"
#include "settingsBase.h"
#include "structuralSimilarity.h"
#include "symSettingsBase.h"
#include "symsLoadingFailure.h"
#include "transform.h"
#include "updateSymSettingsBase.h"
#include "updateSymsActions.h"
#include "views.h"
#include "warnings.h"

#pragma warning(push, 0)

#define WIN32_LEAN_AND_MEAN
#include <Windows.h>

#include <thread>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#pragma warning(pop)

#ifndef UNIT_TESTING

#include "cmapPerspective.h"

#pragma warning(push, 0)

#include <numeric>

#pragma warning(pop)

#endif  // UNIT_TESTING

using namespace std;
using namespace std::chrono;
using namespace boost::lockfree;
using namespace cv;

// UnitTesting project launches an instance of Pic2Sym to visualize any
// discovered issues, so it won't refer viewMismatches and viewMisfiltered from
// below. But it will start a Pic2Sym instance that will do such calls.
#ifndef UNIT_TESTING

void viewMismatches(const string& testTitle, const Mat& mismatches) noexcept {
  const int twiceTheRows = mismatches.rows, rows = twiceTheRows >> 1,
            cols = mismatches.cols;
  const Mat reference =
                mismatches.rowRange(0, rows),  // upper half is the reference
      result =
          mismatches.rowRange(rows, twiceTheRows);  // lower half is the result

  // Comparator window size should stay within ~ 800x600
  // Enlarge up to 3 times if resulting rows < 600.
  // Enlarge also when resulted width would be less than 140 (width when the
  // slider is visible)
  const double resizeFactor = max(140. / cols, min(600. / rows, 3.));

  ostringstream oss;
  oss << "View mismatches for " << testTitle;
  const string title(oss.str());

  Comparator comp;
  comp.permitResize();
  comp.resize(4 + (int)ceil(cols * resizeFactor),
              70 + (int)ceil(rows * resizeFactor));
  comp.setTitle(title.c_str());
  comp.setStatus("Press Esc to close this window");
  comp.setReference(reference);
  comp.setResult(result, 90);  // Emphasize the references

  Controller::handleRequests();
}

void viewMisfiltered(const string& testTitle, const Mat& misfiltered) noexcept {
  const String winName = testTitle;
  namedWindow(winName);
  setWindowProperty(winName, cv::WND_PROP_AUTOSIZE, cv::WINDOW_NORMAL);
  imshow(winName, misfiltered);
  displayStatusBar(winName, "Press Esc to close this window");
  waitKey();
}

#endif  // UNIT_TESTING not defined

void pauseAfterError() noexcept {
  string line;
  cout << "\nPress Enter to leave\n";
  getline(cin, line);
}

void showUsage() noexcept {
  cout << R"(Usage:
There are 3 launch modes:
A) Normal launch mode (no parameters)
    Pic2Sym.exe

B) Timing a certain scenario (5 parameters)
    Pic2Sym.exe timing "<caseName>" "<settingsPath>" "<imagePath>" "<reportFilePath>"

C) View mismatches launch mode (Support for Unit Testing, using 2 parameters)
    Pic2Sym.exe mismatches "<testTitle>"

D) View misfiltered symbols launch mode (Support for Unit Testing, using 2 parameters)
    Pic2Sym.exe misfiltered "<testTitle>"

)";
  pauseAfterError();
}

/// Displays a histogram with the distribution of the weights of the symbols
/// from the charmap
static void viewSymWeightsHistogram(const VPixMapSym& theSyms) noexcept {
#ifndef UNIT_TESTING
  vector<double> symSums;
  for (const unique_ptr<const IPixMapSym>& pms : theSyms)
    symSums.push_back(pms->getAvgPixVal());

  static const size_t MaxBinHeight = 256ULL;
  const size_t binsCount = min(256ULL, symSums.size());
  const double smallestSum = symSums.front(), largestSum = symSums.back(),
               sumsSpan = largestSum - smallestSum;
  vector<size_t> hist(binsCount, 0ULL);
  const auto itBegin = symSums.cbegin();
  ptrdiff_t prevCount = 0LL;
  for (size_t bin = 0ULL; bin < binsCount; ++bin) {
    const auto it = upper_bound(
        CBOUNDS(symSums), smallestSum + sumsSpan * (bin + 1.) / binsCount);
    const auto curCount = distance(itBegin, it);
    hist[bin] = size_t(curCount - prevCount);
    prevCount = curCount;
  }
  const double maxBinValue = (double)*max_element(CBOUNDS(hist));
  for (size_t& binValue : hist)
    binValue = (size_t)round(binValue * MaxBinHeight / maxBinValue);
  Mat histImg((int)MaxBinHeight, (int)binsCount, CV_8UC1, Scalar(255U));
  for (size_t bin = 0ULL; bin < binsCount; ++bin)
    if (hist[bin] > 0ULL)
      histImg.rowRange(int(MaxBinHeight - hist[bin]), (int)MaxBinHeight)
          .col((int)bin) = 0U;
  imshow("histogram", histImg);
  waitKey(1);

#else   // UNIT_TESTING defined
  UNREFERENCED_PARAMETER(theSyms);
#endif  // UNIT_TESTING
}

string FontEngine::getFontType() const noexcept(!UT) {
  ostringstream oss;
  oss << getFamily() << '_' << getStyle() << '_' << getEncoding();
  // getEncoding() throws logic_error for incomplete font configuration
  // The exception propagates only in UnitTesting

  return oss.str();
}

#pragma warning(disable : WARN_THROWS_ALTHOUGH_NOEXCEPT)
string MatchEngine::getIdForSymsToUse() const noexcept(!UT) {
  const unsigned sz = cfg.getSS().getFontSz();
  if (!ISettings::isFontSizeOk(sz))
    THROW_WITH_VAR_MSG(__FUNCTION__ " read invalid font size: " + to_string(sz),
                       domain_error);

  ostringstream oss;
  oss << fe.getFontType() << '_' << sz;
  // getFontType() throws logic_error for incomplete font configuration
  // The exception propagates only in UnitTesting

  return oss.str();
}
#pragma warning(default : WARN_THROWS_ALTHOUGH_NOEXCEPT)

const string MatchSettings::toString(bool verbose) const noexcept {
  ostringstream oss;
  if (verbose) {
    if (hybridResultMode)
      oss << "hybridResultMode : true\n";

#define REPORT_POSITIVE_PARAM(paramName) \
  if (paramName > 0.)                    \
  oss << #paramName " : " << paramName << '\n'

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

    if (threshold4Blank > 0U)
      oss << "threshold4Blank : " << threshold4Blank << '\n';

  } else {
    oss << hybridResultMode << '_' << kSsim << '_' << kCorrel << '_' << kSdevFg
        << '_' << kSdevEdge << '_' << kSdevBg << '_' << kContrast << '_'
        << kMCsOffset << '_' << kCosAngleMCs << '_' << kSymDensity << '_'
        << threshold4Blank;
  }

  return oss.str();
}

void Transformer::updateStudiedCase(int rows, int cols) noexcept {
  const IMatchSettings& ms = cfg.getMS();
  ostringstream oss;

  // no extension yet
  oss << img.name() << '_' << me.getIdForSymsToUse() << '_'
      << ms.toString(false) << '_' << cols << '_' << rows;
  studiedCase =
      oss.str();  // this text is included in the result & trace file names
}

#ifdef MONITOR_SKIPPED_MATCHING_ASPECTS

void MatchAssessorSkip::reportSkippedAspects() const noexcept {
  cout << "\nTransformation finished. "
          "Reporting skipped aspects from a total of "
       << totalIsBetterMatchCalls << " isBetterMatch calls:\n";
  for (size_t i = 0ULL; i < enabledAspectsCount; ++i) {
    if (skippedAspects[i] == 0ULL)
      continue;
    cout << "\t\t" << setw(25) << left << enabledAspects[i]->name() << " : "
         << setw(10) << right << skippedAspects[i] << " times"
         << " (Complexity : " << setw(8) << fixed << setprecision(3) << right
         << enabledAspects[i]->relativeComplexity() << ")"
         << " [" << setw(5) << fixed << setprecision(2) << right
         << (100. * skippedAspects[i] / totalIsBetterMatchCalls)
         << "% of the calls]\n";
  }
  cout << endl;
}

#endif  // MONITOR_SKIPPED_MATCHING_ASPECTS

const string Controller::textForCmapStatusBar(
    unsigned upperSymsCount /* = 0U*/) const noexcept {
  // crashes the program for incomplete font configuration
  const string& enc = fe.getEncoding();

  assert(nullptr != fe.getFamily());  // is empty in the worst case
  assert(nullptr != fe.getStyle());   // is empty in the worst case

  ostringstream oss;
  oss << "Font type '" << fe.getFamily() << ' ' << fe.getStyle() << "', size "
      << cfg.getSS().getFontSz() << ", encoding '" << enc << '\'' << " : "
      << ((upperSymsCount != 0U)
              ? upperSymsCount

              // crashes the program for incomplete font configuration
              : (unsigned)fe.symsSet().size())
      << " symbols";
  return oss.str();
}

const string Controller::textHourGlass(const std::string& prefix,
                                       double progress) const noexcept {
  ostringstream oss;
  oss << prefix << " (" << fixed << setprecision(0) << progress * 100. << "%)";
  return oss.str();
}

void Controller::symbolsChanged() {
  if (!pCmi)
    THROW_WITH_CONST_MSG(__FUNCTION__ " called before "
                         "ensureExistenceCmapInspect()!", logic_error);

  // Timing the update.
  // timer's destructor will report duration and will close the hourglass
  // window. These actions will be launched in this GUI updating thread, after
  // processing all previously registered actions.
  Timer timer = glyphsProgressTracker->createTimerForGlyphs();

  updatingSymbols.test_and_set();
  updating1stCmapPage.clear();
  pCmi->setBrowsable(false);

  // Starting a thread to perform the actual change of the symbols,
  // while preserving this thread for GUI updating
  thread([&]() noexcept {
    glyphsUpdateMonitor->setTasksDetails(
        {
            .01,   // determine optimal square-fitting for the symbols
            .185,  // load & filter symbols that fit the square
            .01,   // load & filter extra-squeezed symbols
            .005,  // determine coverageOfSmallGlyphs
            .17,   // computing specific symbol-related values
            .61,   // clustering the small symbols
            .01    // reorders clusters
        },
        timer);

    try {
      fe.setFontSz(cfg.getSS().getFontSz());
      me.updateSymbols();

    } catch (const NormalSymsLoadingFailure& excObj) {
      // capture it in one thread, then pack it for the other thread
      const string errText(excObj.what());
      // An exception with the same errText will be thrown in the main thread
      // when executing next action

#pragma warning(disable : WARN_EXPLICIT_NEW_OR_DELETE)
      updateSymsActionsQueue.push(new BasicUpdateSymsAction(
          [errText] { throw NormalSymsLoadingFailure(errText); }));
#pragma warning(default : WARN_EXPLICIT_NEW_OR_DELETE)

      updatingSymbols.clear();  // signal task completion
      return;

    } catch (const TinySymsLoadingFailure& excObj) {
      // capture it in one thread, then pack it for the other thread
      const string errText(excObj.what());
      // An exception with the same errText will be thrown in the main thread
      // when executing next action

#pragma warning(disable : WARN_EXPLICIT_NEW_OR_DELETE)
      updateSymsActionsQueue.push(new BasicUpdateSymsAction(
          [errText] { throw TinySymsLoadingFailure(errText); }));
#pragma warning(default : WARN_EXPLICIT_NEW_OR_DELETE)

      updatingSymbols.clear();  // signal task completion
      return;
    }

    // Symbols have been changed in the model. Only GUI must be updated
    // The GUI will display the 1st cmap page.
    // This must happen after an eventual early preview of it:
    // - we need to wait for the preview to finish if it started before this
    // point
    // - we have to prevent an available preview to be displayed after the
    // official version
    while (updating1stCmapPage.test_and_set())
      this_thread::sleep_for(1ms);
    updatingSymbols.clear();  // signal task completion
  })
      .detach();  // termination captured by updatingSymbols flag

  const auto performRegisteredActions = [&] {  // lambda used twice below
    // Only read pAction pointers, but use it as a unique_ptr
    IUpdateSymsAction* pAction = nullptr;
    unique_ptr<IUpdateSymsAction> action;

    while (updateSymsActionsQueue.pop(pAction)) {  // perform available actions
      action.reset(pAction);

#pragma warning(disable : WARN_SEH_NOT_CAUGHT)
      try {
        action->perform();
      } catch (...) {
        // Discarding all remaining actions
        while (updateSymsActionsQueue.pop(pAction))
          action.reset(pAction);

        action.reset();  // makes sure this happens, as the throw below skips it

        // Postponing the processing of the exception
        // See above NormalSymsLoadingFailure & TinySymsLoadingFailure being
        // thrown
        throw;
      }
#pragma warning(default : WARN_SEH_NOT_CAUGHT)

      waitKey(1);
    }
  };

  while (updatingSymbols.test_and_set())  // loop while work is carried out
    performRegisteredActions();
  performRegisteredActions();  // perform any remaining actions

  // Official versions of the status bar and the 1st cmap page
  pCmi->setStatus(textForCmapStatusBar());
  pCmi->updatePagesCount((unsigned)fe.symsSet().size());
  pCmi->setBrowsable();

  extern const bool ViewSymWeightsHistogram;
  if (ViewSymWeightsHistogram)
    viewSymWeightsHistogram(fe.symsSet());
}

#ifndef UNIT_TESTING

Controller::~Controller() noexcept {
  destroyAllWindows();
}

#pragma warning(disable : WARN_THROWS_ALTHOUGH_NOEXCEPT)
void Controller::display1stPageIfFull(const VPixMapSym& syms) const
    noexcept(!UT) {
  if (!pCmi)
    THROW_WITH_CONST_MSG(__FUNCTION__ " should be called after "
                         "ensureExistenceCmapInspect()!", logic_error);

  if ((unsigned)syms.size() != pCmi->getSymsPerPage())
    return;

  // Copy all available symbols from the 1st page in current thread
  // to ensure they get presented in the original order by the other thread
  unique_ptr<vector<Mat>> matSyms = make_unique<vector<Mat>>();

  matSyms->reserve(syms.size());
  const auto fontSz = getFontSize();
  for (const unique_ptr<const IPixMapSym>& pms : syms)
    matSyms->emplace_back(pms->toMat(fontSz, true));

  // Starting the thread that builds the 'pre-release' version of the 1st cmap
  // page
  thread([&, matSymsOwner = move(matSyms) ]() noexcept {
    pCmi->showUnofficial1stPage(*matSymsOwner, updating1stCmapPage,
                                updateSymsActionsQueue);
  })
      .detach();  // termination doesn't matter

  assert(!matSyms);
}
#pragma warning(default : WARN_THROWS_ALTHOUGH_NOEXCEPT)

void Controller::handleRequests() noexcept {
  for (;;) {
    // When pressing ESC, prompt the user if he wants to exit
    if (27 == waitKey() &&
        IDYES == MessageBox(nullptr, L"Do you want to leave the application?",
                            L"Question",
                            MB_ICONQUESTION | MB_YESNOCANCEL | MB_TASKMODAL |
                                MB_SETFOREGROUND))
      break;
  }
}

#pragma warning(disable : WARN_THROWS_ALTHOUGH_NOEXCEPT)
void Controller::hourGlass(double progress,
                           const string& title /* = ""*/,
                           bool async /* = false*/) const noexcept(!UT) {
  if (progress < -EPS || progress > EPSp1)
    THROW_WITH_CONST_MSG(__FUNCTION__ " needs progress to be between 0 .. 1!",
                         invalid_argument);

  static const String waitWin = "Please Wait!";

  if (async) {  // enqueue the call
#pragma warning(disable : WARN_EXPLICIT_NEW_OR_DELETE)
    updateSymsActionsQueue.push(new BasicUpdateSymsAction(
        [&, progress ]() noexcept { hourGlass(progress, title); }));
#pragma warning(default : WARN_EXPLICIT_NEW_OR_DELETE)

  } else {  // direct call or one async call due now
    if (progress == 0.) {
      // no status bar, nor toolbar
      namedWindow(waitWin, cv::WINDOW_GUI_NORMAL);
      moveWindow(waitWin, 0, 400);

#ifndef _DEBUG  // destroyWindow in Debug mode triggers deallocation of invalid
                // block
    } else if (progress == 1.) {
      destroyWindow(waitWin);

#endif  // in Debug mode, leaving the hourGlass window visible, but with 100% as
        // title

    } else {
      ostringstream oss;
      if (title.empty())
        oss << waitWin;
      else
        oss << title;
      const string hourGlassText = textHourGlass(oss.str(), progress);
      setWindowTitle(waitWin, hourGlassText);
    }
  }
}
#pragma warning(default : WARN_THROWS_ALTHOUGH_NOEXCEPT)

#pragma warning(disable : WARN_THROWS_ALTHOUGH_NOEXCEPT)
void Controller::updateStatusBarCmapInspect(unsigned upperSymsCount /* = 0U*/,
                                            const string& suffix /* = ""*/,
                                            bool async /* = false*/) const
    noexcept(!UT) {
  if (!pCmi)
    THROW_WITH_CONST_MSG(__FUNCTION__ " should be called after "
                         "ensureExistenceCmapInspect()!", logic_error);

  const string newStatusBarText(textForCmapStatusBar(upperSymsCount) + suffix);
  if (async) {  // placing a task in the queue for the GUI updating thread
#pragma warning(disable : WARN_EXPLICIT_NEW_OR_DELETE)
    updateSymsActionsQueue.push(
        new BasicUpdateSymsAction([&, newStatusBarText ]() noexcept {
          pCmi->setStatus(newStatusBarText);
        }));
#pragma warning(default : WARN_EXPLICIT_NEW_OR_DELETE)
  } else {  // direct call or one async call due now
    pCmi->setStatus(newStatusBarText);
  }
}
#pragma warning(default : WARN_THROWS_ALTHOUGH_NOEXCEPT)

#pragma warning(disable : WARN_THROWS_ALTHOUGH_NOEXCEPT)
void Controller::reportDuration(const std::string& text, double durationS) const
    noexcept(!UT) {
  if (!pCmi)
    THROW_WITH_CONST_MSG(__FUNCTION__ " should be called after "
                         "ensureExistenceCmapInspect()!", logic_error);

  ostringstream oss;
  oss << text << ' ' << durationS << " s!";
  const string cmapOverlayText(oss.str());
  cout << '\n' << cmapOverlayText << '\n' << endl;
  pCmi->setOverlay(cmapOverlayText, 3000);
}
#pragma warning(default : WARN_THROWS_ALTHOUGH_NOEXCEPT)

bool Controller::updateResizedImg(const IResizedImg& resizedImg_) noexcept {
  const ResizedImg& paramVal = dynamic_cast<const ResizedImg&>(resizedImg_);
  const bool result =
      !resizedImg || (dynamic_cast<const ResizedImg&>(*resizedImg) != paramVal);

  if (result)
    resizedImg = make_unique<const ResizedImg>(paramVal);
  comp.setReference(resizedImg->get());
  if (result)
    comp.resize();

  return result;
}

void Controller::showResultedImage(double completionDurationS) const noexcept {
  // Display the result at the end of the transformation
  comp.setResult(t.getResult());
  if (completionDurationS > 0.) {
    reportDuration("The transformation took", completionDurationS);
    t.setDuration(completionDurationS);
  }
}

Comparator::Comparator() noexcept : CvWin("Pic2Sym") {
  content(noImage);

  extern const int Comparator_trackMax;
  extern const String Comparator_transpTrackName;
  extern const string Comparator_initial_title, Comparator_statusBar;

  createTrackbar(Comparator_transpTrackName, winName(), &trackPos,
                 Comparator_trackMax, &Comparator::updateTransparency,
                 reinterpret_cast<void*>(this));
  Comparator::updateTransparency(trackPos,
                                 reinterpret_cast<void*>(this));  // mandatory

  // Default initial configuration
  setPos(0, 0);
  permitResize(false);
  setTitle(Comparator_initial_title);
  setStatus(Comparator_statusBar);
}

void Comparator::resize() const noexcept {
  // Comparator window is to be placed within 1024x768 top-left part of the
  // screen
  static constexpr double
      HEIGHT_TITLE_TOOLBAR_SLIDER_STATUS = 70,
      WIDTH_LATERAL_BORDERS = 4,
      H_NUMERATOR =
          768 -
          HEIGHT_TITLE_TOOLBAR_SLIDER_STATUS,  // desired max height -
                                               // HEIGHT_TITLE_TOOLBAR_SLIDER_STATUS
      W_NUMERATOR =
          1024 -
          WIDTH_LATERAL_BORDERS;  // desired max width - WIDTH_LATERAL_BORDERS

  // Resize window to preserve the aspect ratio of the loaded image,
  // while not enlarging it, nor exceeding 1024 x 768
  const int h = initial.rows, w = initial.cols;
  const double k = min(1., min(H_NUMERATOR / h, W_NUMERATOR / w));
  const int winHeight = (int)round(k * h + HEIGHT_TITLE_TOOLBAR_SLIDER_STATUS),
            winWidth = (int)round(k * w + WIDTH_LATERAL_BORDERS);

  CvWin::resize(winWidth, winHeight);
}

extern const Size CmapInspect_pageSz;

namespace {
const String CmapInspectWinName = "Charmap View";

/// type of a function extracting the negative mask from an iterator
template <typename Iterator>
using NegSymExtractor = std::function<const Mat(const typename Iterator&)>;

#pragma warning(disable : WARN_THROWS_ALTHOUGH_NOEXCEPT)
/**
Creates a page from the cmap to be displayed within the Cmap View.
@param it iterator to the first element to appear on the page
@param itEnd iterator after the last element to appear on the page
@param negSymExtractor function extracting the negative mask from each iterator
@param content the resulted page as a matrix (image)
@param grid the 'hive' for the glyphs to be displayed
@param fontSz provides the font size
@param areClustersIgnored adapts the display of clusters depending on their
importance
@param clusterOffsets where does each cluster start
@param idxOfFirstSymFromPage index of the first symbol to be displayed on the
page

@throw invalid_argument if idxOfFirstSymFromPage > max(clusterOffsets)

Exception to be only reported, not handled
*/
template <typename Iterator>
void populateGrid(Iterator it,
                  Iterator itEnd,
                  NegSymExtractor<Iterator> negSymExtractor,
                  Mat& content,
                  const Mat& grid,
                  int fontSz,
                  bool areClustersIgnored,
                  const set<unsigned>& clusterOffsets = {},
                  unsigned idxOfFirstSymFromPage = UINT_MAX) noexcept {
  content = grid.clone();
  const unsigned symsToShow = (unsigned)distance(it, itEnd);
  const int cellSide = 1 + fontSz, height = CmapInspect_pageSz.height,
            width = CmapInspect_pageSz.width;

  // Place each 'negative' glyph within the grid
  for (int r = cellSide; it != itEnd && r < height; r += cellSide) {
    const Range rowRange(r - fontSz, r);
    for (int c = cellSide; it != itEnd && c < width; c += cellSide, ++it) {
      const vector<Mat> glyphChannels(3, negSymExtractor(it));
      Mat glyphAsIfColor, region(content, rowRange, Range(c - fontSz, c));
      merge(glyphChannels, glyphAsIfColor);
      glyphAsIfColor.copyTo(region);
    }
  }

  if (clusterOffsets.empty() || UINT_MAX == idxOfFirstSymFromPage)
    return;

  // Display cluster limits if last 2 parameters provide this information
  const auto showMark = [&](unsigned offsetNewCluster, bool endMark) noexcept {
    static const Vec3b ClusterMarkColor(0U, 0U, 255U),
        ClustersEndMarkColor(128U, 0U, 64U);

    const unsigned symsInArow = (unsigned)((width - 1) / cellSide);
    const auto [posQuot, posRem] = div((int)offsetNewCluster, (int)symsInArow);
    const int r = posQuot * cellSide + 1;
    const int c = posRem * cellSide;
    const Vec3b splitColor = endMark ? ClustersEndMarkColor : ClusterMarkColor;
    if (areClustersIgnored) {  // use dashed lines as splits
      const Point up(c, r), down(c, r + fontSz - 1);
      const int dashLen =
          (fontSz + 4) /
          5;  // ceiling(fontSz/5) - at most 3 dashes with 2 breaks in between
      LineIterator lit(content, up, down, 4);
      for (int idx = 0, lim = lit.count; idx < lim; ++idx, ++lit) {
        if (0 == (1 & (idx / dashLen)))
          content.at<Vec3b>(lit.pos()) = splitColor;
      }

    } else {  // use filled lines as splits
      const Mat clusterMark(fontSz, 1, CV_8UC3, splitColor);
      clusterMark.copyTo((const Mat&)content.col(c).rowRange(r, r + fontSz));
    }
  };

  const auto itBegin = clusterOffsets.cbegin();
  if (const unsigned firstClusterSz = *std::next(itBegin) - *itBegin;
      firstClusterSz < 2U) {
    // show the end mark before the 1st symbol to signal there are no
    // non-trivial clusters
    showMark(0U, true);
    return;
  }

  auto itCo = clusterOffsets.lower_bound(idxOfFirstSymFromPage);
  if (itCo == clusterOffsets.cend())
    THROW_WITH_VAR_MSG(
        __FUNCTION__ " - the provided parameter idxOfFirstSymFromPage=" +
            to_string(idxOfFirstSymFromPage) +
            " is above the range covered by the clusterOffsets!",
        invalid_argument);

  if (itCo != itBegin) {  // true if the required page is not the 1st one
    if (const unsigned prevClustSz = *itCo - *std::prev(itCo); prevClustSz < 2U)
      // When even the last cluster on the previous page was trivial,
      // this page and following ones will contain only trivial clusters.
      // So, nothing to mark, therefore just leave.
      return;
  }

  // Here the size of previous cluster is >= 2
  for (unsigned offsetNewCluster = *itCo - idxOfFirstSymFromPage;
       offsetNewCluster < symsToShow; ++itCo) {
    const unsigned curClustSz = *std::next(itCo) - *itCo;
    const bool firstTrivialCluster = curClustSz < 2U;

    // mark cluster beginning or the end of non-trivial clusters
    showMark(offsetNewCluster, firstTrivialCluster);
    if (firstTrivialCluster)
      break;  // stop marking trivial clusters

    offsetNewCluster += curClustSz;
  }
}
#pragma warning(default : WARN_THROWS_ALTHOUGH_NOEXCEPT)
}  // anonymous namespace

void CmapInspect::populateGrid(const ICmapPerspective::VPSymDataCItPair& itPair,
                               const set<unsigned>& clusterOffsets,
                               unsigned idxOfFirstSymFromPage) noexcept {
  ICmapPerspective::VPSymDataCIt it = itPair.first, itEnd = itPair.second;
  ::populateGrid(
      it, itEnd,
      (NegSymExtractor<ICmapPerspective::VPSymDataCIt>)  // conversion
          [](const ICmapPerspective::VPSymDataCIt& iter) noexcept->Mat {
            if ((*iter)->isRemovable())
              return 255U - (*iter)->getNegSym();
            return (*iter)->getNegSym();
          },
      content(), grid, (int)fontSz, !cmapPresenter.areClustersUsed(),
      clusterOffsets, idxOfFirstSymFromPage);
}

void CmapInspect::showUnofficial1stPage(
    vector<Mat>& symsOn1stPage,
    atomic_flag& updating1stCmapPage,
    LockFreeQueue& updateSymsActionsQueue) noexcept {
  std::shared_ptr<Mat> unofficial = std::make_shared<Mat>();
  ::populateGrid(CBOUNDS(symsOn1stPage),
                 (NegSymExtractor<vector<Mat>::const_iterator>)  // conversion
                     [](const vector<Mat>::const_iterator& iter) noexcept {
                       return *iter;
                     },
                 *unofficial, grid, (int)fontSz,
                 !cmapPresenter.areClustersUsed());

  symsOn1stPage.clear();  // discard values now

  // If the main worker didn't register its intention to render already the
  // official 1st cmap page display the unofficial early version. Otherwise just
  // leave
  if (!updating1stCmapPage
           .test_and_set()) {  // holds the value on true after acquiring it
    // Creating local copies that can be passed by value to
    // Unofficial1stPageCmap's parameter
    atomic_flag* pUpdating1stCmapPage = &updating1stCmapPage;
    const String winNameCopy = winName();

#pragma warning(disable : WARN_EXPLICIT_NEW_OR_DELETE)
    updateSymsActionsQueue.push(new BasicUpdateSymsAction([
      pUpdating1stCmapPage, winNameCopy,
      unofficial
    ]() noexcept {
      imshow(winNameCopy, *unofficial);

      pUpdating1stCmapPage->clear();  // puts its value on false, to be acquired
                                      // by the official version publisher
    }));
#pragma warning(default : WARN_EXPLICIT_NEW_OR_DELETE)
  }
}

CmapInspect::CmapInspect(const IPresentCmap& cmapPresenter_,
                         const ISelectSymbols& symsSelector,
                         const unsigned& fontSz_) noexcept
    : CvWin(CmapInspectWinName),
      cmapPresenter(cmapPresenter_),
      fontSz(fontSz_) {
  extern const String CmapInspect_pageTrackName;
  content(grid = createGrid());

  /*
  `CmapInspect::updatePageIdx` from below needs to work with a `void*` of actual
  type `ICmapInspect*`. But, passing `reinterpret_cast<void*>(this)` will fail,
  since `this` is a `CmapInspect*` and the actual address of
  `(ICmapInspect*)this` is different from `this` because `ICmapInspect` isn't
  the top inherited interface.
  */
  void* const thisAsICmapInspectPtr =
      reinterpret_cast<void*>((ICmapInspect*)this);

  createTrackbar(CmapInspect_pageTrackName, winName(), &page, 1,
                 &CmapInspect::updatePageIdx, thisAsICmapInspectPtr);
  CmapInspect::updatePageIdx(page, thisAsICmapInspectPtr);  // mandatory call

  setPos(424, 0);       // Place cmap window on x axis between 424..1064
  permitResize(false);  // Ensure the user sees the symbols exactly their size

  // Support for investigating selected symbols from one or more charmaps:
  // - mouse moves over the Charmap will display in the status bar the code of
  // the pointed symbol
  // - Ctrl + left mouse click will append the pointed symbol to the current
  // list
  // - left mouse double-click will save the current list and then it will clear
  // it
  setMouseCallback(
      CmapInspectWinName,
      [](int event, int x, int y, int flags, void* userdata) noexcept {
        const ISelectSymbols* pss = static_cast<ISelectSymbols*>(userdata);
        if (event == EVENT_MOUSEMOVE) {  // Mouse move
          if (const ISymData* psd = pss->pointedSymbol(x, y))
            pss->displaySymCode(psd->getCode());
        } else if ((event == EVENT_LBUTTONDBLCLK) &&
                   ((flags & EVENT_FLAG_CTRLKEY) ==
                    0)) {  // Ctrl key not pressed and left mouse double-click
          pss->symbolsReadyToInvestigate();
        } else if ((event == EVENT_LBUTTONUP) &&
                   ((flags & EVENT_FLAG_CTRLKEY) !=
                    0)) {  // Ctrl key pressed and left mouse click
          if (const ISymData* psd = pss->pointedSymbol(x, y))
            pss->enlistSymbolForInvestigation(*psd);
        }
      },

      // Pass &symsSelector (which is const ISelectSymbols*) to 'void* userdata'
      // from above.
      // The userdata is reconverted back there: 'const ISelectSymbols* pss'
      reinterpret_cast<void*>(const_cast<ISelectSymbols*>(&symsSelector)));
}

Mat CmapInspect::createGrid() noexcept {
  static const Scalar GridColor(255U, 200U, 200U);

  Mat emptyGrid(CmapInspect_pageSz, CV_8UC3, Scalar::all(255U));

  cellSide = 1U + fontSz;
  symsPerRow = (((unsigned)CmapInspect_pageSz.width - 1U) / cellSide);
  symsPerPage =
      symsPerRow * (((unsigned)CmapInspect_pageSz.height - 1U) / cellSide);

  // Draws horizontal & vertical cell borders
  for (int i = 0; i < CmapInspect_pageSz.width; i += cellSide)
    emptyGrid.col(i).setTo(GridColor);
  for (int i = 0; i < CmapInspect_pageSz.height; i += cellSide)
    emptyGrid.row(i).setTo(GridColor);
  return emptyGrid;
}

void CmapPerspective::reset(
    const VSymData& symsSet,
    const vector<vector<unsigned>>& symsIndicesPerCluster_) noexcept {
  assert(!symsSet.empty());                 // Adviced
  assert(!symsIndicesPerCluster_.empty());  // Adviced

  const auto symsCount = symsSet.size(),
             clustersCount = symsIndicesPerCluster_.size();

  vector<const vector<unsigned>*> symsIndicesPerCluster(clustersCount);
  size_t clustIdx = 0ULL;
  for (const vector<unsigned>& clusterMembers : symsIndicesPerCluster_)
    symsIndicesPerCluster[clustIdx++] = &clusterMembers;

  // View the clusters in descending order of their size

  // Typically, there are only a few clusters larger than 1 element.
  // This partition separates the actual formed clusters from one-of-a-kind
  // elements leaving less work to perform to the sort executed afterwards.
  // Using the stable algorithm version, to preserve avgPixVal sorting
  // set by ClusterEngine::process
  auto itFirstClusterWithOneItem = stable_partition(
      BOUNDS(symsIndicesPerCluster), [](const vector<unsigned>* a) noexcept {
        // Place actual clusters at the beginning of the vector
        return a->size() > 1ULL;
      });

  // Sort non-trivial clusters in descending order of their size.
  // Using the stable algorithm version, to preserve avgPixVal sorting
  // set by ClusterEngine::process
  stable_sort(begin(symsIndicesPerCluster), itFirstClusterWithOneItem, [
  ](const vector<unsigned>* a, const vector<unsigned>* b) noexcept {
    return a->size() > b->size();
  });

  pSyms.resize(symsCount);

  size_t offset = 0ULL;
  clusterOffsets.clear();
  clusterOffsets.insert((unsigned)offset);
  for (const vector<unsigned>* clusterMembers : symsIndicesPerCluster) {
    const auto prevOffset = offset;
    offset += clusterMembers->size();
    clusterOffsets.emplace_hint(end(clusterOffsets), (unsigned)offset);
    for (size_t idxPSyms = prevOffset, idxCluster = 0ULL; idxPSyms < offset;
         ++idxPSyms, ++idxCluster)
      pSyms[idxPSyms] = symsSet[(size_t)(*clusterMembers)[idxCluster]].get();
  }
}

ICmapPerspective::VPSymDataCItPair CmapPerspective::getSymsRange(
    unsigned from,
    unsigned count) const noexcept {
  const auto sz = pSyms.size();
  const VPSymDataCIt itEnd = pSyms.cend();
  if ((size_t)from >= sz)
    return make_pair(itEnd, itEnd);

  const VPSymDataCIt itStart = next(pSyms.cbegin(), from);
  const auto maxCount = sz - (size_t)from;
  if ((size_t)count >= maxCount)
    return make_pair(itStart, itEnd);

  return make_pair(itStart, next(itStart, count));
}

const set<unsigned>& CmapPerspective::getClusterOffsets() const noexcept {
  return clusterOffsets;
}

#endif  // UNIT_TESTING not defined
