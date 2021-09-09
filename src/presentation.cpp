/******************************************************************************
 The application Pic2Sym approximates images by a
 grid of colored symbols with colored backgrounds.

 Copyrights from the libraries used by the program:
 - (c) 2003-2021 Boost (www.boost.org)
     License: doc/licenses/Boost.lic
     http://www.boost.org/LICENSE_1_0.txt
 - (c) 2015-2021 OpenCV (www.opencv.org)
     License: doc/licenses/OpenCV.lic
     http://opencv.org/license/
 - (c) 1996-2021 The FreeType Project (www.freetype.org)
     License: doc/licenses/FTL.txt
     http://git.savannah.gnu.org/cgit/freetype/freetype2.git/plain/docs/FTL.TXT
 - (c) 1997-2021 OpenMP Architecture Review Board (www.openmp.org)
   (c) Microsoft Corporation (implementation for OpenMP C/C++ v2.0 March 2002)
     See: https://msdn.microsoft.com/en-us/library/8y6825x5.aspx
 - (c) 1995-2021 zlib software (Jean-loup Gailly and Mark Adler - www.zlib.net)
     License: doc/licenses/zlib.lic
     http://www.zlib.net/zlib_license.html
 - (c) 2015-2021 Microsoft Guidelines Support Library - github.com/microsoft/GSL
     License: doc/licenses/MicrosoftGSL.lic
     https://raw.githubusercontent.com/microsoft/GSL/main/LICENSE


 (c) 2016-2021 Florin Tulba <florintulba@yahoo.com>

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
// This keeps precompiled.h first; Otherwise header sorting might move it

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
#include "symsChangeIssues.h"
#include "transform.h"
#include "updateSymSettingsBase.h"
#include "updateSymsActions.h"
#include "views.h"
#include "warnings.h"

#pragma warning(push, 0)

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <Windows.h>

#include <future>
#include <ranges>
#include <thread>
#include <tuple>

#include <gsl/gsl>

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
using namespace gsl;

namespace pic2sym {

using ui::BasicUpdateSymsAction;

// UnitTesting project launches an instance of Pic2Sym to visualize any
// discovered issues, so it won't refer viewMismatches and viewMisfiltered from
// below. But it will start a Pic2Sym instance that will do such calls.
#ifndef UNIT_TESTING

/// When pressing ESC, prompt the user if he/she wants to exit
void leaveWhenDemanded() noexcept {
  for (;;) {
    if (EscKeyCode == waitKey() &&
        IDYES == MessageBox(nullptr, L"Do you want to leave the application?",
                            L"Question",
                            MB_ICONQUESTION | MB_YESNOCANCEL | MB_TASKMODAL |
                                MB_SETFOREGROUND))
      break;
  }
}

void viewMismatches(string_view testTitle, const Mat& mismatches) noexcept {
  const int twiceTheRows{mismatches.rows};
  const int rows{twiceTheRows >> 1};
  const int cols{mismatches.cols};

  // upper half is the reference
  const Mat reference{mismatches.rowRange(0, rows)};

  // lower half is the result
  const Mat result{mismatches.rowRange(rows, twiceTheRows)};

  // Comparator window size should stay within ~ 800x600
  // Enlarge up to 3 times if resulting rows < 600.
  // Enlarge also when resulted width would be less than 140 (width when the
  // slider is visible)
  const double resizeFactor{max(140. / cols, min(600. / rows, 3.))};

  ostringstream oss;
  oss << "View mismatches for " << testTitle;
  const string title(oss.str());

  ui::Comparator comp;
  comp.permitResize();

  waitKey(1);  // ensures permit resize is actually considered before resizing

  comp.resize(4 + narrow_cast<int>(ceil(cols * resizeFactor)),
              70 + narrow_cast<int>(ceil(rows * resizeFactor)));
  comp.setTitle(title.c_str());
  comp.setStatus("Press Esc to close this window");
  comp.setReference(reference);
  comp.setResult(result, 90);  // Emphasize the references

  leaveWhenDemanded();
}

void viewMisfiltered(string_view testTitle, const Mat& misfiltered) noexcept {
  const String winName{testTitle};
  namedWindow(winName);
  setWindowProperty(winName, cv::WND_PROP_AUTOSIZE, cv::WINDOW_NORMAL);
  imshow(winName, misfiltered);
  displayStatusBar(winName, "Press Esc to close this window");

  leaveWhenDemanded();
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
static void viewSymWeightsHistogram(const syms::VPixMapSym& theSyms
                                    [[maybe_unused]]) noexcept {
#ifndef UNIT_TESTING
  vector<double> symSums;
  for (const unique_ptr<const syms::IPixMapSym>& pms : theSyms)
    symSums.push_back(pms->getAvgPixVal());

  static constexpr size_t MaxBinHeight{256ULL};
  const size_t binsCount{min(256ULL, size(symSums))};
  const double smallestSum{symSums.front()};
  const double largestSum{symSums.back()};
  const double sumsSpan{largestSum - smallestSum};
  vector<size_t> hist(binsCount, 0ULL);
  const auto itBegin = begin(symSums);
  ptrdiff_t prevCount{};
  for (size_t bin{}; bin < binsCount; ++bin) {
    const auto it = ranges::upper_bound(
        symSums, smallestSum + sumsSpan * (bin + 1.) / binsCount);
    const auto curCount = distance(itBegin, it);
    hist[bin] = size_t(curCount - prevCount);
    prevCount = curCount;
  }
  const double maxBinValue{(double)*ranges::max_element(hist)};
  for (size_t& binValue : hist)
    binValue = (size_t)round(binValue * MaxBinHeight / maxBinValue);
  Mat histImg{(int)MaxBinHeight, (int)binsCount, CV_8UC1, Scalar{255.}};
  for (size_t bin{}; bin < binsCount; ++bin)
    if (hist[bin] > 0ULL)
      histImg.rowRange(int(MaxBinHeight - hist[bin]), (int)MaxBinHeight)
          .col((int)bin) = 0U;
  imshow("histogram", histImg);
  waitKey(1);

#endif  // UNIT_TESTING
}

string syms::FontEngine::getFontType() const noexcept(!UT) {
  ostringstream oss;
  oss << getFamily() << '_' << getStyle() << '_' << getEncoding();
  // getEncoding() throws logic_error for incomplete font configuration
  // The exception propagates only in UnitTesting

  return oss.str();
}

#pragma warning(disable : WARN_THROWS_ALTHOUGH_NOEXCEPT)
string match::MatchEngine::getIdForSymsToUse() const noexcept(!UT) {
  const unsigned sz{cfg->getSS().getFontSz()};
  EXPECTS_OR_REPORT_AND_THROW(
      cfg::ISettings::isFontSizeOk(sz), domain_error,
      HERE.function_name() + " read invalid font size: "s + to_string(sz));

  ostringstream oss;
  oss << fe->getFontType() << '_' << sz;
  // getFontType() throws logic_error for incomplete font configuration
  // The exception propagates only in UnitTesting

  return oss.str();
}
#pragma warning(default : WARN_THROWS_ALTHOUGH_NOEXCEPT)

string cfg::MatchSettings::toString(bool verbose) const noexcept {
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

void transform::Transformer::updateStudiedCase(int rows, int cols) noexcept {
  const cfg::IMatchSettings& ms = cfg->getMS();
  ostringstream oss;

  // no extension yet
  oss << img->name() << '_' << me->getIdForSymsToUse() << '_'
      << ms.toString(false) << '_' << cols << '_' << rows;
  studiedCase =
      oss.str();  // this text is included in the result & trace file names
}

#if defined(MONITOR_SKIPPED_MATCHING_ASPECTS) && !defined(UNIT_TESTING)

void match::MatchAssessorSkip::reportSkippedAspects() const noexcept {
  cout << "\nTransformation finished. "
          "Reporting skipped aspects from a total of "
       << totalIsBetterMatchCalls << " isBetterMatch calls:\n";
  for (size_t i{}; i < enabledAspectsCount; ++i) {
    if (!skippedAspects[i])
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

#endif  // defined(MONITOR_SKIPPED_MATCHING_ASPECTS) && !defined(UNIT_TESTING)

string Controller::textForCmapStatusBar(
    unsigned upperSymsCount /* = 0U*/) const noexcept {
  // crashes the program if incomplete font configuration
  const string& enc = fe->getEncoding();

  assert(fe->getFamily());  // is empty in the worst case
  assert(fe->getStyle());   // is empty in the worst case

  ostringstream oss;
  // crashes the program if incomplete font configuration
  oss << "Font type '" << fe->getFamily() << ' ' << fe->getStyle() << "', size "
      << cfg->getSS().getFontSz() << ", encoding " << quoted(enc, '\'') << " : "
      << (upperSymsCount ? upperSymsCount
                         : narrow_cast<unsigned>(fe->symsSet().size()))
      << " symbols";
  return oss.str();
}

string Controller::textHourGlass(const std::string& prefix,
                                 double progress) const noexcept {
  ostringstream oss;
  oss << prefix << " (" << fixed << setprecision(0) << progress * 100. << "%)";
  return oss.str();
}

extern const bool ViewSymWeightsHistogram;

void Controller::symbolsChanged() {
  EXPECTS_OR_REPORT_AND_THROW_CONST_MSG(
      pCmi, logic_error,
      HERE.function_name() + " called before ensureExistenceCmapInspect()!"s);

  // Timing the update.
  // timer's destructor will report duration and will close the hourglass
  // window. These actions will be launched in this GUI updating thread, after
  // processing all previously registered actions.
  Timer timer{glyphsProgressTracker->createTimerForGlyphs()};

  // just make sure updatingSymbols gets set; no test, no wait
  ignore = updatingSymbols.test_and_set();

  updating1stCmapPage.clear();

  pCmi->setBrowsable(false);

  // Starting a thread to perform the actual change of the symbols,
  // while preserving this thread for GUI updating
  updatesSymbols = jthread{[&]() noexcept {
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

    auto _ = finally([&]() noexcept {
      /*
      Wait for an eventual thread created to order showing an unofficial
      version of the 1st Cmap page.

      If none of the cmaps so far contained enough symbols to fill a page,
      or if the ordersDisplayOfUnofficial1stCmapPage refers to a previous cmap,
      which was already waited for, the valid() test will be false.

      Otherwise - when current cmap has enough symbols, the
      ordersDisplayOfUnofficial1stCmapPage will have a fresh value, which can be
      waited for.
      */
      if (ordersDisplayOfUnofficial1stCmapPage.valid())
        ordersDisplayOfUnofficial1stCmapPage.wait();

      // Signal task completion
      updatingSymbols.clear();
      updatingSymbols.notify_one();
    });

    try {
      fe->setFontSz(cfg->getSS().getFontSz());
      me->updateSymbols();

    } catch (const syms::NormalSymsLoadingFailure& excObj) {
      // capture it in one thread, then pack it for the other thread
      const string errText{excObj.what()};
      // An exception with the same errText will be thrown in the main thread
      // when executing next action

      updateSymsActionsQueue.push(
          owner<BasicUpdateSymsAction*>{new BasicUpdateSymsAction{
              [errText] { throw syms::NormalSymsLoadingFailure{errText}; }}});

      return;

    } catch (const syms::TinySymsLoadingFailure& excObj) {
      // capture it in one thread, then pack it for the other thread
      const string errText{excObj.what()};
      // An exception with the same errText will be thrown in the main thread
      // when executing next action

      updateSymsActionsQueue.push(
          owner<BasicUpdateSymsAction*>{new BasicUpdateSymsAction{
              [errText] { throw syms::TinySymsLoadingFailure{errText}; }}});

      return;

    } catch (const ui::AbortedJob&) {
      // the aborted operation reports already to cerr; nothing to do
    }

    /*
    Symbols have been changed in the model. Only GUI must be updated
    The GUI will display the 1st cmap page.
    This must happen after an eventual early preview of it:
    - we need to wait for the preview to finish if it started before this
    point
    - we have to prevent an available preview to be displayed after the
    official version
    */
    updating1stCmapPage.wait(true);
    while (updating1stCmapPage.test_and_set())
      updating1stCmapPage.wait(true);  // Loops until it is cleared externally
  }};

  // [j]threads don't have a builtin mechanism to just check if they finished.
  // future-s however do have it: valid() && wait_for(0ms)==ready
  future<void> abortHandler;

  // lambda used twice below
  const auto performRegisteredActions = [&] {
    // Only read pAction pointers, but use it as a unique_ptr
    owner<ui::IUpdateSymsAction*> pAction = nullptr;
    unique_ptr<ui::IUpdateSymsAction> action;

    const auto discardRemainingActions = [&] {
      while (updateSymsActionsQueue.pop(pAction))
        action.reset(pAction);

      action.reset();
    };

    // Perform available actions, if not requested to abort
    while (updateSymsActionsQueue.pop(pAction)) {
      action.reset(pAction);  // own a different pointer and free previous one

      if (leaving.test()) {  // Test if the user wants to exit
        // Ensure updatesSymbols stops
        if (updatesSymbols.joinable()) {
          glyphsUpdateMonitor->abort();
          updatesSymbols.request_stop();
          updatesSymbols.join();
        }

        discardRemainingActions();
        return;
      }

      // Allow the user to abort and leave the app.
      // Wait for the user to answer to the async question without prompting
      // him/her again
      if ((!abortHandler.valid() ||
           abortHandler.wait_for(0ms) == future_status::ready) &&
          EscKeyCode == waitKey(1)) {
        // Prompt and wait for the user answer asynchronously.
        // waitKey had to remain within the main thread.
        abortHandler = async(launch::async, [&]() noexcept {
          if (IDYES == MessageBox(nullptr,
                                  L"Do you want to abort changing the symbols "
                                  L"and leave the application?",
                                  L"Question",
                                  MB_ICONQUESTION | MB_YESNOCANCEL |
                                      MB_TASKMODAL | MB_SETFOREGROUND)) {
            ignore = leaving.test_and_set();
            leaving.notify_all();
          }
        });
      }

#pragma warning(disable : WARN_SEH_NOT_CAUGHT)
      try {
        action->perform();
      } catch (...) {
        discardRemainingActions();

        // Postponing the processing of the exception
        // See above NormalSymsLoadingFailure & TinySymsLoadingFailure being
        // thrown
        throw;
      }
#pragma warning(default : WARN_SEH_NOT_CAUGHT)
    }
  };

  // Loop while work is carried out.
  // updatingSymbols was set before the thread
  // and it gets cleared only from the thread
  while (updatingSymbols.test_and_set())
    performRegisteredActions();  // no need to wait when there are tasks

  // perform/discard any remaining actions
  performRegisteredActions();

  if (leaving.test())
    reportAndThrow<syms::SymsChangeInterrupted>(
        "Symbols set change interrupted by user! Leaving ...");

  // Official versions of the status bar and the 1st cmap page
  pCmi->setStatus(textForCmapStatusBar());
  pCmi->updatePagesCount(narrow_cast<unsigned>(fe->symsSet().size()));
  pCmi->setBrowsable();

  if (ViewSymWeightsHistogram)
    viewSymWeightsHistogram(fe->symsSet());

  updatingSymbols.clear();
  updating1stCmapPage.clear();
}

#ifndef UNIT_TESTING

Controller::~Controller() noexcept {
  int key{-EscKeyCode};  // anything else but ESC

  // This is the main event loop, when the app is idle
  // (not handling any user requests yet)
  while (!leaving.test()) {
    /*
    Any delay provided to this waitKey becomes useless if it processes an event
    which uses a subsequent waitKey with a specified new delay.
    I reckon the timer of the first waitKey gets overwritten by the one from the
    second waitKey, which expires and lets the first waitKey without a timer at
    all.
    To interrupt the first waitKey exceptions are required.

    Other waitKey is needed while processing a user request.
    That waitKey needs to be within the main thread, as well.
    */
    try {
      key = waitKey();  // Don't rely on a delay here! See comments above why.
    } catch (const syms::SymsChangeInterrupted&) {
      continue;
    }

    if (EscKeyCode != key)
      continue;

    // Getting the user answer can be synchronous, as the app was idle
    if (IDYES == MessageBox(nullptr, L"Do you want to leave the application?",
                            L"Question",
                            MB_ICONQUESTION | MB_YESNOCANCEL | MB_TASKMODAL |
                                MB_SETFOREGROUND)) {
      ignore = leaving.test_and_set();
      leaving.notify_all();
    }
  }

  destroyAllWindows();
}

void Controller::display1stPageIfFull(const syms::VPixMapSym& syms) const
    noexcept(!UT) {
  EXPECTS_OR_REPORT_AND_THROW_CONST_MSG(
      pCmi, logic_error,
      HERE.function_name() + " called before ensureExistenceCmapInspect()!"s);

  if (narrow_cast<unsigned>(size(syms)) != pCmi->getSymsPerPage())
    return;

  // Copy all available symbols from the 1st page in current thread
  // to ensure they get presented in the original order by the other thread
  static vector<Mat> symsOn1stPage;
  symsOn1stPage.clear();  // ensure empty each time!
  symsOn1stPage.reserve(size(syms));
  const auto fontSz = getFontSize();
  for (const unique_ptr<const syms::IPixMapSym>& pms : syms)
    symsOn1stPage.emplace_back(pms->toMat(fontSz, true));

  // Build the unofficial version of the 1st cmap page asynchronously
  ordersDisplayOfUnofficial1stCmapPage = async(launch::async, [&]() noexcept {
    pCmi->showUnofficial1stPage(symsOn1stPage, updating1stCmapPage,
                                updateSymsActionsQueue);
  });
}

#pragma warning(disable : WARN_THROWS_ALTHOUGH_NOEXCEPT)
void Controller::hourGlass(double progress,
                           const string& title /* = ""*/,
                           bool async /* = false*/) const noexcept(!UT) {
  EXPECTS_OR_REPORT_AND_THROW_CONST_MSG(
      progress >= -Eps && progress <= EpsPlus1, invalid_argument,
      HERE.function_name() + " needs progress to be between 0 .. 1!"s);

  static const String waitWin{"Please Wait!"s};

  if (async) {  // enqueue the call
    updateSymsActionsQueue.push(
        owner<BasicUpdateSymsAction*>{new BasicUpdateSymsAction{
            [&, progress, title_ = title] { hourGlass(progress, title_); }}});

  } else {  // direct call or one async call due now
    if (!progress) {
      // no status bar, nor toolbar
      namedWindow(waitWin, cv::WINDOW_GUI_NORMAL);
      moveWindow(waitWin, 0, 400);

    } else if (progress == 1.) {
      destroyWindow(waitWin);

    } else {
      ostringstream oss;
      if (title.empty())
        oss << waitWin;
      else
        oss << title;
      const string hourGlassText{textHourGlass(oss.str(), progress)};
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
  EXPECTS_OR_REPORT_AND_THROW_CONST_MSG(
      pCmi, logic_error,
      HERE.function_name() + " called before ensureExistenceCmapInspect()!"s);

  const string newStatusBarText{textForCmapStatusBar(upperSymsCount) + suffix};
  if (async) {  // placing a task in the queue for the GUI updating thread
    updateSymsActionsQueue.push(
        owner<BasicUpdateSymsAction*>{new BasicUpdateSymsAction{
            [&, newStatusBarText] { pCmi->setStatus(newStatusBarText); }}});
  } else {  // direct call or one async call due now
    pCmi->setStatus(newStatusBarText);
  }
}
#pragma warning(default : WARN_THROWS_ALTHOUGH_NOEXCEPT)

#pragma warning(disable : WARN_THROWS_ALTHOUGH_NOEXCEPT)
void Controller::reportDuration(string_view text, double durationS) const
    noexcept(!UT) {
  EXPECTS_OR_REPORT_AND_THROW_CONST_MSG(
      pCmi, logic_error,
      HERE.function_name() + " called before ensureExistenceCmapInspect()!"s);

  ostringstream oss;
  oss << text << ' ' << durationS << " s!";
  const string cmapOverlayText{oss.str()};
  cout << '\n' << cmapOverlayText << '\n' << endl;
  pCmi->setOverlay(cmapOverlayText, 3'000);
}
#pragma warning(default : WARN_THROWS_ALTHOUGH_NOEXCEPT)

bool Controller::updateResizedImg(
    const input::IResizedImg& resizedImg_) noexcept {
  using namespace input;
  const ResizedImg& paramVal = dynamic_cast<const ResizedImg&>(resizedImg_);
  const bool result{!resizedImg ||
                    (dynamic_cast<const ResizedImg&>(*resizedImg) != paramVal)};

  if (result)
    resizedImg = make_unique<const ResizedImg>(paramVal);
  comp->setReference(resizedImg->get());
  if (result)
    comp->resize();

  return result;
}

void Controller::showResultedImage(double completionDurationS) const noexcept {
  // Display the result at the end of the transformation
  comp->setResult(t->getResult());
  if (completionDurationS > 0.) {
    reportDuration("The transformation took", completionDurationS);
    t->setDuration(completionDurationS);
  }
}

extern const int Comparator_trackMax;
extern const String Comparator_transpTrackName;
extern const string Comparator_initial_title;
extern const string Comparator_statusBar;

ui::Comparator::Comparator() noexcept : ui::CvWin("Pic2Sym") {
  content(noImage);

  createTrackbar(Comparator_transpTrackName, winName(), &trackPos,
                 Comparator_trackMax, &Comparator::updateTransparency,
                 static_cast<void*>(this));
  Comparator::updateTransparency(trackPos,
                                 static_cast<void*>(this));  // mandatory

  // Default initial configuration
  setPos(90, 90);
  permitResize(false);
  setTitle(Comparator_initial_title);
  setStatus(Comparator_statusBar);
}

void ui::Comparator::resize() const noexcept {
  // Comparator window is to be placed within 1024x768 top-left part of the
  // screen
  static constexpr double HeightTitleToolbarSliderStatus{70};
  static constexpr double WidthLateralBorders{4};

  // desired max height - HeightTitleToolbarSliderStatus
  static constexpr double HeightNumerator{768 - HeightTitleToolbarSliderStatus};

  // desired max width - WidthLateralBorders
  static constexpr double WidthNumerator{1024 - WidthLateralBorders};

  // Resize window to preserve the aspect ratio of the loaded image,
  // while not enlarging it, nor exceeding 1024 x 768
  const int h{initial.rows};
  const int w{initial.cols};
  const double k{min(1., min(HeightNumerator / h, WidthNumerator / w))};
  const int winHeight{
      narrow_cast<int>(round(k * h + HeightTitleToolbarSliderStatus))};
  const int winWidth{narrow_cast<int>(round(k * w + WidthLateralBorders))};

  CvWin::resize(winWidth, winHeight);
}

extern const Size CmapInspect_pageSz;

namespace {
const String CmapInspectWinName{"Charmap View"};

/// Helper for showing a page on Cmap View
class GridHelper {
 public:
  /// type of a function extracting the negative mask pointed by an iterator
  template <typename Iterator>
  using NegSymExtractor = std::function<cv::Mat(const Iterator&)>;

  /**
  Prepares displaying a Cmap page
  @param result the resulted page as a matrix (image)
  @param grid the 'hive' for the glyphs to be displayed
  @param fontSz provides the font size
  @param clusterOffsets where does each cluster start
  */
  GridHelper(Mat& result,
             const Mat& grid,
             const int fontSz,
             const set<unsigned>& clusterOffsets = {}) noexcept
      : _result(&result),
        _grid(&grid),
        _fontSz(fontSz),
        _clusterOffsets(&clusterOffsets) {}
  GridHelper(const GridHelper&) = default;
  GridHelper(GridHelper&&) = delete;
  GridHelper& operator=(const GridHelper&) = default;
  void operator=(GridHelper&&) = delete;
  ~GridHelper() noexcept {}

  /// adapts the display of clusters depending on their importance
  GridHelper& ignoreClusters(bool confirmIgnore = true) noexcept {
    _ignoreClusters = confirmIgnore;
    return *this;
  }

  /// index of the first symbol to be displayed on the page
  GridHelper& idxOfFirstSymFromPage(unsigned idx) noexcept {
    _idxOfFirstSymFromPage = idx;
    return *this;
  }

#pragma warning(disable : WARN_THROWS_ALTHOUGH_NOEXCEPT)
  /**
  Creates a page from the cmap to be displayed within the Cmap View.
  @param symsRange range of elements to appear on the page
  @param negSymExtractor function extracting the negative mask pointed by each
  iterator

  @return the requested page from Cmap as reference to the result parameter from
  the constructor

  @throw invalid_argument if idxOfFirstSymFromPage > max(clusterOffsets)

  Exception to be only reported, not handled
  */
  template <ranges::forward_range R>
  Mat& populate(
      R&& symsRange,
      NegSymExtractor<ranges::iterator_t<R>> negSymExtractor) noexcept {
    *_result = _grid->clone();
    const unsigned symsToShow{narrow_cast<unsigned>(size(symsRange))};
    const int cellSide{1 + _fontSz};
    const int height{CmapInspect_pageSz.height};
    const int width{CmapInspect_pageSz.width};

    // Place each 'negative' glyph within the grid
    auto it = begin(symsRange);
    const auto itEnd = end(symsRange);
    for (int r{cellSide}; it != itEnd && r < height; r += cellSide) {
      const Range rowRange{r - _fontSz, r};
      for (int c{cellSide}; it != itEnd && c < width; c += cellSide, ++it) {
        const vector<Mat> glyphChannels(3, negSymExtractor(it));
        Mat glyphAsIfColor;
        Mat region{*_result, rowRange, Range{c - _fontSz, c}};
        merge(glyphChannels, glyphAsIfColor);
        glyphAsIfColor.copyTo(region);
      }
    }

    if (_clusterOffsets->empty() || UINT_MAX == _idxOfFirstSymFromPage)
      return *_result;

    // Display cluster limits if requested
    const auto showMark = [&](unsigned offsetNewCluster,
                              bool endMark) noexcept {
      static const Vec3b ClusterMarkColor{0U, 0U, 255U};
      static const Vec3b ClustersEndMarkColor{128U, 0U, 64U};

      const unsigned symsInArow{(unsigned)((width - 1) / cellSide)};
      const auto [posQuot, posRem] =
          div((int)offsetNewCluster, (int)symsInArow);
      const int r{posQuot * cellSide + 1};
      const int c{posRem * cellSide};
      const Vec3b splitColor{endMark ? ClustersEndMarkColor : ClusterMarkColor};
      if (_ignoreClusters) {  // use dashed lines as splits
        const Point up{c, r};
        const Point down{c, r + _fontSz - 1};

        // ceiling(_fontSz/5) - at most 3 dashes with 2 breaks in between
        const int dashLen{(_fontSz + 4) / 5};
        LineIterator lit{*_result, up, down, 4};
        for (int idx{}, lim{lit.count}; idx < lim; ++idx, ++lit) {
          if (!(1 & (idx / dashLen)))
            _result->at<Vec3b>(lit.pos()) = splitColor;
        }

      } else {  // use filled lines as splits
        const Mat clusterMark{_fontSz, 1, CV_8UC3, splitColor};
        clusterMark.copyTo(
            (const Mat&)_result->col(c).rowRange(r, r + _fontSz));
      }
    };

    const auto itBegin = _clusterOffsets->cbegin();
    if (const unsigned firstClusterSz{*std::next(itBegin) - *itBegin};
        firstClusterSz < 2U) {
      // show the end mark before the 1st symbol to signal there are no
      // non-trivial clusters
      showMark(0U, true);
      return *_result;
    }

    auto itCo = _clusterOffsets->lower_bound(_idxOfFirstSymFromPage);
    if (itCo == _clusterOffsets->cend())
      reportAndThrow<invalid_argument>(
          HERE.function_name() + " - the provided idxOfFirstSymFromPage="s +
          to_string(_idxOfFirstSymFromPage) +
          " is above the range covered by the clusterOffsets!"s);

    if (itCo != itBegin) {  // true if the required page is not the 1st one
      if (const unsigned prevClustSz{*itCo - *std::prev(itCo)};
          prevClustSz < 2U)
        // When even the last cluster on the previous page was trivial,
        // this page and following ones will contain only trivial clusters.
        // So, nothing to mark, therefore just leave.
        return *_result;
    }

    // Here the size of previous cluster is >= 2
    for (unsigned offsetNewCluster{*itCo - _idxOfFirstSymFromPage};
         offsetNewCluster < symsToShow; ++itCo) {
      const unsigned curClustSz{*std::next(itCo) - *itCo};
      const bool firstTrivialCluster{curClustSz < 2U};

      // mark cluster beginning or the end of non-trivial clusters
      showMark(offsetNewCluster, firstTrivialCluster);
      if (firstTrivialCluster)
        break;  // stop marking trivial clusters

      offsetNewCluster += curClustSz;
    }

    return *_result;
  }
#pragma warning(default : WARN_THROWS_ALTHOUGH_NOEXCEPT)

 private:
  not_null<Mat*> _result;      ///< the resulted page as a matrix (image)
  not_null<const Mat*> _grid;  ///< the 'hive' for the glyphs to be displayed
  int _fontSz;                 ///< font size
  not_null<const set<unsigned>*>
      _clusterOffsets;  ///< where does each cluster start

  /// adapts the display of clusters depending on their importance
  bool _ignoreClusters{false};

  /// index of the first symbol to be displayed on the page
  unsigned _idxOfFirstSymFromPage{UINT_MAX};
};
}  // anonymous namespace

void ui::CmapInspect::populateGrid(
    const ICmapPerspective::VPSymDataRange& symsRange,
    const set<unsigned>& clusterOffsets,
    unsigned idxOfFirstSymFromPage) noexcept {
  GridHelper gh{content(), grid, (int)*fontSz, clusterOffsets};
  gh.ignoreClusters(!cmapPresenter->areClustersUsed())
      .idxOfFirstSymFromPage(idxOfFirstSymFromPage);
  gh.populate(symsRange,
              [](const ICmapPerspective::VPSymDataCIt& iter) noexcept -> Mat {
                if ((*iter)->isRemovable())
                  return 255U - (*iter)->getNegSym();
                return (*iter)->getNegSym();
              });
}

void ui::CmapInspect::showUnofficial1stPage(
    const vector<Mat>& symsOn1stPage,
    atomic_flag& updating1stCmapPage,
    LockFreeQueue& updateSymsActionsQueue) noexcept {
  // Just leave if the main worker has already registered its intention to
  // render the official cmap
  if (updating1stCmapPage.test())
    return;

  // For some reason, newOwnerUnofficial1stCmapPage =
  // move(pUnofficial1stCmapPage) with unique_ptr does not work within the
  // lambda capture from below
  owner<Mat*> pUnofficial1stCmapPage = new Mat;
  GridHelper gh{*pUnofficial1stCmapPage, grid, (int)*fontSz};
  gh.ignoreClusters(!cmapPresenter->areClustersUsed());
  gh.populate(
      symsOn1stPage,
      [](const vector<Mat>::const_iterator& iter) noexcept { return *iter; });

  // If the main worker still didn't register its intention to render the
  // official 1st cmap page, display the unofficial early version. Otherwise
  // just leave
  if (!updating1stCmapPage.test_and_set()) {
    // holds the value on true after acquiring it

    updateSymsActionsQueue.push(
        owner<BasicUpdateSymsAction*>{new BasicUpdateSymsAction{
            [winNameCopy = winName(),

             // updating1stCmapPage is a field of Controller, so it will not
             // vanish; Thus its address can be taken
             pUpdating1stCmapPage = &updating1stCmapPage,

             // For some reason, newOwnerUnofficial1stCmapPage =
             // move(pUnofficial1stCmapPage) with unique_ptr does not work
             pUnofficial1stCmapPage] {
              // Cease ownership of the pointer
              auto newOwnerUnofficial1stCmapPage =
                  unique_ptr<Mat>{pUnofficial1stCmapPage};
              imshow(winNameCopy, *newOwnerUnofficial1stCmapPage);

              // puts its value on false, to be acquired
              // by the official version publisher
              pUpdating1stCmapPage->clear();
              pUpdating1stCmapPage->notify_one();
            }}});

    pUnofficial1stCmapPage = nullptr;  // lost ownership within lambda

  } else {
    delete pUnofficial1stCmapPage;
  }
}

extern const String CmapInspect_pageTrackName;

ui::CmapInspect::CmapInspect(const IPresentCmap& cmapPresenter_,
                             const ISelectSymbols& symsSelector,
                             const unsigned& fontSz_) noexcept
    : CvWin{CmapInspectWinName},
      cmapPresenter(&cmapPresenter_),
      fontSz(&fontSz_) {
  content(grid = createGrid());

  /*
  `CmapInspect::updatePageIdx` from below needs to work with a `void*` of actual
  type `ICmapInspect*`. But, passing `static_cast<void*>(this)` will fail,
  since `this` is a `CmapInspect*` and the actual address of
  `(ICmapInspect*)this` is different from `this` because `ICmapInspect` isn't
  the top inherited interface.
  */
  void* const thisAsICmapInspectPtr = static_cast<void*>((ICmapInspect*)this);

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
        not_null<const ISelectSymbols*> pss =
            static_cast<ISelectSymbols*>(userdata);

        if (event == EVENT_MOUSEMOVE) {
          // Mouse move
          if (const syms::ISymData* psd = pss->pointedSymbol(x, y))
            pss->displaySymCode(psd->getCode());
        } else if ((event == EVENT_LBUTTONDBLCLK) &&
                   !(flags & EVENT_FLAG_CTRLKEY)) {
          // Ctrl key not pressed and left mouse double-click
          pss->symbolsReadyToInvestigate();
        } else if ((event == EVENT_LBUTTONUP) && (flags & EVENT_FLAG_CTRLKEY)) {
          // Ctrl key pressed and left mouse click
          if (const syms::ISymData* psd = pss->pointedSymbol(x, y))
            pss->enlistSymbolForInvestigation(*psd);
        }
      },

      // Pass &symsSelector (which is const ISelectSymbols*) to 'void* userdata'
      // from above.
      // The userdata is reconverted back there: 'const ISelectSymbols* pss'
      static_cast<void*>(const_cast<ISelectSymbols*>(&symsSelector)));
}

Mat ui::CmapInspect::createGrid() noexcept {
  static const Scalar GridColor{255U, 200U, 200U};

  Mat emptyGrid{CmapInspect_pageSz, CV_8UC3, Scalar::all(255.)};

  cellSide = 1U + *fontSz;
  symsPerRow = (((unsigned)CmapInspect_pageSz.width - 1U) / cellSide);
  symsPerPage =
      symsPerRow * (((unsigned)CmapInspect_pageSz.height - 1U) / cellSide);

  // Draws horizontal & vertical cell borders
  for (int i{}; i < CmapInspect_pageSz.width; i += cellSide)
    emptyGrid.col(i).setTo(GridColor);
  for (int i{}; i < CmapInspect_pageSz.height; i += cellSide)
    emptyGrid.row(i).setTo(GridColor);
  return emptyGrid;
}

void ui::CmapPerspective::reset(
    const syms::VSymData& symsSet,
    const vector<vector<unsigned>>& symsIndicesPerCluster_) noexcept {
  Expects(!symsSet.empty());
  Expects(!symsIndicesPerCluster_.empty());

  const auto symsCount = size(symsSet),
             clustersCount = size(symsIndicesPerCluster_);

  vector<const vector<unsigned>*> symsIndicesPerCluster(clustersCount);
  size_t clustIdx{};
  for (const vector<unsigned>& clusterMembers : symsIndicesPerCluster_)
    symsIndicesPerCluster[clustIdx++] = &clusterMembers;

  // View the clusters in descending order of their size

  // Typically, there are only a few clusters larger than 1 element.
  // This partition separates the actual formed clusters from one-of-a-kind
  // elements leaving less work to perform to the sort executed afterwards.
  // Using the stable algorithm version, to preserve avgPixVal sorting
  // set by ClusterEngine::process
  const auto& [itFirstClusterWithOneItem, _] = ranges::stable_partition(
      symsIndicesPerCluster, [](not_null<const vector<unsigned>*> a) noexcept {
        // Place actual clusters at the beginning of
        // the vector
        return size(*a) > 1ULL;
      });

  // Sort non-trivial clusters in descending order of their size.
  // Using the stable algorithm version, to preserve avgPixVal sorting
  // set by ClusterEngine::process
  stable_sort(begin(symsIndicesPerCluster), itFirstClusterWithOneItem,
              [](not_null<const vector<unsigned>*> a,
                 not_null<const vector<unsigned>*> b) noexcept {
                return size(*a) > size(*b);
              });

  pSyms.resize(symsCount);

  size_t offset{};
  clusterOffsets.clear();
  clusterOffsets.insert((unsigned)offset);
  for (not_null<const vector<unsigned>*> clusterMembers :
       symsIndicesPerCluster) {
    const auto prevOffset = offset;
    offset += size(*clusterMembers);
    clusterOffsets.emplace_hint(end(clusterOffsets), (unsigned)offset);
    for (size_t idxPSyms{prevOffset}, idxCluster{}; idxPSyms < offset;
         ++idxPSyms, ++idxCluster)
      pSyms[idxPSyms] = symsSet[(size_t)(*clusterMembers)[idxCluster]].get();
  }
}

ui::ICmapPerspective::VPSymDataRange ui::CmapPerspective::getSymsRange(
    unsigned from,
    unsigned count) const noexcept {
  const auto sz = size(pSyms);
  const VPSymDataCIt itEnd{pSyms.cend()};
  if ((size_t)from >= sz)
    return ranges::subrange<VPSymDataCIt>();  // empty range

  const VPSymDataCIt itStart{next(pSyms.cbegin(), from)};
  const auto maxCount = sz - (size_t)from;
  if ((size_t)count >= maxCount)
    return ranges::subrange(itStart, itEnd);

  return ranges::subrange(itStart, next(itStart, count));
}

const set<unsigned>& ui::CmapPerspective::getClusterOffsets() const noexcept {
  return clusterOffsets;
}

#endif  // UNIT_TESTING not defined

}  // namespace pic2sym
