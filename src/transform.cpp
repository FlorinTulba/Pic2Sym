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

#include "transform.h"

#include "bestMatchBase.h"
#include "controllerBase.h"
#include "imgBasicData.h"
#include "jobMonitorBase.h"
#include "match.h"
#include "matchAssessment.h"
#include "matchEngineBase.h"
#include "matchParamsBase.h"
#include "ompTrace.h"
#include "patchBase.h"
#include "picTransformProgressTracker.h"
#include "preselectManager.h"
#include "resizedImg.h"
#include "settingsBase.h"
#include "symSettingsBase.h"
#include "symsChangeIssues.h"
#include "taskMonitor.h"
#include "timing.h"
#include "transformSupportBase.h"
#include "transformTrace.h"

#pragma warning(push, 0)

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <Windows.h>

#ifndef UNIT_TESTING

#include "warnings.h"

// The project uses parallelism
#include <omp.h>

#else  // UNIT_TESTING defined
// Unit Tests don't use parallelism, to ensure that at least the sequential code
// works as expected

/// @return 0 - the index of the unique thread used
extern int __cdecl omp_get_thread_num(void);

#endif  // UNIT_TESTING

#include <filesystem>
#include <numeric>
#include <tuple>

#include <opencv2/imgproc/imgproc.hpp>

#pragma warning(pop)

namespace pic2sym {

/// Checks if the user wants to cancel the image transformation by pressing ESC
extern void checkCancellationRequest(std::future<void>&,
                                     std::atomic_flag&) noexcept;

}  // namespace pic2sym

#ifndef UNIT_TESTING

#include "appStart.h"

#pragma warning(push, 0)

#include <opencv2/highgui/highgui.hpp>

#pragma warning(pop)

namespace pic2sym {

void checkCancellationRequest(std::future<void>& abortHandler,
                              std::atomic_flag& isCanceled) noexcept {
  // Allow the user to abort the image transformation.
  // Wait for the user to answer to the async question without prompting
  // him/her again
  if ((!abortHandler.valid() ||
       abortHandler.wait_for(0ms) == std::future_status::ready) &&
      EscKeyCode == cv::waitKey(1)) {
    // Prompt and wait for the user answer asynchronously.
    // waitKey had to remain within the main thread.
    abortHandler = std::async(std::launch::async, [&]() noexcept {
      if (IDYES == MessageBox(nullptr,
                              L"Do you want to abort the image transformation?",
                              L"Question",
                              MB_ICONQUESTION | MB_YESNOCANCEL | MB_TASKMODAL |
                                  MB_SETFOREGROUND)) {
        std::ignore = isCanceled.test_and_set();
      }
    });
  }
}

}  // namespace pic2sym

#endif  // UNIT_TESTING not defined

using namespace std;
using namespace std::filesystem;
using namespace cv;
using namespace gsl;

namespace pic2sym {

/**
Handy flag during development:
- without it, same scenario couldn't be tested again unless deleting the
result file from the Output folder
- just set it to true within the unit you're updating, do the tests, then
remove the assignment
*/
extern bool AllowReprocessingCases{false};

extern const Size BlurWinSize;
extern const double BlurStandardDeviation;
extern const bool ParallelizeTr_PatchRowLoops;

using namespace syms;
using namespace match;

namespace transform {

#ifndef UNIT_TESTING

/// Tackles the problems related to saved results
class ResultFileManager final {
 public:
  ResultFileManager(const ResultFileManager&) noexcept = default;
  ResultFileManager(ResultFileManager&&) noexcept = default;

  // Reference fields disallow the assignment operators
  void operator=(const ResultFileManager&) = delete;
  void operator=(ResultFileManager&&) = delete;

  ResultFileManager(
      const string&
          studiedCase,  ///< unique id describing the transformation params
      const atomic_flag& isCanceled_,  ///< reference to the cancel flag
      Mat& result_,                    ///< reference to the result
      IActiveTimer& timer_  ///< reference to the timer used for the current
                            ///< transformation
      ) noexcept
      : isCanceled(&isCanceled_), result(&result_), timer(&timer_) {
    if (static bool outputFolderCreated{false}; !outputFolderCreated) {
      createOutputFolder();
      outputFolderCreated = true;
    }

    // Generating a JPG result file (minor quality loss, but less space)
    resultFile = AppStart::dir();
    resultFile.append("Output").append(studiedCase).concat(".jpg");

    if (!AllowReprocessingCases && exists(resultFile)) {
      *result = imread(resultFile.string(), ImreadModes::IMREAD_UNCHANGED);
      timer->cancel(
          "This image has already been transformed under these settings.\n"
          "Displaying the available result!");
      alreadyProcessedCase = true;
    }
  }

  ~ResultFileManager() noexcept {
    if (alreadyProcessedCase)
      return;

    const bool canceled{isCanceled->test()};
    if (canceled) {
      timer->cancel("Image transformation was canceled!");

      // Still saving the partial result, but with a timestamp before the jpg
      // extension
      resultFile = resultFile.replace_extension()
                       .concat("_")
                       .concat(to_string(time(nullptr)))
                       .concat(".jpg");

    } else
      timer->pause();  // don't time result serialization

    cout << "Writing result to " << resultFile << '\n' << endl;
    imwrite(resultFile.string(), *result);

    if (!canceled)
      timer->resume();  // let any further processing get timed
  }

  /// Creates 'Output' directory which stores generated results
  static void createOutputFolder() noexcept {
    // Ensure there is an Output folder
    path outputFolder{AppStart::dir()};
    if (!exists(outputFolder.append("Output")))
      create_directory(outputFolder);
  }

  bool detectedPreviouslyProcessedCase() const noexcept {
    return alreadyProcessedCase;
  }

 private:
  /// Monitors if the process of the file was canceled
  not_null<const atomic_flag*> isCanceled;

  not_null<Mat*> result;  ///< reference to the result
  path resultFile;        ///< path of the result

  /// Reference to the timer used for the current transformation
  not_null<IActiveTimer*> timer;

  /// Previously studied cases don't need reprocessing
  bool alreadyProcessedCase{false};
};

#else  // UNIT_TESTING defined

/// Unit Tests don't need any management for the result file
class ResultFileManager final {
 public:
  ResultFileManager(const string&,
                    const atomic_flag&,
                    Mat&,
                    IActiveTimer&) noexcept {}
  ~ResultFileManager() noexcept = default;

  ResultFileManager(const ResultFileManager&) noexcept = default;
  ResultFileManager(ResultFileManager&&) noexcept = default;

  void operator=(const ResultFileManager&) = delete;
  void operator=(ResultFileManager&&) = delete;

  bool detectedPreviouslyProcessedCase() const noexcept { return false; }
};

#endif  // UNIT_TESTING

#if defined _DEBUG && !defined UNIT_TESTING
/// In Debug mode (not UnitTesting), when the transformation wasn't canceled,
/// log the parameters of the matches
static void logDataForBestMatches(
    const bool isCanceled_,
    const string& studiedCase,
    unsigned sz,
    int h,
    int w,
    bool usesUnicode,
    const vector<vector<unique_ptr<IBestMatch>>>& draftMatches) noexcept {
  if (isCanceled_)
    return;

  // Log support (DEBUG mode only)
  TransformTrace tt{studiedCase, sz, usesUnicode};
  for (int r{}; r < h; r += sz) {
    const vector<unique_ptr<IBestMatch>>& draftRow =
        draftMatches[size_t((unsigned)r / sz)];
    for (int c{}; c < w; c += sz) {
      const IBestMatch& draftMatch = *draftRow[size_t((unsigned)c / sz)];

      // Log the data about best match (DEBUG mode only)
      tt.newEntry((unsigned)r, (unsigned)c, draftMatch);
    }
  }
}

#else  // UNIT_TESTING || !_DEBUG  - ignore the call to logDataForBestMatches
static void logDataForBestMatches(auto&&...) noexcept {}

#endif  // _DEBUG, UNIT_TESTING

using namespace match;

Transformer::Transformer(IController& ctrler_,
                         const p2s::cfg::ISettings& cfg_,
                         IMatchEngine& me_,
                         p2s::input::IBasicImgData& img_) noexcept
    : ctrler(&ctrler_),
      ptpt(&ctrler_.getPicTransformProgressTracker()),
      cfg(&cfg_),
      me(&me_),
      img(&img_),
      transformSupport{
          IPreselManager::concrete().createTransformSupport(me_,
                                                            cfg_.getMS(),
                                                            resized,
                                                            resizedBlurred,
                                                            draftMatches,
                                                            me->support())} {}

void Transformer::run() {
  // [j]threads don't have a builtin mechanism to just check if they finished.
  // future-s however do have it: valid() && wait_for(0ms)==ready
  future<void> abortHandler;
  isCanceled.clear();
  durationS = 0.;

  using p2s::ui::TaskMonitor;

  static TaskMonitor preparations{
      "preparations of the timer, image, symbol sets and result",
      *transformMonitor};

  Timer timer{ptpt->createTimerForImgTransform()};

  try {
    me->updateSymbols();  // throws for invalid cmap/size
  } catch (const TinySymsLoadingFailure&) {
    timer.invalidate();
    ptpt->transformFailedToStart();
    SymsLoadingFailure::informUser(
        "Couldn't load the tiny versions of the selected font, "
        "thus the transformation was aborted!");
    return;
  }

  sz = cfg->getSS().getFontSz();

  // In UnitTesting throws logic_error when no image
  const p2s::input::ResizedImg resizedImg{img->original(), cfg->getIS(), sz};
  const bool newResizedImg{ctrler->updateResizedImg(resizedImg)};
  const Mat& resizedVersion = resizedImg.get();
  h = resizedVersion.rows;
  w = resizedVersion.cols;
  updateStudiedCase(h, w);

  ResultFileManager rf{studiedCase, isCanceled, result, timer};
  if (rf.detectedPreviouslyProcessedCase())
    return;

  const unsigned patchesPerRow{(unsigned)w / sz};
  const unsigned patchesPerCol{(unsigned)h / sz};
  initDraftMatches(newResizedImg, resizedVersion, patchesPerCol, patchesPerRow);

  try {
    me->getReady();
  } catch (const exception& e) {
    infoMsg(e.what(), "Manageable Error");
    return;
  }

  symsCount = me->getSymsCount();

  result = resizedBlurred.clone();       // initialize the result with a simple
                                         // blur. Mandatory clone!
  ptpt->presentTransformationResults();  // show the blur as draft result

  const double preparationsDuration{timer.elapsed()};

  // If the duration of the preparations took more than .7 seconds,
  // consider the weight of preparations for the transformation to be 10%
  // Otherwise, the weight of these preparations is negligible, say 0.1%
  const double preparationsWeight = (preparationsDuration > .7) ? .1 : .001;
  const double transformationWeight = 1. - preparationsWeight;

  transformMonitor->setTasksDetails(
      {// preparations of the timer, image, symbol sets and result
       preparationsWeight,

       // transformation of the image's patches
       transformationWeight},
      timer);

  preparations.taskDone();
  cout << "The " << preparations.monitoredTask()
       << " preceding the transformation took " << fixed << setprecision(2)
       << preparationsDuration << "s." << endl;

  // Transformation task can be aborted only after processing several rows of
  // patches with a new symbols batch. Therefore the total steps required to
  // complete the task is the symbols count multiplied by the number of rows
  // of patches.
  static TaskMonitor imgTransformTaskMonitor{
      "transformation of the image's patches", *transformMonitor};

  imgTransformTaskMonitor.setTotalSteps((size_t)symsCount * patchesPerCol);

  // symsBatchSz is `volatile` => every batch might have a different size
  for (unsigned fromIdx{}, batchSz{symsBatchSz.load(memory_order_acquire)},
       upperIdx{min(batchSz, symsCount)};
       !isCanceled.test() && fromIdx < symsCount;
       fromIdx = upperIdx, batchSz = symsBatchSz.load(memory_order_acquire),
       upperIdx = ((batchSz == UINT_MAX) ? symsCount
                                         : min(upperIdx + batchSz, symsCount)))
    considerSymsBatch(fromIdx, upperIdx, imgTransformTaskMonitor, abortHandler);

  if (!isCanceled.test()) {
#if defined(MONITOR_SKIPPED_MATCHING_ASPECTS) && !defined(UNIT_TESTING)

    me->assessor().reportSkippedAspects();

#endif  // defined(MONITOR_SKIPPED_MATCHING_ASPECTS) && !defined(UNIT_TESTING)

    imgTransformTaskMonitor.taskDone();
  }

  logDataForBestMatches(isCanceled.test(), studiedCase, sz, h, w,
                        me->usesUnicode(), draftMatches);
}

void Transformer::initDraftMatches(bool newResizedImg,
                                   const Mat& resizedVersion,
                                   unsigned patchesPerCol,
                                   unsigned patchesPerRow) noexcept {
  // processing new resized image
  if (newResizedImg || draftMatches.empty()) {
    const bool isColor{img->isColor()};
    resized = resizedVersion;
    GaussianBlur(resized, resizedBlurred, BlurWinSize, BlurStandardDeviation,
                 BlurStandardDeviation, BORDER_REPLICATE);

    transformSupport->initDrafts(isColor, sz, patchesPerCol, patchesPerRow);

  } else {  // processing same ResizedImg
    transformSupport->resetDrafts(patchesPerCol);
  }
}

void Transformer::considerSymsBatch(
    unsigned fromIdx,
    unsigned upperIdx,
    p2s::ui::AbsTaskMonitor& imgTransformTaskMonitor,
    future<void>& abortHandler) noexcept {
  // Cannot set finalizedRows as reduction(+ : finalizedRows) in the for
  // below, since its value is checked during the loop - the same story as for
  // isCanceled
  volatile size_t finalizedRows{};  // volatile and omp atomic is enough

  const int patchesPerCol{h / (int)sz};
  const size_t rowsOfPatches{size_t(patchesPerCol)};
  const size_t batchSz{size_t(upperIdx - fromIdx)};
  const size_t prevSteps{(size_t)fromIdx * rowsOfPatches};

#pragma warning(disable : WARN_CODE_ANALYSIS_IGNORES_OPENMP)
#pragma omp parallel if (ParallelizeTr_PatchRowLoops)
#pragma omp for schedule(dynamic) nowait
  for (int r{}; r < patchesPerCol; ++r) {
    if (isCanceled.test())
      // OpenMP doesn't accept break, so just continue with empty iterations
      continue;

    OMP_PRINTF(ParallelizeTr_PatchRowLoops, "syms batch: [%d - %d); row = %d",
               fromIdx, upperIdx, r);

    transformSupport->approxRow(r, w, sz, fromIdx, upperIdx, result);

#pragma omp atomic
    ++finalizedRows;

    // #pragma omp master not allowed in for
    if (!omp_get_thread_num() && (finalizedRows < rowsOfPatches)) {
      imgTransformTaskMonitor.taskAdvanced(prevSteps + batchSz * finalizedRows);

      // Only master thread checks cancellation status
      checkCancellationRequest(abortHandler, isCanceled);
    }
  }  // rows loop
#pragma warning(default : WARN_CODE_ANALYSIS_IGNORES_OPENMP)

  if (isCanceled.test())
    imgTransformTaskMonitor.taskAborted();
  else
    imgTransformTaskMonitor.taskAdvanced(
        prevSteps + batchSz * rowsOfPatches);  // another finished batch

  // At the end of this batch, display draft result, unless this is the last
  // batch. For the last batch (upperIdx == symsCount), the timer's destructor
  // will display the result (final draft) and it will also report
  // either the transformation duration or the fact that the transformation
  // was canceled.
  if (upperIdx < symsCount)
    ptpt->presentTransformationResults();
}

void Transformer::setSymsBatchSize(int symsBatchSz_) noexcept {
  if (symsBatchSz_ <= 0)
    symsBatchSz.store(UINT_MAX, memory_order_release);
  else
    symsBatchSz.store((unsigned)symsBatchSz_, memory_order_release);
}

Transformer& Transformer::useTransformMonitor(
    p2s::ui::AbsJobMonitor& transformMonitor_) noexcept {
  transformMonitor = &transformMonitor_;
  return *this;
}

}  // namespace transform
}  // namespace pic2sym
