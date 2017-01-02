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

#include "transform.h"
#include "matchAssessment.h"
#include "settings.h"
#include "matchParams.h"
#include "patch.h"
#include "preselectManager.h"
#include "transformSupport.h"
#include "preselectSyms.h"
#include "match.h"
#include "jobMonitorBase.h"
#include "taskMonitor.h"
#include "picTransformProgressTracker.h"
#include "transformTrace.h"
#include "ompTrace.h"
#include "appStart.h"
#include "timing.h"
#include "misc.h"

#pragma warning ( push, 0 )

#include <Windows.h>

#ifndef UNIT_TESTING
// The project uses parallelism
#include <omp.h>

#else // UNIT_TESTING defined
// Unit Tests don't use parallelism, to ensure that at least the sequential code works as expected
extern int __cdecl omp_get_thread_num(void); // returns 0 - the index of the unique thread used

#endif // UNIT_TESTING

#include <sstream>
#include <numeric>

#include <boost/filesystem/operations.hpp>

#pragma warning ( pop )

/// Checks if the user wants to cancel the image transformation by pressing ESC
extern bool checkCancellationRequest();

#ifndef UNIT_TESTING

#pragma warning ( push, 0 )

#include <opencv2/highgui/highgui.hpp>

#pragma warning ( pop )

bool checkCancellationRequest() {
	// cancel if the user presses ESC and then confirms his abort request
	return cv::waitKey(1) == 27 &&
		IDYES == MessageBox(nullptr,
					L"Do you want to abort the image transformation?", L"Question",
					MB_ICONQUESTION | MB_YESNOCANCEL | MB_TASKMODAL | MB_SETFOREGROUND);
}

#endif // UNIT_TESTING not defined

using namespace std;
using namespace boost::filesystem;
using namespace cv;

/**
Handy flag during development:
- without it, same scenario couldn't be tested again unless deleting the result file from the Output folder
- just set it to true within the unit you're updating, do the tests, then remove the assignment
*/
extern bool AllowReprocessingCases = false;

#ifndef UNIT_TESTING

/// Tackles the problems related to saved results
struct ResultFileManager {
	volatile bool &isCanceled;	///< monitors if the process of the file was canceled
	Mat &result;				///< reference to the result
	path resultFile;			///< path of the result
	Timer &timer;				///< reference to the timer used for the current transformation
	bool alreadyProcessedCase = false;	///< previously studied cases don't need reprocessing

	/// Creates 'Output' directory which stores generated results
	static void createOutputFolder() {
		// Ensure there is an Output folder
		path outputFolder = AppStart::dir();
		if(!exists(outputFolder.append("Output")))
			create_directory(outputFolder);
	}

	inline bool detectedPreviouslyProcessedCase() const { return alreadyProcessedCase; }

	ResultFileManager(const string &studiedCase,	///< unique id describing the transformation params
					  volatile bool &isCanceled_,	///< reference to the cancel flag
					  Mat &result_,					///< reference to the result
					  Timer &timer_					///< reference to the timer used for the current transformation
					  ) : isCanceled(isCanceled_), result(result_), timer(timer_) {
		static bool outputFolderCreated = false;
		if(!outputFolderCreated) {
			createOutputFolder();
			outputFolderCreated = true;
		}

		// Generating a JPG result file (minor quality loss, but less space)
		resultFile = AppStart::dir();
		resultFile.append("Output").append(studiedCase).concat(".jpg");
	
		if(!AllowReprocessingCases && exists(resultFile)) {
			result = imread(resultFile.string(), ImreadModes::IMREAD_UNCHANGED);
			timer.cancel("This image has already been transformed under these settings.\n"
						 "Displaying the available result!");
			alreadyProcessedCase = true;
		}
	}

	void operator=(const ResultFileManager&) = delete;

	~ResultFileManager() {
		if(alreadyProcessedCase)
			return;

		const bool canceled = isCanceled;
		if(canceled) {
			timer.cancel("Image transformation was canceled!");

			// Still saving the partial result, but with a timestamp before the jpg extension
			resultFile = resultFile.replace_extension().
				concat("_").concat(to_string(time(nullptr))).concat(".jpg");
		
		} else timer.pause(); // don't time result serialization

		cout<<"Writing result to "<<resultFile<<endl<<endl;
		imwrite(resultFile.string(), result);

		if(!canceled)
			timer.resume(); // let any further processing get timed
	}
};

#else // UNIT_TESTING defined

/// Unit Tests don't need any management for the result file
struct ResultFileManager {
	ResultFileManager(...) {}
	void operator=(const ResultFileManager&) = delete;

	inline bool detectedPreviouslyProcessedCase() const { return false; }
};

#endif // UNIT_TESTING

#if defined _DEBUG && !defined UNIT_TESTING
/// In Debug mode (not UnitTesting), when the transformation wasn't canceled, log the parameters of the matches
static void logDataForBestMatches(volatile bool &isCanceled,
						   const string &studiedCase, unsigned sz, int h, int w, bool usesUnicode,
						   const std::vector<std::vector<BestMatch>> &draftMatches) {
	if(isCanceled)
		return;
	TransformTrace tt(studiedCase, sz, usesUnicode); // log support (DEBUG mode only)
	for(int r = 0; r<h; r += sz) {
		auto &draftRow = draftMatches[(unsigned)r/sz];
		for(int c = 0; c<w; c += sz) {
			const auto &draftMatch = draftRow[(unsigned)c/sz];
			tt.newEntry((unsigned)r, (unsigned)c, draftMatch); // log the data about best match (DEBUG mode only)
		}
	}
}

#else // UNIT_TESTING || !_DEBUG  - ignore the call to logDataForBestMatches
#	define logDataForBestMatches(...)

#endif // _DEBUG, UNIT_TESTING

extern const Size BlurWinSize;
extern const double BlurStandardDeviation;
extern const bool ParallelizeTr_PatchRowLoops;

Transformer::Transformer(const IPicTransformProgressTracker &ctrler_, const Settings &cfg_,
						 MatchEngine &me_, Img &img_) :
		ctrler(ctrler_), cfg(cfg_), me(me_), img(img_) {}

void Transformer::run() {
	isCanceled = false;
	durationS = 0.;

#pragma warning ( disable : WARN_THREAD_UNSAFE )
	static TaskMonitor preparations("preparations of the timer, image, symbol sets and result", *transformMonitor);
#pragma warning ( default : WARN_THREAD_UNSAFE )

	Timer timer = ctrler.createTimerForImgTransform();

	me.updateSymbols(); // throws for invalid cmap/size

	sz = cfg.symSettings().getFontSz();
	
	std::shared_ptr<const ResizedImg> resizedImg =
		std::make_shared<const ResizedImg>(img, cfg.imgSettings(), sz); // throws when no image
	const bool newResizedImg =
		const_cast<IPicTransformProgressTracker&>(ctrler).updateResizedImg(resizedImg);
	const Mat &resizedVersion = resizedImg->get();
	h = resizedVersion.rows; w = resizedVersion.cols;
	updateStudiedCase(h, w);

	ResultFileManager rf(studiedCase, isCanceled, result, timer);
	if(rf.detectedPreviouslyProcessedCase())
		return;

	const unsigned patchesPerRow = (unsigned)w/sz, patchesPerCol = (unsigned)h/sz;
	initDraftMatches(newResizedImg, resizedVersion, patchesPerCol, patchesPerRow);

	me.getReady();
	symsCount = me.getSymsCount();

	result = resizedBlurred.clone(); // initialize the result with a simple blur. Mandatory clone!
	ctrler.presentTransformationResults(); // show the blur as draft result

	const double preparationsDuration = timer.elapsed(),

			// If the duration of the preparations took more than .7 seconds,
			// consider the weight of preparations for the transformation to be 10%
			// Otherwise, the weight of these preparations is negligible, say 0.1%
			preparationsWeight = (preparationsDuration > .7) ? .1 : .001,
			transformationWeight = 1. - preparationsWeight;

	transformMonitor->setTasksDetails({
		preparationsWeight,		// preparations of the timer, image, symbol sets and result
		transformationWeight	// transformation of the image's patches
	}, timer);

	preparations.taskDone();
	cout<<"The "<<preparations.monitoredTask()<<" preceding the transformation took "
		<<fixed<<setprecision(2)<<preparationsDuration<<"s."<<endl;

	// Transformation task can be aborted only after processing several rows of patches with a new symbols batch.
	// Therefore the total steps required to complete the task is the symbols count multiplied by the number of rows of patches.
#pragma warning ( disable : WARN_THREAD_UNSAFE )
	static TaskMonitor imgTransformTaskMonitor("transformation of the image's patches", *transformMonitor);
#pragma warning ( default : WARN_THREAD_UNSAFE )

	imgTransformTaskMonitor.setTotalSteps((size_t)symsCount * (size_t)patchesPerCol);

	// symsBatchSz is volatile => every batch might have a different size
	for(unsigned fromIdx = 0U, batchSz = const_cast<unsigned&>(symsBatchSz), upperIdx = min(batchSz, symsCount);
			!isCanceled && fromIdx < symsCount;
			fromIdx = upperIdx, batchSz = const_cast<unsigned&>(symsBatchSz),
			upperIdx = ((batchSz == UINT_MAX) ? symsCount : min(upperIdx + batchSz, symsCount)))
		considerSymsBatch(fromIdx, upperIdx, imgTransformTaskMonitor);

	if(!isCanceled) {
#ifdef MONITOR_SKIPPED_MATCHING_ASPECTS
		me.assessor().reportSkippedAspects();
#endif // MONITOR_SKIPPED_MATCHING_ASPECTS

		imgTransformTaskMonitor.taskDone();
	}

	logDataForBestMatches(isCanceled, studiedCase, sz, h, w, me.usesUnicode(), draftMatches);
}

void Transformer::initDraftMatches(bool newResizedImg, const Mat &resizedVersion,
								   unsigned patchesPerCol, unsigned patchesPerRow) {
	if(preselManager == nullptr)
		THROW_WITH_CONST_MSG("Please call 'usePreselManager()' before " __FUNCTION__, logic_error);

	auto &trSupport = preselManager->transformSupport();

	// processing new resized image
	if(newResizedImg || draftMatches.empty()) {
		const bool isColor = img.isColor();
		resized = resizedVersion;
		GaussianBlur(resized, resizedBlurred, BlurWinSize,
					 BlurStandardDeviation, BlurStandardDeviation, BORDER_REPLICATE);
		
		trSupport.initDrafts(isColor, sz, patchesPerCol, patchesPerRow);

	} else { // processing same ResizedImg
		trSupport.resetDrafts(patchesPerCol);
	}
}

void Transformer::considerSymsBatch(unsigned fromIdx, unsigned upperIdx, TaskMonitor &imgTransformTaskMonitor) {
	// Cannot set finalizedRows as reduction(+ : finalizedRows) in the for below,
	// since its value is checked during the loop - the same story as for isCanceled
	volatile size_t finalizedRows = 0U;

	assert(preselManager != nullptr);
	auto &trSupport = preselManager->transformSupport();
	const int patchesPerCol = h / (int)sz;
	const size_t rowsOfPatches = size_t(patchesPerCol),
				batchSz = size_t(upperIdx - fromIdx),
				prevSteps = (size_t)fromIdx * rowsOfPatches;

#pragma omp parallel if(ParallelizeTr_PatchRowLoops)
#pragma omp for schedule(dynamic) nowait
	for(int r = 0; r < patchesPerCol; ++r) {
		if(isCanceled)
			continue; // OpenMP doesn't accept break, so just continue with empty iterations

		ompPrintf(ParallelizeTr_PatchRowLoops, "syms batch: [%d - %d); row = %d", fromIdx, upperIdx, r);

		trSupport.approxRow(r, w, sz, fromIdx, upperIdx, result);

#pragma omp atomic
		++finalizedRows;

		// #pragma omp master not allowed in for
		if((omp_get_thread_num() == 0) && (finalizedRows < rowsOfPatches)) {
			imgTransformTaskMonitor.taskAdvanced(prevSteps + batchSz * finalizedRows);
			// Only master thread checks cancellation status
			if(checkCancellationRequest())
				isCanceled = true;
		}
	} // rows loop
	
	if(isCanceled)
		imgTransformTaskMonitor.taskAborted();
	else
		imgTransformTaskMonitor.taskAdvanced(
						prevSteps + batchSz * rowsOfPatches); // another finished batch

	// At the end of this batch, display draft result, unless this is the last batch.
	// For the last batch (upperIdx == symsCount), the timer's destructor
	// will display the result (final draft) and it will also report
	// either the transformation duration or the fact that the transformation was canceled.
	if(upperIdx < symsCount)
		ctrler.presentTransformationResults();
}

void Transformer::setSymsBatchSize(int symsBatchSz_) {
	if(symsBatchSz_ <= 0)
		symsBatchSz = UINT_MAX;
	else
		symsBatchSz = (unsigned)symsBatchSz_;
}

Transformer& Transformer::useTransformMonitor(AbsJobMonitor &transformMonitor_) {
	transformMonitor = &transformMonitor_;
	return *this;
}

Transformer& Transformer::usePreselManager(PreselManager &preselManager_) {
	preselManager = &preselManager_;
	return *this;
}
