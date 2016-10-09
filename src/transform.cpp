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

#include "transform.h"
#include "picTransformProgressTracker.h"
#include "misc.h"
#include "timing.h"
#include "settings.h"
#include "appStart.h"
#include "matchParams.h"
#include "patch.h"
#include "jobMonitorBase.h"
#include "taskMonitor.h"
#include "transformTrace.h"
#include "ompTrace.h"

#include <omp.h>

#include <sstream>
#include <numeric>

#include <boost/filesystem/operations.hpp>

/// Checks if the user wants to cancel the image transformation by pressing ESC
static bool checkCancellationRequest();

#ifndef UNIT_TESTING
#include <opencv2/highgui.hpp>

bool checkCancellationRequest() {
	return cv::waitKey(1) == 27; // cancel if the user presses ESC
}
#endif // UNIT_TESTING not defined

using namespace std;
using namespace boost::filesystem;
using namespace cv;

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

	bool detectedPreviouslyProcessedCase() const { return alreadyProcessedCase; }

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
	
		if(exists(resultFile)) {
			result = imread(resultFile.string(), ImreadModes::IMREAD_UNCHANGED);
			timer.cancel("This image has already been transformed under these settings.\n"
						 "Displaying the available result!");
			alreadyProcessedCase = true;
		}
	}

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

#endif // UNIT_TESTING not defined

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
#else // ignore the call to logDataForBestMatches
#define logDataForBestMatches(...)
#endif // _DEBUG && !UNIT_TESTING

extern const Size BlurWinSize;
extern const double BlurStandardDeviation;
extern const bool ParallelizeTr_PatchRowLoops;
extern const unsigned SymsBatch_defaultSz;

Transformer::Transformer(const IPicTransformProgressTracker &ctrler_, const Settings &cfg_,
						 MatchEngine &me_, Img &img_) :
		ctrler(ctrler_), cfg(cfg_), me(me_), img(img_), symsBatchSz(SymsBatch_defaultSz) {}

void Transformer::run() {
	isCanceled = false;

	static TaskMonitor preparations("preparations of the timer, image, symbol sets and result", *transformMonitor);

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

#ifndef UNIT_TESTING
	ResultFileManager rf(studiedCase, isCanceled, result, timer);
	if(rf.detectedPreviouslyProcessedCase())
		return;
#endif
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
	});

	preparations.taskDone();
	cout<<"The "<<preparations.monitoredTask()<<" preceding the transformation took "
		<<fixed<<setprecision(2)<<preparationsDuration<<"s."<<endl;

	// Transformation task can be aborted only after processing several rows of patches with a new symbols batch.
	// Therefore the total steps required to complete the task is the symbols count multiplied by the number of rows of patches.
	static TaskMonitor imgTransformTaskMonitor("transformation of the image's patches", *transformMonitor);
	imgTransformTaskMonitor.setTotalSteps((size_t)symsCount * (size_t)patchesPerCol);

	// symsBatchSz is volatile => every batch might have a different size
	for(unsigned fromIdx = 0U, upperIdx = min(const_cast<unsigned&>(symsBatchSz), symsCount);
			!isCanceled && fromIdx < symsCount;
			fromIdx = upperIdx, upperIdx = min(upperIdx + symsBatchSz, symsCount))
		considerSymsBatch(fromIdx, upperIdx, imgTransformTaskMonitor);

	if(!isCanceled) {
#ifdef _DEBUG
		cout<<"Transformation finished. Reporting skipped aspects from a total of "<<me.totalIsBetterMatchCalls<<" isBetterMatch calls: ";
		copy(CBOUNDS(me.skippedAspects), ostream_iterator<size_t>(cout, ", "));
		cout<<"\b\b  "<<endl;
#endif // _DEBUG

		imgTransformTaskMonitor.taskDone();
	}

	logDataForBestMatches(isCanceled, studiedCase, sz, h, w, me.usesUnicode(), draftMatches);
}

void Transformer::initDraftMatches(bool newResizedImg, const Mat &resizedVersion,
								   unsigned patchesPerCol, unsigned patchesPerRow) {
	// processing new resized image
	if(newResizedImg || draftMatches.empty()) {
		resized = resizedVersion; isColor = img.isColor();
		GaussianBlur(resized, resizedBlurred, BlurWinSize, BlurStandardDeviation, 0., BORDER_REPLICATE);
		
		draftMatches.clear(); draftMatches.reserve(patchesPerCol);
		for(int r = 0; r<h; r += sz) {
			const Range rowRange(r, r+sz);

			draftMatches.emplace_back();
			auto &draftMatchesRow = draftMatches.back(); draftMatchesRow.reserve(patchesPerRow);
			for(int c = 0; c<w; c += sz) {
				const Range colRange(c, c+sz);
				const Mat patch(resized, rowRange, colRange),
					blurredPatch(resizedBlurred, rowRange, colRange);

				// Building a Patch with the blurred patch computed for its actual borders
				draftMatchesRow.emplace_back(Patch(patch, blurredPatch, isColor));
			}
		}
	} else { // processing same ResizedImg
		for(unsigned r = 0U; r<patchesPerCol; ++r) {
			auto &draftMatchesRow = draftMatches[r];
			for(unsigned c = 0U; c<patchesPerRow; ++c) {
				auto &draftMatch = draftMatchesRow[c];
				draftMatch.reset(); // leave nothing but the Patch field
			}
		}
	}
}

void Transformer::considerSymsBatch(unsigned fromIdx, unsigned upperIdx, TaskMonitor &imgTransformTaskMonitor) {
	volatile size_t finalizedRows = 0U;
	const size_t rowsOfPatches = size_t((unsigned)h/sz),
				batchSz = size_t(upperIdx - fromIdx),
				prevSteps = (size_t)fromIdx * rowsOfPatches;

#pragma omp parallel if(ParallelizeTr_PatchRowLoops)
#pragma omp for schedule(dynamic) nowait
	for(int r = 0; r<h; r += sz) {
		if(isCanceled)
			continue; // OpenMP doesn't accept break, so just continue with empty iterations

		ompPrintf(ParallelizeTr_PatchRowLoops, "syms batch: %d - %d; row = %d", fromIdx, upperIdx-1, r);
		const Range rowRange(r, r+sz);
		auto &draftMatchesRow = draftMatches[(unsigned)r/sz];

		for(int c = 0; c<w; c += sz) {
			const Range colRange(c, c+sz);

			auto &draftMatch = draftMatchesRow[(unsigned)c/sz];
			if(me.findBetterMatch(draftMatch, fromIdx, upperIdx)) {
				const Mat &approximation = draftMatch.bestVariant.approx;
				Mat destRegion(result, rowRange, colRange);
				approximation.copyTo(destRegion);
			}
		} // columns loop

#pragma omp atomic
		++finalizedRows;

		// #pragma omp master not allowed in for
		if((omp_get_thread_num() == 0) && (finalizedRows < rowsOfPatches)) {
			imgTransformTaskMonitor.taskAdvanced(prevSteps + batchSz * finalizedRows);
			// Only master thread checks cancellation status
			if(checkCancellationRequest()) {
				isCanceled = true;
				imgTransformTaskMonitor.taskAborted();
			}
		}
	} // rows loop
	
	if(!isCanceled)
		imgTransformTaskMonitor.taskAdvanced(prevSteps + batchSz * rowsOfPatches);

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
