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

#ifndef H_TRANSFORM
#define H_TRANSFORM

#include "matchEngine.h"
#include "transformCompletion.h"

#pragma warning ( push, 0 )

#include <string>
#include <vector>

#include <opencv2/core/core.hpp>

#pragma warning ( pop )

#ifndef UNIT_TESTING

extern const unsigned SymsBatch_defaultSz;

#endif // UNIT_TESTING not defined

// Forward declarations
struct ISettings;		// global settings
struct IPicTransformProgressTracker;	// data & views manager
class Timer;
class TaskMonitor;
struct IBestMatch;
struct IBasicImgData;
class TransformSupport;

/// Transformer allows images to be approximated as a table of colored symbols from font files.
class Transformer : public ITransformCompletion {
protected:
	IController &ctrler;	///< controller
	std::shared_ptr<IPicTransformProgressTracker> ptpt;	///< image transformation management from the controller
	AbsJobMonitor *transformMonitor;	///< observer of the transformation process who reports its progress

	const ISettings &cfg;		///< general configuration
	MatchEngine &me;			///< approximating patches
	IBasicImgData &img;			///< basic information about the current image to process

	cv::Mat result;				///< the result of the transformation

	std::string studiedCase;	///< unique id for the studied case
	cv::Mat resized;			///< resized version of the original
	cv::Mat resizedBlurred;		///< blurred version of the resized original

	std::vector<std::vector<std::unique_ptr<IBestMatch>>> draftMatches;	///< temporary best matches

	// Keep this after previous fields, as it depends on them
	std::unique_ptr<TransformSupport> transformSupport;	///< initializes and updates draft matches

	double durationS = 0.;		///< transformation duration in seconds

	int w = 0;					///< width of the resized image
	int h = 0;					///< height of the resized image
	unsigned sz = 0U;			///< font size used during transformation
	unsigned symsCount = 0U;	///< symbols count within the used cmap
	
#ifndef UNIT_TESTING
	// when UNIT_TESTING is not defined, start with batching SymsBatch_defaultSz symbols and allow canceling
	volatile unsigned symsBatchSz = SymsBatch_defaultSz;	///< runtime control of how large next symbol batches are
	volatile bool isCanceled = false;						///< has the process been canceled?

#else // no batching, nor canceling in Unit Testing mode
	unsigned symsBatchSz = UINT_MAX;	///< runtime control of how large next symbol batches are
	bool isCanceled = false;			///< has the process been canceled?

#endif // UNIT_TESTING

	void updateStudiedCase(int rows, int cols); ///< Updates the unique id for the studied case

	/// Makes sure draftMatches will be computed for correct resized img
	void initDraftMatches(bool newResizedImg, const cv::Mat &resizedVersion,
						  unsigned patchesPerCol, unsigned patchesPerRow);

	/// Improves the result by analyzing the symbols in range [fromIdx, upperIdx) under the supervision of imgTransformTaskMonitor
	void considerSymsBatch(unsigned fromIdx, unsigned upperIdx, TaskMonitor &imgTransformTaskMonitor);

public:
	Transformer(IController &ctrler_, const ISettings &cfg_,
				MatchEngine &me_, IBasicImgData &img_);
	void operator=(const Transformer&) = delete;

	const cv::Mat& getResult() const override final { return result; }
	void setDuration(double durationS_) override { durationS = durationS_; }

	inline double duration() const { return durationS; }

	/**
	Updates symsBatchSz.
	@param symsBatchSz_ the value to set. If 0 is provided, batching symbols
		gets disabled for the rest of the transformation, ignoring any new slider positions.
	*/
	void setSymsBatchSize(int symsBatchSz_);

	void run();	///< applies the configured transformation onto current/new image

	Transformer& useTransformMonitor(AbsJobMonitor &transformMonitor_); ///< setting the transformation monitor
};

#endif // H_TRANSFORM
