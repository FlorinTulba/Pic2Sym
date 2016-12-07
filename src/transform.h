/************************************************************************************************
 The application Pic2Sym approximates images by a
 grid of colored symbols with colored backgrounds.

 This file belongs to the Pic2Sym project.

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

#ifndef H_TRANSFORM
#define H_TRANSFORM

#include "matchEngine.h"
#include "img.h"

#pragma warning ( push, 0 )

#include <string>
#include <vector>

#include <opencv2/core/core.hpp>

#pragma warning ( pop )

// Forward declarations
class Settings;		// global settings
struct IPicTransformProgressTracker;	// data & views manager
class Timer;
class TaskMonitor;

/// Transformer allows images to be approximated as a table of colored symbols from font files.
class Transformer {
protected:
	const IPicTransformProgressTracker &ctrler;	///< data & views manager
	AbsJobMonitor *transformMonitor;	///< observer of the transformation process who reports its progress

	const Settings &cfg;		///< general configuration
	MatchEngine &me;			///< approximating patches
	Img &img;					///< current image to process

	cv::Mat result;				///< the result of the transformation

	std::string studiedCase;	///< unique id for the studied case
	cv::Mat resized;			///< resized version of the original
	cv::Mat resizedForTinySyms;	///< resized version of the original used by tiny symbols preselection
	cv::Mat resizedBlurred;		///< blurred version of the resized original
	cv::Mat resBlForTinySyms;	///< blurred version of the resized used by tiny symbols preselection

	std::vector<std::vector<BestMatch>> draftMatches;	///< temporary best matches
	std::vector<std::vector<BestMatch>> draftMatchesForTinySyms; ///< temporary best matches used by tiny symbols preselection

	int w = 0;					///< width of the resized image
	int h = 0;					///< height of the resized image
	unsigned sz = 0U;			///< font size used during transformation
	unsigned symsCount = 0U;	///< symbols count within the used cmap

	volatile unsigned symsBatchSz;	///< runtime control of how large next symbol batches are

	volatile bool isCanceled = false;	///< has the process been canceled?

	void updateStudiedCase(int rows, int cols); ///< Updates the unique id for the studied case

	/// Makes sure draftMatches will be computed for correct resized img
	void initDraftMatches(bool newResizedImg, const cv::Mat &resizedVersion,
						  unsigned patchesPerCol, unsigned patchesPerRow);

	/// Improves the result by analyzing the symbols in range [fromIdx, upperIdx) under the supervision of imgTransformTaskMonitor
	void considerSymsBatch(unsigned fromIdx, unsigned upperIdx, TaskMonitor &imgTransformTaskMonitor);

public:
	Transformer(const IPicTransformProgressTracker &ctrler_, const Settings &cfg_,
				MatchEngine &me_, Img &img_);
	void operator=(const Transformer&) = delete;

	void run();	///< applies the configured transformation onto current/new image

	const cv::Mat& getResult() const { return result; }

	/**
	Updates symsBatchSz.
	@param symsBatchSz_ the value to set. If 0 is provided, batching symbols
		gets disabled for the rest of the transformation, ignoring any new slider positions.
	*/
	void setSymsBatchSize(int symsBatchSz_);

	Transformer& useTransformMonitor(AbsJobMonitor &transformMonitor_); ///< setting the transformation monitor
};

#endif