/****************************************************************************************
 The application Pic2Sym approximates images by a
 grid of colored symbols with colored backgrounds.

 This file was created on 2016-4-10
 and belongs to the Pic2Sym project.

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

#include "patch.h"
#include "matchEngine.h"

#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

Patch::Patch(const Mat &patch_, const Mat &blurredPatch_, bool isColor_) :
		patch(patch_), blurredPatch(blurredPatch_), sz(patch_.rows), isColor(isColor_) {
	// Don't approximate rather uniform patches
	Mat grayBlurredPatch;
	if(isColor)
		cv::cvtColor(blurredPatch, grayBlurredPatch, cv::COLOR_RGB2GRAY);
	else
		grayBlurredPatch = blurredPatch;

	double minVal, maxVal;
	cv::minMaxIdx(grayBlurredPatch, &minVal, &maxVal); // assessed on blurred patch, to avoid outliers bias
	extern const double Transformer_run_THRESHOLD_CONTRAST_BLURRED;
	if(maxVal-minVal < Transformer_run_THRESHOLD_CONTRAST_BLURRED)
		needsApproximation = false;

	const cv::Mat &patch2Process = patch; // blurredPatch
	if(isColor)
		cv::cvtColor(patch2Process, grayPatchD, cv::COLOR_RGB2GRAY);
	else
		grayPatchD = patch2Process;
	grayPatchD.convertTo(grayPatchD, CV_64FC1);
}

const BestMatch Patch::approximate(const MatchSettings &ms, const MatchEngine &me) const {
	if(!needsApproximation)
		return BestMatch(*this).updatePatchApprox(ms);

	// The patch is less uniform, so it needs approximation
	return me.approxPatch(*this);
}