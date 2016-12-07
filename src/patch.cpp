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

#include "patch.h"
#include "matchEngine.h"
#include "misc.h"

#pragma warning ( push, 0 )

#include <opencv2/imgproc/imgproc.hpp>

#pragma warning ( pop )

using namespace std;
using namespace cv;

const Mat& Patch::matrixToApprox() const {
	if(needsApproximation)
		return grayD;

	THROW_WITH_CONST_MSG(__FUNCTION__ " shouldn't be called when needsApproximation is false!", logic_error);
}

Patch::Patch(const Mat &orig_, const Mat &blurred_, bool isColor_) :
		orig(orig_), blurred(blurred_), isColor(isColor_) {
	// Don't approximate rather uniform patches
	Mat grayBlurred;
	if(isColor)
		cvtColor(blurred, grayBlurred, COLOR_RGB2GRAY);
	else
		grayBlurred = blurred;

	double minVal, maxVal;
	minMaxIdx(grayBlurred, &minVal, &maxVal); // assessed on blurred patch, to avoid outliers bias
	extern const double Transformer_run_THRESHOLD_CONTRAST_BLURRED;
	if(maxVal-minVal < Transformer_run_THRESHOLD_CONTRAST_BLURRED) {
		needsApproximation = false;
		return;
	}

	// Configurable source of transformation - either the patch itself, or its blurred version:
	extern const bool Transform_BlurredPatches_InsteadOf_Originals;
	const Mat &patch2Process = Transform_BlurredPatches_InsteadOf_Originals ?
				blurred : orig;
	if(isColor)
		cvtColor(patch2Process, grayD, COLOR_RGB2GRAY);
	else
		grayD = patch2Process.clone();
	grayD.convertTo(grayD, CV_64FC1);
}
