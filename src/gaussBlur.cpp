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

#include "gaussBlur.h"
#include "misc.h"

#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

GaussBlur::GaussBlur(double desiredSigma, unsigned kernelWidth_/* = 0U*/) {
	configure(desiredSigma, kernelWidth_);
}

GaussBlur& GaussBlur::configure(double desiredSigma, unsigned kernelWidth_/* = 0U*/) {
	if(desiredSigma < 0.)
		THROW_WITH_CONST_MSG("desiredSigma should be > 0 in " __FUNCTION__, invalid_argument);

	if(kernelWidth_ != 0U && (kernelWidth_ & 1U) != 1U)
		THROW_WITH_CONST_MSG("kernelWidth_ should be an odd value or 0 in " __FUNCTION__, invalid_argument);

	nonTinySymsParams.sigma = desiredSigma;
	nonTinySymsParams.kernelWidth = kernelWidth_;

	// Tiny symbols should use a sigma = desiredSigma/2. and kernel whose width is
	// next odd value >= kernelWidth_/2.
	tinySymsParams.sigma = desiredSigma * .5;
	tinySymsParams.kernelWidth = (kernelWidth_>>1) | 1U;

	return *this;
}

void GaussBlur::doProcess(const cv::Mat &toBlur, cv::Mat &blurred, bool forTinySym) const {
	if(forTinySym)
		GaussianBlur(toBlur, blurred,
					Size(tinySymsParams.kernelWidth, tinySymsParams.kernelWidth),
					tinySymsParams.sigma, tinySymsParams.sigma, BORDER_REPLICATE);
	else
		GaussianBlur(toBlur, blurred,
					Size(nonTinySymsParams.kernelWidth, nonTinySymsParams.kernelWidth),
					nonTinySymsParams.sigma, nonTinySymsParams.sigma, BORDER_REPLICATE);
}

const GaussBlur& GaussBlur::configuredInstance() {
	// Gaussian blur with desired standard deviation and window width
	extern const int StructuralSimilarity_RecommendedWindowSide;
	extern const double StructuralSimilarity_SIGMA;
	static GaussBlur result(StructuralSimilarity_SIGMA, StructuralSimilarity_RecommendedWindowSide);
	return result;
}
