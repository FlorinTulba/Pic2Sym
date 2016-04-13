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

#ifndef H_PATCH
#define H_PATCH

#include <opencv2/core/core.hpp>

// forward declarations
class MatchEngine;
class MatchSettings;

/**
Separates patch approximation logic from the rest.
*/
class Patch {
protected:
	cv::Mat grayD;					///< gray version of the patch to process with double values

public:
	bool needsApproximation = true;	///< patches that appear uniform use 'blurredPatch' as approximation
	const cv::Mat orig;		///< the patch to approximate
	const unsigned sz;		///< patch size
	const bool isColor;		///< is the patch color or grayscale
	const cv::Mat blurred;	///< the blurred version of the orig

	/**
	Performs the transformation of the patch.
	@param orig_ patch to be approximated
	@param blurred_ blurred version of the patch
	@param isColor_ type of image - color => true; grayscale => false
	*/
	Patch(const cv::Mat &orig_, const cv::Mat &blurred_, bool isColor_);

	/// specifies which matrix to use during the approximation process
	const cv::Mat& matrixToApprox() const { return grayD; }
};

#endif