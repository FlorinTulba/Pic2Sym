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

#ifndef H_PATCH
#define H_PATCH

#pragma warning ( push, 0 )

#include <opencv2/core/core.hpp>

#pragma warning ( pop )

// Forward declarations
class MatchEngine;
class MatchSettings;

/**
Holds useful patch information.

It decides whether this patch needs approximation or not - uniform patches
don't produce interesting approximations.

*/
struct Patch {
	cv::Mat grayD;			///< gray version of the patch to process (its data type is double)
	const cv::Mat orig;		///< the patch to approximate
	const cv::Mat blurred;	///< the blurred version of the orig

	const bool isColor;		///< is the patch color or grayscale?

	bool needsApproximation = true;	///< patches that appear uniform use 'blurred' as approximation

	/**
	Initializer

	@param orig_ patch to be approximated
	@param blurred_ blurred version of the patch, either considering real borders, or replicated ones
	@param isColor_ type of image - color => true; grayscale => false
	*/
	Patch(const cv::Mat &orig_, const cv::Mat &blurred_, bool isColor_);
	void operator=(const Patch&) = delete;

	/// specifies which matrix to use during the approximation process
	const cv::Mat& matrixToApprox() const;

#ifdef UNIT_TESTING
	/// Constructor delegating its job to the one with 3 parameters
	Patch(const cv::Mat &orig_);
#endif //UNIT_TESTING
};

#endif // H_PATCH