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

#ifndef H_PATCH_BASE
#define H_PATCH_BASE

#pragma warning ( push, 0 )

#include <memory>

#include <opencv2/core/core.hpp>

#pragma warning ( pop )

/**
Base for the Patch class.

It decides whether this patch needs approximation or not - uniform patches
don't produce interesting approximations.
*/
struct IPatch /*abstract*/ {
	virtual const cv::Mat& getOrig() const = 0;		///< the patch to approximate
	virtual const cv::Mat& getBlurred() const = 0;	///< the blurred version of the orig

	virtual bool isColored() const = 0;		///< is the patch color or grayscale?
	virtual bool nonUniform() const = 0;	///< patches that appear uniform use 'blurred' as approximation

	/// Specifies which matrix to use during the approximation process (gray, of type double)
	virtual const cv::Mat& matrixToApprox() const = 0;

	virtual ~IPatch() = 0 {}

	virtual std::unique_ptr<const IPatch> clone() const = 0;	///< @return a clone of itself
};

#endif // H_PATCH_BASE
