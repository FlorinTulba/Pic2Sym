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

#ifndef H_BLUR_CUDA
#define H_BLUR_CUDA

#include "floatType.h"

#include <driver_types.h>

/**
Performs box blur using CUDA.
@param imgBuff the image to be blurred
@param result the blurred image
@param toBlurDev device buffer for the image to be blurred
@param blurredDev device buffer for the blurred image
@param rows number of rows of the image
@param cols number of columns of the image
@param buffSz space occupied by input/output data
@param maskWidthsDev device buffer for the widths of the masks used within each iteration
@param iterations how many times to apply the blur (the iterations might use different width masks)
@param largestMaskRadius the radius of the largest mask used during the iterations
@param scaler factor to rescale the result in the same range as the original
@param streamId the id of the stream that handles this blur operation
*/
void boxBlur(const fp *imgBuff, fp *result,
			 fp *toBlurDev, fp *blurredDev,
			 unsigned rows, unsigned cols, size_t buffSz,
			 unsigned *maskWidthsDev, unsigned iterations,
			 unsigned largestMaskRadius, fp scaler,
			 cudaStream_t streamId);

/**
Performs stack blur using CUDA.
@param imgBuff the image to be blurred
@param result the blurred image
@param toBlurDev device buffer for the image to be blurred
@param blurredDev device buffer for the blurred image
@param rows number of rows of the image
@param stride length of a row of the image
@param radius the radius of the blur algorithm
*/
void stackBlur(const fp *imgBuff, fp *result,
			   fp *toBlurDev, fp *blurredDev,
			   unsigned rows, unsigned stride, unsigned radius);

#endif // H_BLUR_CUDA not defined
