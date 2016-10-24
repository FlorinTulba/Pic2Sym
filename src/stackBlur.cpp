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

/*
Stack blurring algorithm

Note this is a different algorithm than Stacked Integral Image (SII).

Brought minor modifications (see comments from StackBlur::Impl::apply()) to:
Stack Blur Algorithm by Mario Klingemann <mario@quasimondo.com>:
http://www.codeproject.com/Articles/42192/Fast-Image-Blurring-with-CUDA
under license: http://www.codeproject.com/info/cpol10.aspx

It was included in the project since it also presents a working version for CUDA that provides
great time-performance improvement.
Credits for this CUDA version to Michael <lioucr@hotmail.com> - http://home.so-net.net.tw/lioucy
*/

#include "stackBlur.h"
#include "misc.h"

#include <algorithm>

#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

// Handle class
class StackBlur::Impl {
	friend class StackBlur;

	static const int stack_blur8_mul[];	///< array with some multipliers
	static const int stack_blur8_shr[];	///< array with some shift factors

	unsigned r = 1U;	///< filter radius (valid range: 1..254)

	Impl() {}

	/// Reconfigure the filter through a new desired standard deviation
	Impl& setSigma(double desiredSigma) {
		if(desiredSigma < 0.)
			THROW_WITH_CONST_MSG("desiredSigma should be > 0 in " __FUNCTION__, invalid_argument);

		// Empirical relation, based on which radius minimizes the L2 error
		// compared to a Gaussian with a given standard deviation
		static const double RatioR_Sigma = 2.125;
		r = unsigned(round(RatioR_Sigma * desiredSigma));

		r = max(1U, min(254U, r));

		return *this;
	}

	/// Reconfigure the filter through a new radius
	Impl& setRadius(unsigned radius) {
		if(radius == 0U || radius > 254U)
			THROW_WITH_CONST_MSG("Parameter radius must be in range 1..254 in " __FUNCTION__, invalid_argument);
		r = radius;

		return *this;
	}

	/**
	Implementation of the Stack Blur Algorithm by Mario Klingemann (<mario@quasimondo.com>)
	for single channel images with pixel of type double.

	Introduced changes:
	- processing isn't performed in-place

	- rearranged some loops

	- scaling of the result pixels uses a mul_sum that already incorporates
	  shr_sum right shifting from original implementation:
			mul_sum = stack_blur8_mul[r] / 2^stack_blur8_shr[r]

	- used a single scaling / result pixel instead of 2:
	  Resulted pixel gets multiplied only during the processing of the columns with
			mul_sum_sq = mul_sum^2, and not twice with mul_sum during each traversal. 
	*/
	void apply(const cv::Mat &toBlur, cv::Mat &blurred) const {
		double * const pResult = (double*)blurred.data;
		const double * const pToProcess = (double*)toBlur.data; // after processing the rows, it will be reassigned with a const_cast
		const unsigned w = (unsigned)toBlur.cols, h = (unsigned)toBlur.rows,
					stride = (unsigned)toBlur.step1(),
					wm1 = w - 1U, hm1 = h - 1U,
					rp1 = r + 1U, div = (r<<1) | 1U,
					xp0 = min(wm1, r), yp0 = min(hm1, r);
		const double rp2 = double(r + 2U),
					mul_sum = stack_blur8_mul[r] / pow(2, stack_blur8_shr[r]),
					mul_sum_sq = mul_sum * mul_sum; // multiply only once, during the processing of columns

		unsigned x, y, xp, yp, i, stack_ptr, stack_start;

		const double *src_pix_ptr;
		double *dst_pix_ptr, *stack_pix_ptr;
		double * const stack = new double[div];
		double sumT, sumIn, sumOut;

		y = 0U;
		do { // Process image rows, this outer while-loop will be parallel computed by CUDA instead
			// Get input and output weights
			const unsigned row_addr = y * stride;
			double pix = sumT = sumOut = stack[0] = *(src_pix_ptr = pToProcess + row_addr);
			sumIn = 0.;
			for(i = 1U; i <= xp0; ++i) {
				stack[i] = pix;
				sumT += pix * (i + 1U);  sumOut += pix;
				stack[i + r] = pix = *++src_pix_ptr;
				sumT += pix * (rp1 - i);  sumIn += pix;
			}
			if(i <= r) { // for a radius larger than image width
				const unsigned count = rp1 - i;
				const double total = pix * count;
				sumOut += total;  sumIn += total;  sumT += rp2 * total;
				fill(stack + i, stack + rp1, pix);  fill(stack + i+r, stack + div, pix);
			}

			stack_ptr = r;
			src_pix_ptr = pToProcess + ((xp = xp0) + row_addr);
			dst_pix_ptr = pResult + row_addr;

			x = 0U;
			do { // Blur image rows
				*dst_pix_ptr++ = sumT/* * mul_sum*/; // multiply only once with its square, during the processing of columns
				sumT -= sumOut;

				stack_start = stack_ptr + rp1;
				if(stack_start >= div)
					stack_start -= div;

				sumOut -= *(stack_pix_ptr = stack + stack_start);

				if(xp < wm1) {
					++xp;  ++src_pix_ptr;
				}

				sumT += (sumIn += (*stack_pix_ptr = *src_pix_ptr));
				if(++stack_ptr >= div)
					stack_ptr = 0U;

				stack_pix_ptr = stack + stack_ptr;

				sumOut += *stack_pix_ptr;
				sumIn -= *stack_pix_ptr;
			} while(++x < w); // Blur image rows
		} while(++y < h); // Process image rows, this outer while-loop will be parallel computed by CUDA instead

		// In-place update of the result for processing the columns
		const_cast<const double*>(pToProcess) = pResult;
		x = 0U;
		do { // Process image columns, this outer while-loop will be parallel computed by CUDA instead
			double pix = sumT = sumOut = stack[0] = *(src_pix_ptr = pToProcess + x);
			sumIn = 0.;
			for(i = 1U; i <= yp0; ++i) {
				stack[i] = pix;
				sumT += pix * (i + 1U);  sumOut += pix;
				stack[i + r] = pix = *(src_pix_ptr += stride);
				sumT += pix * (rp1 - i);  sumIn += pix;
			}
			if(i <= r) { // for a radius larger than image height
				const unsigned count = rp1 - i;
				const double total = pix * count;
				sumOut += total;  sumIn += total;  sumT += rp2 * total;
				fill(stack + i, stack + rp1, pix);  fill(stack + i+r, stack + div, pix);
			}

			stack_ptr = r;
			src_pix_ptr = pToProcess + (x + ((yp = yp0) * stride));
			dst_pix_ptr = pResult + x;

			y = 0U;
			do { // Blur image columns
				*dst_pix_ptr = sumT/* * mul_sum*/ * mul_sum_sq; // multiply only once with the square of mul_sum
				dst_pix_ptr += stride;

				sumT -= sumOut;
				stack_start = stack_ptr + rp1;
				if(stack_start >= div)
					stack_start -= div;

				sumOut -= *(stack_pix_ptr = stack + stack_start);

				if(yp < hm1) {
					++yp;  src_pix_ptr += stride;
				}

				sumT += (sumIn += (*stack_pix_ptr = *src_pix_ptr));
				if(++stack_ptr >= div)
					stack_ptr = 0U;

				stack_pix_ptr = stack + stack_ptr;

				sumOut += *stack_pix_ptr;
				sumIn -= *stack_pix_ptr;
			} while(++y < h); // Blur image columns
		} while(++x < w); // Process image columns, this outer while-loop will be parallel computed by CUDA instead

		delete stack;
	}
};

const int StackBlur::Impl::stack_blur8_mul[] = {
	512, 512, 456, 512, 328, 456, 335, 512, 405, 328, 271, 456, 388, 335, 292, 512,
	454, 405, 364, 328, 298, 271, 496, 456, 420, 388, 360, 335, 312, 292, 273, 512,
	482, 454, 428, 405, 383, 364, 345, 328, 312, 298, 284, 271, 259, 496, 475, 456,
	437, 420, 404, 388, 374, 360, 347, 335, 323, 312, 302, 292, 282, 273, 265, 512,
	497, 482, 468, 454, 441, 428, 417, 405, 394, 383, 373, 364, 354, 345, 337, 328,
	320, 312, 305, 298, 291, 284, 278, 271, 265, 259, 507, 496, 485, 475, 465, 456,
	446, 437, 428, 420, 412, 404, 396, 388, 381, 374, 367, 360, 354, 347, 341, 335,
	329, 323, 318, 312, 307, 302, 297, 292, 287, 282, 278, 273, 269, 265, 261, 512,
	505, 497, 489, 482, 475, 468, 461, 454, 447, 441, 435, 428, 422, 417, 411, 405,
	399, 394, 389, 383, 378, 373, 368, 364, 359, 354, 350, 345, 341, 337, 332, 328,
	324, 320, 316, 312, 309, 305, 301, 298, 294, 291, 287, 284, 281, 278, 274, 271,
	268, 265, 262, 259, 257, 507, 501, 496, 491, 485, 480, 475, 470, 465, 460, 456,
	451, 446, 442, 437, 433, 428, 424, 420, 416, 412, 408, 404, 400, 396, 392, 388,
	385, 381, 377, 374, 370, 367, 363, 360, 357, 354, 350, 347, 344, 341, 338, 335,
	332, 329, 326, 323, 320, 318, 315, 312, 310, 307, 304, 302, 299, 297, 294, 292,
	289, 287, 285, 282, 280, 278, 275, 273, 271, 269, 267, 265, 263, 261, 259
};

const int StackBlur::Impl::stack_blur8_shr[] = {
	9, 11, 12, 13, 13, 14, 14, 15, 15, 15, 15, 16, 16, 16, 16, 17,
	17, 17, 17, 17, 17, 17, 18, 18, 18, 18, 18, 18, 18, 18, 18, 19,
	19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 20, 20, 20,
	20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 21,
	21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21,
	21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 22, 22, 22, 22, 22, 22,
	22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22,
	22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 23,
	23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23,
	23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23,
	23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23,
	23, 23, 23, 23, 23, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24,
	24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24,
	24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24,
	24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24,
	24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24
};

StackBlur::Impl& StackBlur::nonTinySyms() {
	static Impl implem;
	return implem;
}

StackBlur::Impl& StackBlur::tinySyms() {
	static Impl implem;
	return implem;
}

void StackBlur::doProcess(const cv::Mat &toBlur, cv::Mat &blurred, bool forTinySym) const {
	if(forTinySym)
		tinySyms().apply(toBlur, blurred);
	else
		nonTinySyms().apply(toBlur, blurred);
}

StackBlur::StackBlur(unsigned radius) {
	setRadius(radius);
}

StackBlur& StackBlur::setSigma(double desiredSigma) {
	nonTinySyms().setSigma(desiredSigma);

	// Tiny symbols should use a sigma = desiredSigma/2.
	tinySyms().setSigma(desiredSigma * .5);

	return *this;
}

StackBlur& StackBlur::setRadius(unsigned radius) {
	nonTinySyms().setRadius(radius);

	// Tiny symbols should use half the radius from normal symbols
	tinySyms().setRadius(max((radius>>1), 1U));

	return *this;
}

const StackBlur& StackBlur::configuredInstance() {
	static StackBlur result(1U);
	static bool initialized = false;
	if(!initialized) {
		// Stack blur with desired standard deviation
		extern const double StructuralSimilarity_SIGMA;
		result.setSigma(StructuralSimilarity_SIGMA);
		initialized = true;
	}
	return result;
}
