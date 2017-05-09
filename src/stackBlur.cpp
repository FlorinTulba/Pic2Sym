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

/*
Stack blurring algorithm

Note this is a different algorithm than Stacked Integral Image (SII).

Brought several modifications (see comments from StackBlur::Impl::apply()) to:
	Stack Blur Algorithm by Mario Klingemann <mario@quasimondo.com>:
	http://www.codeproject.com/Articles/42192/Fast-Image-Blurring-with-CUDA
	under license: http://www.codeproject.com/info/cpol10.aspx

It was included in the project since it also presents a working version for CUDA.
*/

#include "stackBlur.h"
#include "floatType.h"
#include "warnings.h"

#pragma warning ( push, 0 )

#include <algorithm>

#include <opencv2/imgproc/imgproc.hpp>

#pragma warning ( pop )

using namespace std;
using namespace cv;

/// Specialization of AbsStackBlurImpl
class StackBlurImpl : public AbsStackBlurImpl {
public:
	/**
	Implementation of the Stack Blur Algorithm by Mario Klingemann (<mario@quasimondo.com>)
	for single channel images with pixel of type fp.

	Introduced changes:
	- processing isn't performed in-place

	- rearranged some loops

	- scaling of the result pixels uses a mul_sum that already incorporates
	  shr_sum right shifting from original implementation:
			mul_sum = multipliers[r] / 2^shiftFactors[r]

	- used a single scaling / result pixel instead of 2:
	  Resulted pixel gets multiplied only during the processing of the columns with
			mul_sum_sq = mul_sum^2, and not twice with mul_sum during each traversal.
	*/
	void apply(const cv::Mat &toBlur, cv::Mat &blurred) const override {
		static const unsigned fpSz = (unsigned)sizeof(float);
		fp * const pResult = (fp*)blurred.data;
		const fp * const pToProcess = (fp*)toBlur.data; // after processing the rows, it will be reassigned with a const_cast
		const unsigned w = (unsigned)toBlur.cols, h = (unsigned)toBlur.rows,
			stride = (unsigned)toBlur.step[0],
			wm1 = w - 1U, hm1 = h - 1U,
			rp1 = r + 1U, div = (r<<1) | 1U,
			xp0 = min(wm1, r), yp0 = min(hm1, r);
		const fp rp2 = fp(r + 2U),
			mul_sum = multipliers[r] / fp(1 << shiftFactors[r]),
			mul_sum_sq = mul_sum * mul_sum; // multiply only once, during the processing of columns

		unsigned x, y, xp, yp, i, stack_ptr, stack_start;

		unsigned char *auxPtr = nullptr;
		const fp *src_pix_ptr;
		fp *dst_pix_ptr, *stack_pix_ptr;
		fp * const stack = new fp[div];
		fp sumT, sumIn, sumOut;

		y = 0U;
		do { // Process image rows, this outer while-loop will be parallel computed by CUDA instead
			// Get input and output weights
			const unsigned row_addr = y * stride;
			auxPtr = (unsigned char*)pToProcess + row_addr;
			fp pix = sumT = sumOut = stack[0] = *(src_pix_ptr = (fp*)auxPtr);
			sumIn = 0.;
			for(i = 1U; i <= xp0; ++i) {
				stack[i] = pix;
				sumT += pix * (i + 1U);  sumOut += pix;
				stack[i + r] = pix = *++src_pix_ptr;
				sumT += pix * (rp1 - i);  sumIn += pix;
			}
			if(i <= r) { // for a radius larger than image width
				const unsigned count = rp1 - i;
				const fp total = pix * count;
				sumOut += total;  sumIn += total;  sumT += rp2 * total;
				fill(stack + i, stack + rp1, pix);  fill(stack + i+r, stack + div, pix);
			}

			stack_ptr = r;
			auxPtr += fpSz * (xp = xp0);
			src_pix_ptr = (fp*)auxPtr;
			auxPtr = (unsigned char*)pResult + row_addr;
			dst_pix_ptr = (fp*)auxPtr;

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
#pragma warning ( disable : WARN_LVALUE_CAST )
		const_cast<const fp*>(pToProcess) = pResult;
#pragma warning ( default : WARN_LVALUE_CAST )
		x = 0U;
		do { // Process image columns, this outer while-loop will be parallel computed by CUDA instead
			fp pix = sumT = sumOut = stack[0] = *(src_pix_ptr = pToProcess + x);
			sumIn = 0.;
			for(i = 1U; i <= yp0; ++i) {
				stack[i] = pix;
				sumT += pix * (i + 1U);  sumOut += pix;
				auxPtr = (unsigned char*)src_pix_ptr + stride;
				stack[i + r] = pix = *(src_pix_ptr = (fp*)auxPtr);
				sumT += pix * (rp1 - i);  sumIn += pix;
			}
			if(i <= r) { // for a radius larger than image height
				const unsigned count = rp1 - i;
				const fp total = pix * count;
				sumOut += total;  sumIn += total;  sumT += rp2 * total;
				fill(stack + i, stack + rp1, pix);  fill(stack + i+r, stack + div, pix);
			}

			stack_ptr = r;
			auxPtr = (unsigned char*)pToProcess + (x * fpSz + ((yp = yp0) * stride));
			src_pix_ptr = (fp*)auxPtr;
			dst_pix_ptr = pResult + x;

			y = 0U;
			do { // Blur image columns
				*dst_pix_ptr = sumT/* * mul_sum*/ * mul_sum_sq; // multiply only once with the square of mul_sum
				auxPtr = (unsigned char*)dst_pix_ptr + stride;
				dst_pix_ptr = (fp*)auxPtr;

				sumT -= sumOut;
				stack_start = stack_ptr + rp1;
				if(stack_start >= div)
					stack_start -= div;

				sumOut -= *(stack_pix_ptr = stack + stack_start);

				if(yp < hm1) {
					++yp;
					auxPtr = (unsigned char*)src_pix_ptr + stride;
					src_pix_ptr = (fp*)auxPtr;
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

AbsStackBlurImpl& StackBlur::nonTinySyms() {
	static StackBlurImpl impl;
	return impl;
}

AbsStackBlurImpl& StackBlur::tinySyms() {
	static StackBlurImpl impl;
	return impl;
}

StackBlur::StackBlur(unsigned radius) : TStackBlur<StackBlur>(radius) {}
