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

Brought several modifications (see comments from StackBlurCUDA::Impl::apply()) to:
	Stack Blur Algorithm by Mario Klingemann <mario@quasimondo.com>:
	http://www.codeproject.com/Articles/42192/Fast-Image-Blurring-with-CUDA
	under license: http://www.codeproject.com/info/cpol10.aspx

It was included in the project since it also presents a working version for CUDA.
Credits for this CUDA version to Michael <lioucr@hotmail.com> - http://home.so-net.net.tw/lioucy
*/

#include "stackBlurCUDA.h"
#include "blurCUDA.h"
#include "streamsManager.h"
#include "floatType.h"
#include "warnings.h"
#include "util.h"

#pragma warning ( push, 0 )

#include <cuda_runtime.h>

#pragma warning ( pop )

#ifndef UNIT_TESTING

#pragma warning ( push, 0 )

// The project uses parallelism
#include <omp.h>

#pragma warning ( pop )

#else // UNIT_TESTING defined
// Unit Tests don't use parallelism, to ensure that at least the sequential code works as expected
extern int __cdecl omp_get_num_procs(void); // returns 1 - using a single CPU
extern int __cdecl omp_get_thread_num(void); // returns 0 - the index of the unique thread used

#endif // UNIT_TESTING

using namespace std;
using namespace cv;

/// Delegates the Stack blur execution to CUDA
class StackBlurCUDAImpl : public AbsStackBlurImpl {
	const size_t maxDataPixels;	///< max size of the image to blur in pixels

	fp *toBlurDev = nullptr;	///< region allocated on device for the image to blur
	fp *blurredDev = nullptr;	///< region allocated on device for the resulted blur

public:
	/// Allocates space (on the device) for the image to be blurred and the result
	StackBlurCUDAImpl(bool forTinySyms) : AbsStackBlurImpl(),
			maxDataPixels(forTinySyms ? BlurEngine::PixelsCountTinySym() : BlurEngine::PixelsCountLargestData()) {
		const size_t buffSz = maxDataPixels * sizeof(fp) * StreamsManager::streams().count(); // one separate buffer for each stream
		CHECK_OP(cudaMalloc((void**)&toBlurDev, buffSz));
		CHECK_OP(cudaMalloc((void**)&blurredDev, buffSz));
	}

	/// Releases the space for the image to be blurred and the result
	~StackBlurCUDAImpl() {
		CHECK_OP_NO_THROW(cudaFree(blurredDev));
		CHECK_OP_NO_THROW(cudaFree(toBlurDev));
	}

	/**
	Implementation of the Stack Blur Algorithm by Mario Klingemann (<mario@quasimondo.com>)
	for single channel images with pixel of type fp.

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
	void apply(const cv::Mat &toBlur, cv::Mat &blurred) const override {
		assert(toBlur.type() == CV_32FC1);

		// Input and output data are mapped in the device memory separately for each stream - ioOffset from below provides the location
		const auto cpuId = omp_get_thread_num(); // There's a stream for each CPU
		const auto ioOffset = cpuId * maxDataPixels;

		stackBlur((const fp*)toBlur.data, (fp*)blurred.data,
				  &toBlurDev[ioOffset], &blurredDev[ioOffset],
				  (unsigned)toBlur.rows, (unsigned)toBlur.step[0], r);
	}
};

 AbsStackBlurImpl& StackBlurCUDA::nonTinySyms() {
	static StackBlurCUDAImpl impl(false);
	return impl;
}

AbsStackBlurImpl& StackBlurCUDA::tinySyms() {
	static StackBlurCUDAImpl impl(true);
	return impl;
}

StackBlurCUDA::StackBlurCUDA(unsigned radius) : TStackBlur<StackBlurCUDA>(radius) {}

