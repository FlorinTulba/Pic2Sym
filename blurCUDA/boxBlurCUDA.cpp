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

#include "boxBlurCUDA.h"
#include "blurCUDA.h"
#include "streamsManager.h"
#include "floatType.h"
#include "warnings.h"
#include "misc.h"
#include "util.h"

#pragma warning ( push, 0 )

#include <cuda_runtime.h>

#include <vector>
#include <algorithm>

#include <omp.h>

#pragma warning ( pop )

using namespace std;
using namespace cv;

unsigned BoxBlurCUDA::BlockDimRows() { return 64U; }

unsigned BoxBlurCUDA::BlockDimCols() { return 32U; }

bool BoxBlurCUDA::preconditionsOk() {
	cudaDeviceProp deviceProps;
	if(!cudaInitOk(&deviceProps))
		return false;

	extern const unsigned Settings_MAX_FONT_SIZE;
	const size_t requiredShMem = size_t(2U * BlockDimCols() * Settings_MAX_FONT_SIZE) * sizeof(fp);
	if(requiredShMem  >  deviceProps.sharedMemPerBlock) {
		cerr<<"Current configuration of the 'boxCUDA' algorithm requires more shared device memory per block ("
			<<requiredShMem<<") than available ("<<deviceProps.sharedMemPerBlock<<")!"<<endl;
		return false;
	}

	return true;
}

BoxBlurCUDA::BoxBlurCUDA(unsigned boxWidth_/* = 1U*/, unsigned iterations_/* = 1U*/) :
		TBoxBlur<BoxBlurCUDA>(boxWidth_, iterations_) {
	if(!preconditionsOk())
		THROW_WITH_CONST_MSG("Preconditions for creating BoxBlurCUDA are not met!", logic_error);
}

/// Delegates the Box blur execution to CUDA
class BoxBlurCUDAImpl : public AbsBoxBlurImpl {
	const size_t maxDataPixels;	///< max size of the image to blur in pixels

	fp *toBlurDev = nullptr;	///< region allocated on device for the image to blur (different for every stream)
	fp *blurredDev = nullptr;	///< region allocated on device for the resulted blur (different for every stream)

	// Next masks remain constant for all blur operations performed based on the given configuration.
	// However, they can't go to constant device memory, since their number (the iterations count)
	// is variable, so the information about how much constant device memory to allocate is unknown.
	vector<unsigned> maskWidths;		///< widths of the masks used within each iteration
	unsigned *maskWidthsDev = nullptr;	///< region allocated on device for the widths of the masks used within each iteration
	unsigned largestMaskRadius = 1U;	///< the radius of the largest mask used during the iterations
	fp scaler = 1.f;					///< factor to rescale the result in the same range as the original

	/// Keeps maskWidths & maskWidthsDev in sync with the inherited fields iterations, wl, wu, countWl, countWu
	void updateMasks() {
		if(maskWidthsDev != nullptr)
			CHECK_OP(cudaFree(maskWidthsDev));

		assert(wl > 1U || (wl == 1U && countWl == 0U));
		assert(wl + 2U == wu);

		largestMaskRadius = wu >> 1;

		maskWidths.resize(iterations);
		auto itBegin = begin(maskWidths), itEnd = end(maskWidths);
		auto it = next(itBegin, countWl);
		fill(itBegin, it, wl); fill(it, itEnd, wu);

		/*
		Each iteration normally needs to scale the result by 1/maskWidth.
		The adopted solution performs only a final scaling based on these facts:
		- there are iterationsWithLowerWidthMask and iterationsWithUpperWidthMask
		- each mask is applied twice (once horizontally and once vertically)
		*/
		scaler = 1.f / fp(pow(wl, countWl<<1U) * pow(wu, countWu<<1U));

		const size_t maskWidthsSz = sizeof(unsigned) * iterations;
		CHECK_OP(cudaMalloc((void**)&maskWidthsDev, maskWidthsSz));
		CHECK_OP(cudaMemcpy((void*)maskWidthsDev, (void*)maskWidths.data(), maskWidthsSz, cudaMemcpyHostToDevice));
	}

public:
	/// Allocates space (on the device) for the image to be blurred and the result
	BoxBlurCUDAImpl(bool forTinySyms) : AbsBoxBlurImpl(),
			maxDataPixels(forTinySyms ? BlurEngine::PixelsCountTinySym() : BlurEngine::PixelsCountLargestData()) {
		const size_t buffSz = maxDataPixels * sizeof(fp) * StreamsManager::streams().count(); // one separate buffer for each stream
		CHECK_OP(cudaMalloc((void**)&toBlurDev, buffSz));
		CHECK_OP(cudaMalloc((void**)&blurredDev, buffSz));

		updateMasks();
	}

	/// Releases the space for the image to be blurred and the result
	~BoxBlurCUDAImpl() {
		CHECK_OP_NO_THROW(cudaFree(blurredDev));
		CHECK_OP_NO_THROW(cudaFree(toBlurDev));
		CHECK_OP_NO_THROW(cudaFree(maskWidthsDev));
	}

	/// Reconfigure the filter through a new desired standard deviation and a new iterations count
	/// See http://www.web.uwa.edu.au/__data/assets/file/0008/826172/filterdesign.pdf for details
	AbsBoxBlurImpl& setSigma(double desiredSigma, unsigned iterations_ = 1U) override {
		AbsBoxBlurImpl::setSigma(desiredSigma, iterations_);
		updateMasks();
		return *this;
	}

	/// Reconfigure mask width (wl) for performing all iterations and destroys the wu mask
	AbsBoxBlurImpl& setWidth(unsigned boxWidth_) override {
		AbsBoxBlurImpl::setWidth(boxWidth_);;
		updateMasks();
		return *this;
	}

	/// Reconfigure iterations count for wl and destroys the wu mask
	AbsBoxBlurImpl& setIterations(unsigned iterations_) override {
		if(iterations != iterations_) {}
		AbsBoxBlurImpl::setIterations(iterations_);
		updateMasks();
		return *this;
	}

	/// Actual implementation for the current configuration. toBlur is checked; blurred is initialized
	/// See http://www.web.uwa.edu.au/__data/assets/file/0008/826172/filterdesign.pdf for details
	void apply(const Mat &toBlur, Mat &blurred) override {
		assert(toBlur.type() == CV_32FC1);

		if(iterations == 0U) {
			toBlur.copyTo(blurred);
			return;
		}

		const size_t buffSz = size_t(toBlur.rows * toBlur.cols) * sizeof(fp);

		// Input and output data are mapped in the device memory separately for each stream - ioOffset from below provides the location
		const auto cpuId = omp_get_thread_num(); // There's a stream for each CPU
		const auto ioOffset = cpuId * maxDataPixels;
		const auto streamId = StreamsManager::streams()[cpuId];

		boxBlur((const fp*)toBlur.data, (fp*)blurred.data,
				&toBlurDev[ioOffset], &blurredDev[ioOffset],
				(unsigned)toBlur.rows, (unsigned)toBlur.cols, buffSz,
				maskWidthsDev, iterations, largestMaskRadius,
				scaler, streamId);
	}
};

AbsBoxBlurImpl& BoxBlurCUDA::nonTinySyms() {
#pragma warning ( disable : WARN_THREAD_UNSAFE )
	static BoxBlurCUDAImpl impl(false);
#pragma warning ( default : WARN_THREAD_UNSAFE )
	return impl;
}

AbsBoxBlurImpl& BoxBlurCUDA::tinySyms() {
#pragma warning ( disable : WARN_THREAD_UNSAFE )
	static BoxBlurCUDAImpl impl(true);
#pragma warning ( default : WARN_THREAD_UNSAFE )
	return impl;
}

