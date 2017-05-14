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

BoxBlurCUDAmin::BoxBlurCUDAmin(unsigned boxWidth_/* = 1U*/, unsigned iterations_/* = 1U*/) :
		TBoxBlur<BoxBlurCUDAmin>(boxWidth_, iterations_) {}

/// Delegates the Box blur execution to CUDA
class AbsBoxBlurCUDA /*abstract*/ : public AbsBoxBlurImpl {
protected:
	const size_t maxDataPixels;			///< max size of the image to blur in pixels

	// Next masks remain constant for all blur operations performed based on the given configuration.
	vector<unsigned> maskWidths;		///< widths of the masks used within each iteration

	fp scaler = 1.f;					///< factor to rescale the result in the same range as the original

	/// Keeps maskWidths in sync with the inherited fields iterations, wl, wu, countWl, countWu
	virtual void updateMasks() {
		assert(wl > 1U || (wl == 1U && countWl == 0U));
		assert(wl + 2U == wu);

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
	}

	/*virtual*/ ~AbsBoxBlurCUDA() {}

public:
	/// Allocates space (on the device) for the image to be blurred and the result
	AbsBoxBlurCUDA(bool forTinySyms) : AbsBoxBlurImpl(),
			maxDataPixels(forTinySyms ? BlurEngine::PixelsCountTinySym() : BlurEngine::PixelsCountLargestData()) {
		updateMasks();
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
};


/// Delegates the entire Box blur execution to CUDA
class BoxBlurCUDAImpl : public AbsBoxBlurCUDA {
	fp *toBlurDev = nullptr;	///< region allocated on device for the image to blur (different for every stream)
	fp *blurredDev = nullptr;	///< region allocated on device for the resulted blur (different for every stream)

	// Next masks remain constant for all blur operations performed based on the given configuration.
	// However, they can't go to constant device memory, since their number (the iterations count)
	// is variable, so the information about how much constant device memory to allocate is unknown.
	unsigned *maskWidthsDev = nullptr;	///< region allocated on device for the widths of the masks used within each iteration
	unsigned largestMaskRadius = 1U;	///< the radius of the largest mask used during the iterations

	/// Keeps maskWidths & maskWidthsDev in sync with the inherited fields iterations, wl, wu, countWl, countWu
	void updateMasks() override {
		if(maskWidthsDev != nullptr)
			CHECK_OP(cudaFree(maskWidthsDev));

		AbsBoxBlurCUDA::updateMasks();

		largestMaskRadius = wu >> 1;

		const size_t maskWidthsSz = sizeof(unsigned) * iterations;
		CHECK_OP(cudaMalloc((void**)&maskWidthsDev, maskWidthsSz));
		CHECK_OP(cudaMemcpy((void*)maskWidthsDev, (void*)maskWidths.data(), maskWidthsSz, cudaMemcpyHostToDevice));
	}

public:
	/// Allocates space (on the device) for the image to be blurred and the result
	BoxBlurCUDAImpl(bool forTinySyms) : AbsBoxBlurCUDA(forTinySyms) {
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


/// Performs the blur mostly on the CPU, but initializes asynchronously the rolling sums on the GPU
class BoxBlurCUDAminImpl : public AbsBoxBlurCUDA {
	fp *dataDev = nullptr;	///< region allocated on device for processing the rolling sums

	/**
	Performs the blur column-wise.

	@param blurred the result without scaling
	@param dataDev space in the device global memory (for current stream) allocated for computing the blur
	@param streamId the id of the stream performing this blur operation
	*/
	void blurColumnwise(Mat &blurred, fp *dataDev, cudaStream_t streamId) {
		unsigned rows = (unsigned)blurred.rows,
			rowsM1 = rows - 1U,
			cols = (unsigned)blurred.cols,
			maskWidth = maskWidths[0ULL], // there's at most 1 change of maskWidth
			maskRadius = maskWidth >> 1U,
			nextMaskRadius = maskRadius,
			min_maskRadius_rowsM1 = min(maskRadius, rowsM1),
			min_nextMaskRadius_rowsM1 = min_maskRadius_rowsM1,
			providedRows = min_nextMaskRadius_rowsM1 + 1U;
		static const size_t fpSz = sizeof(fp);
		const size_t firstRowDataSz = cols * fpSz;
		size_t maskRadiusRowsDataSz = firstRowDataSz * min_maskRadius_rowsM1,
			nextMaskRadiusRowsDataSz = maskRadiusRowsDataSz;

		// Copy 1st row
		Mat firstRow = blurred.row(0).clone(), nextRelevantRows;
		CHECK_OP(cudaMemcpyAsync((void*)dataDev, (void*)firstRow.data,
			firstRowDataSz, cudaMemcpyHostToDevice, streamId));

		nextRelevantRows = blurred.rowRange(1, (int)providedRows).clone();
		computeInitialRollingSums((fp*)firstRow.data, firstRowDataSz,
								  (fp*)nextRelevantRows.data, maskRadiusRowsDataSz,
								  dataDev, rows,
								  providedRows, maskRadius,
								  streamId);

		assert(iterations >= 1U);
		for(unsigned iter = 0U, lastIter = iterations - 1U; iter < iterations; ++iter) {
#pragma region PrepareMasks
			// Check if current mask is different than the previous one (this might happen only once)
			if(iter > 0U && maskWidth != maskWidths[iter]) {
				maskWidth = maskWidths[iter];
				maskRadius = nextMaskRadius = maskWidth >> 1U;
				min_maskRadius_rowsM1 = min_nextMaskRadius_rowsM1 = min(maskRadius, rowsM1);
				maskRadiusRowsDataSz = nextMaskRadiusRowsDataSz = firstRowDataSz * min_maskRadius_rowsM1;
				providedRows = min_nextMaskRadius_rowsM1 + 1U;
			}

			// Check if next mask is different than the current one (this might happen only once)
			if(iter != lastIter && maskWidth != maskWidths[iter + 1U]) {
				nextMaskRadius = maskWidths[iter + 1U] >> 1U;
				min_nextMaskRadius_rowsM1 = min(nextMaskRadius, rowsM1);
				nextMaskRadiusRowsDataSz = firstRowDataSz * min_nextMaskRadius_rowsM1;
				providedRows = min_nextMaskRadius_rowsM1 + 1U;
			}
#pragma endregion PrepareMasks

			Mat blurredNew = blurred.clone(), // result for this iteration
				oldFirstRow = blurred.row(0),
				oldLastRow = blurred.row((int)rowsM1);
			CHECK_OP(cudaStreamSynchronize(streamId)); // firstRow gets updated with the rolling sums

			// Rolling sums are the first row from the result
			Mat rollingSums = firstRow.clone();

#pragma region PrepareDataForNextRollingSums
			rollingSums.copyTo(blurredNew.row(0));
			unsigned row = 1U, frontIdx = row + nextMaskRadius;
			for(; row <= min_nextMaskRadius_rowsM1; ++row, ++frontIdx) {
				rollingSums += (frontIdx < rows ? blurred.row(int(frontIdx)) : oldLastRow) -
					oldFirstRow;
				rollingSums.copyTo(blurredNew.row(int(row)));
			}
#pragma endregion PrepareDataForNextRollingSums

			// Except for the last iteration, ask the GPU to compute the rolling sums for the next iteration
			if(iter != lastIter) {
				nextRelevantRows = blurredNew.rowRange(1, (int)providedRows).clone();
				computeInitialRollingSums((fp*)firstRow.data, firstRowDataSz,
										  (fp*)nextRelevantRows.data, nextMaskRadiusRowsDataSz,
										  dataDev, rows,
										  providedRows, nextMaskRadius,
										  streamId);
			}

			// compute last rows on the CPU
			for(int tailIdx = int(row - nextMaskRadius) - 1; row < rows; ++row, ++frontIdx, ++tailIdx) {
				rollingSums += (frontIdx < rows ? blurred.row(int(frontIdx)) : oldLastRow) -
					blurred.row(tailIdx);
				rollingSums.copyTo(blurredNew.row(int(row)));
			}

			blurred = blurredNew;
		}
	}

public:
	BoxBlurCUDAminImpl(bool forTinySyms) : AbsBoxBlurCUDA(forTinySyms) {
		const size_t buffSz = maxDataPixels * sizeof(fp) * StreamsManager::streams().count(); // one separate buffer for each stream
		CHECK_OP(cudaMalloc((void**)&dataDev, buffSz));
	}

	/// Releases the space for the image to be blurred and the result
	~BoxBlurCUDAminImpl() {
		CHECK_OP_NO_THROW(cudaFree(dataDev));
	}

	/// Actual implementation for the current configuration. toBlur is checked; blurred is initialized
	/// See http://www.web.uwa.edu.au/__data/assets/file/0008/826172/filterdesign.pdf for details
	void apply(const Mat &toBlur, Mat &blurred) override {
		assert(toBlur.type() == CV_32FC1);

		toBlur.copyTo(blurred);

		if(iterations == 0U)
			return;

		// Input and output data are mapped in the device memory separately for each stream - dataOffset from below provides the location
		const auto cpuId = omp_get_thread_num(); // There's a stream for each CPU
		const size_t dataOffset = cpuId * maxDataPixels;
		const auto streamId = StreamsManager::streams()[cpuId];

		blurColumnwise(blurred, dataDev + dataOffset, streamId);

		// blur row-wise by using the column-wise approach on the transpose and then transposing back the result
		// The 2 transpose operations are barely noticeable on the obtained timing
		transpose(blurred, blurred);
		blurColumnwise(blurred, dataDev + dataOffset, streamId);
		transpose(blurred, blurred);

		blurred *= scaler;
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

AbsBoxBlurImpl& BoxBlurCUDAmin::nonTinySyms() {
#pragma warning ( disable : WARN_THREAD_UNSAFE )
	static BoxBlurCUDAminImpl impl(false);
#pragma warning ( default : WARN_THREAD_UNSAFE )
	return impl;
}

AbsBoxBlurImpl& BoxBlurCUDAmin::tinySyms() {
#pragma warning ( disable : WARN_THREAD_UNSAFE )
	static BoxBlurCUDAminImpl impl(true);
#pragma warning ( default : WARN_THREAD_UNSAFE )
	return impl;
}

