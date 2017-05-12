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

#include "blurCUDA.h"
#include "stackBlurCUDA.h"
#include "streamsManager.h"
#include "util.h"

#include <cuda_runtime.h>

#include <cassert>

#include <omp.h>

#define THREADS			64U
#define ROW_CHUNK_SZ	16U
static const unsigned ROW_CHUNK_SZm1 = ROW_CHUNK_SZ - 1U;

using namespace std;

/// Values to be placed in the constant memory from the device
struct StackBlurConstants {
	unsigned hm1, wm1;
	unsigned minHm1R, minWm1R;
	unsigned complMinHm1R, complMinWm1R;
	unsigned ROW_CHUNK_SZp1pMinWm1R;
	unsigned stride;
	unsigned stackSz;
	unsigned r, rp1;
	fp rp2;
};

/// Since the StackBlurConstants needs to be created also on the device, this serves for its construction
void initStackBlurConstants(StackBlurConstants &vals,
							const unsigned rows, const unsigned width, const unsigned stride,
							const unsigned r) {
	vals.stride = stride;
	vals.hm1 = rows - 1U; vals.wm1 = width - 1U;
	vals.stackSz = 2U * r + 1U;
	vals.r = r; vals.rp1 = r + 1U; vals.rp2 = fp(r + 2U);
	vals.minWm1R = min(vals.wm1, r); vals.minHm1R = min(vals.hm1, r);
	vals.complMinWm1R = vals.wm1 - vals.minWm1R; vals.complMinHm1R = vals.hm1 - vals.minHm1R;
	vals.ROW_CHUNK_SZp1pMinWm1R = vals.minWm1R + 1U + ROW_CHUNK_SZ;
}

__constant__ StackBlurConstants constantsDev; ///< the constants allocated on device

/// Moves the stack pointer and updates sumT, sumIn and sumOut
__device__
void updateStackAndSums(const fp value, fp * const stack, unsigned &stackIdx,
						fp &sumIn, fp &sumOut, fp &sumT) {
	sumT -= sumOut;

	fp * const ptrStackElem = stack + ((stackIdx + constantsDev.rp1) % constantsDev.stackSz);
	sumOut -= *ptrStackElem;
	sumT += (sumIn += (*ptrStackElem = value));

	const fp stackElem = stack[ stackIdx = (stackIdx + 1U) % constantsDev.stackSz ];
	sumOut += stackElem;
	sumIn -= stackElem;
}

/// Copies parts from several rows from the imgBuff (from global device memory) into shared threads block memory
__device__
void readInputDataChunk(const fp * const __restrict__ imgBuff,
						const unsigned chunkStartCol, const unsigned startRow, const unsigned buffDestCol,
						fp * const __restrict__ inpBuff, const unsigned fpSz) {
	const unsigned srcCol = chunkStartCol + buffDestCol;

	// Prevents inpBuff overwritting while it's still read.
	// The accidental overwrite might be triggered by a final warp formed of solely disabled threads (all 32 skip the body of if(enabled) {})
	__syncthreads();

	if(srcCol <= constantsDev.wm1) {
		const unsigned maxRow = min(constantsDev.hm1, (startRow + ROW_CHUNK_SZm1));
		fp *inpBuffPtr = inpBuff + (startRow % THREADS) * ROW_CHUNK_SZ + buffDestCol;
		const unsigned char *imgBuffPtr = (const unsigned char*)imgBuff + startRow * constantsDev.stride + srcCol * fpSz;
		for(unsigned row = startRow; row <= maxRow; ++row, inpBuffPtr += ROW_CHUNK_SZ, imgBuffPtr += constantsDev.stride) {
			*inpBuffPtr = *((fp*)imgBuffPtr);
		}
	}

	__syncthreads(); // Ensures that all relevant data was copied to inpBuff before continuing
}

/// Copies available results from shared threads block memory to the global device memory
__device__
void reportAvailableResults(fp * const __restrict__ resBuff,
							const unsigned chunkStartCol, const unsigned startRow, const unsigned buffDestCol,
							fp * const __restrict__ outBuff, const unsigned fpSz) {
	const unsigned resCol = chunkStartCol + buffDestCol;

	// Prevents outBuff .
	// The accidental broadcast might be triggered by a final warp formed of solely disabled threads (all 32 skip the body of if(enabled) {})
	__syncthreads();

	if(resCol <= constantsDev.wm1) {
		const unsigned maxRow = min(constantsDev.hm1, (startRow + ROW_CHUNK_SZm1));
		fp *outBuffPtr = outBuff + (startRow % THREADS) * ROW_CHUNK_SZ + buffDestCol;
		unsigned char *resBuffPtr = (unsigned char*)resBuff + startRow * constantsDev.stride + resCol * fpSz;
		for(unsigned row = startRow; row <= maxRow; ++row, outBuffPtr += ROW_CHUNK_SZ, resBuffPtr += constantsDev.stride) {
			*((fp*)resBuffPtr) = *outBuffPtr;
		}
	}

	__syncthreads(); // Ensures that all relevant data was broadcasted from outBuff before continuing
}

/// Shared threads block memory reports available results to global memory and overwrites that region with new input data
__device__
void reportAvailableResultsAndOverwriteWithInputDataChunk(const fp * const __restrict__ imgBuff, 
															fp * const __restrict__ resBuff,
															const unsigned chunkStartCol,
															const unsigned startRow,
															const unsigned buffDestCol,
															fp * const __restrict__ ioBuff,
															const unsigned fpSz) {
	const unsigned srcCol = chunkStartCol + buffDestCol;
	const unsigned maxRow = min(constantsDev.hm1, (startRow + ROW_CHUNK_SZm1));
	fp * const startRowIoBuffPtr = ioBuff + (startRow % THREADS) * ROW_CHUNK_SZ + buffDestCol;

	// Prevents:
	// - ioBuff overwritting while it's still read
	// - or its broadcasting while it's still written.
	// The accidental overwrite/broadcast might be triggered by a final warp formed of solely disabled threads (all 32 skip the body of if(enabled) {})
	__syncthreads();

	// reporting available results
	if(srcCol >= constantsDev.ROW_CHUNK_SZp1pMinWm1R) { // output is (constantsDev.minWm1R + 1U) pixels behind the input
		const unsigned resCol = srcCol - constantsDev.ROW_CHUNK_SZp1pMinWm1R;
		unsigned char *resBuffPtr = (unsigned char*)resBuff + startRow * constantsDev.stride + resCol * fpSz;
		const fp *ioBuffPtr = startRowIoBuffPtr;
		for(unsigned row = startRow; row <= maxRow; ++row, ioBuffPtr += ROW_CHUNK_SZ, resBuffPtr += constantsDev.stride) {
			*((fp*)resBuffPtr) = *ioBuffPtr;
		}
	}

	// overwritting with a new chunk of input data
	if(srcCol <= constantsDev.wm1) {
		const unsigned char *imgBuffPtr = (const unsigned char*)imgBuff + startRow * constantsDev.stride + srcCol * fpSz;
		fp *ioBuffPtr = startRowIoBuffPtr;
		for(unsigned row = startRow; row <= maxRow; ++row, ioBuffPtr += ROW_CHUNK_SZ, imgBuffPtr += constantsDev.stride) {
			*ioBuffPtr = *((fp*)imgBuffPtr);
		}
	}

	__syncthreads(); // Ensures that all relevant data was copied to/broadcasted from ioBuff before continuing
}

/// Constructs the stack and the sums required by the algorithm, based on a small part of the input data
__device__
void prepareStackAndSums(fp * const __restrict__ inpBuff, fp * const __restrict__ stack,
						fp &sumIn, fp &sumOut, fp &sumT, unsigned &chunkStartCol,
						const fp * const __restrict__ imgBuff,
						const unsigned rowRangeStart, const unsigned buffDestCol,
						const bool enabled, const unsigned fpSz) {
	readInputDataChunk(imgBuff, chunkStartCol, rowRangeStart, buffDestCol, inpBuff, fpSz);

	fp pix = 0.f;
	const fp * const ptrIn = inpBuff + ROW_CHUNK_SZ * threadIdx.x;
	if(enabled) {
		pix = sumT = sumOut = *stack = *ptrIn;
	}

	unsigned col = 1U;
	for(;;) {
		const unsigned limCol = min(constantsDev.minWm1R, (chunkStartCol + ROW_CHUNK_SZm1));
		if(enabled) {
			for(; col <= limCol; ++col) {
				stack[col] = pix;
				sumT += pix * (col + 1U);  sumOut += pix;
				stack[col + constantsDev.r] = pix = ptrIn[col % ROW_CHUNK_SZ];
				sumT += pix * (constantsDev.rp1 - col);  sumIn += pix;
			}
		}

		if(limCol == constantsDev.minWm1R)
			break;
		
		readInputDataChunk(imgBuff, chunkStartCol += ROW_CHUNK_SZ, rowRangeStart, buffDestCol, inpBuff, fpSz);
	}

	if(enabled && col <= constantsDev.r) {
		const fp total = pix * (constantsDev.rp1 - col);
		sumOut += total;  sumIn += total;  sumT += constantsDev.rp2 * total;

		for(; col <= constantsDev.r; ++col) {
			stack[col] = stack[col + constantsDev.r] = pix;
		}
	}
}

/// Continues the algorithm after the preparation of the stack and sums
__device__
void tackleRows(const fp * const __restrict__ imgBuff,
				fp * const __restrict__ resBuff,
				fp * const __restrict__ ioBuff,
				fp * const __restrict__ stack, fp &sumIn, fp &sumOut, fp &sumT,
				const unsigned row, const unsigned rowRangeStart, const unsigned buffDestCol, 
				unsigned &chunkStartCol,
				const bool enabled, const unsigned fpSz) {
	unsigned col = constantsDev.minWm1R + 1U, stackIdx = constantsDev.r;
	const unsigned remStartCol_RowChunkSz = col % ROW_CHUNK_SZ;
	fp pix = 0.f;
	fp * const ioBuffRow = ioBuff + (row % THREADS) * ROW_CHUNK_SZ;
	fp *ptrIo = ioBuffRow + remStartCol_RowChunkSz;

	if(remStartCol_RowChunkSz == 0U) { // start col is a multiple of ROW_CHUNK_SZ
		// A new chunk needs to be read and there is nothing to report yet
		readInputDataChunk(imgBuff, chunkStartCol += ROW_CHUNK_SZ, rowRangeStart, buffDestCol, ioBuff, fpSz);
	}

	for(;;) {
		const unsigned limCol = min(constantsDev.wm1, (chunkStartCol + ROW_CHUNK_SZm1));

		if(enabled) {
			for(; col <= limCol; ++col) {
				const fp prevSumT = sumT/* * mul_sum*/; // multiply only once with its square, during the processing of columns
				updateStackAndSums(pix = *ptrIo, stack, stackIdx, sumIn, sumOut, sumT);
				*ptrIo++ = prevSumT; // replace with the value obtained for the pixel at (constantsDev.minWm1R + 1) positions to the left
			}
		} else {
			col = limCol + 1U;
		}

		if(limCol == constantsDev.wm1)
			break;

		reportAvailableResultsAndOverwriteWithInputDataChunk(imgBuff, resBuff, 
															 chunkStartCol += ROW_CHUNK_SZ, 
															 rowRangeStart, buffDestCol, ioBuff, fpSz);
		
		ptrIo = ioBuffRow; // no need to add here (col % ROW_CHUNK_SZ), since col is now a multiple of ROW_CHUNK_SZ
	}

	// col is at this point equal to the image width

	// However, the last output columns make use of 'col' that makes sense for them: (constantsDev.minWm1R + 1U) points behind the source columns
	chunkStartCol -= constantsDev.minWm1R + 1U; 

	if(col % ROW_CHUNK_SZ == 0U) { // the col from here refers still to source columns and is equal to image width
		// Reporting a fully processed chunk before computing the last columns
		reportAvailableResults(resBuff, chunkStartCol, rowRangeStart, buffDestCol, ioBuff, fpSz);
		chunkStartCol += ROW_CHUNK_SZ;

		ptrIo = ioBuffRow; // no need to add here (col % ROW_CHUNK_SZ), since col is now a multiple of ROW_CHUNK_SZ

	} else if(col == 1U) { // the col from here refers still to source columns and is equal to image width
		// For images with a single column, ptrIo needs to be reset to 1st column of ioBuff
		ptrIo = ioBuffRow; // no need to add here (col % ROW_CHUNK_SZ), since col is now a multiple of ROW_CHUNK_SZ
	}

	for(col = constantsDev.complMinWm1R;;) { // the col from here refers to output columns
		const unsigned limCol = min(constantsDev.wm1, (chunkStartCol + ROW_CHUNK_SZm1));

		if(enabled) {
			for(; col <= limCol; ++col) {
				*ptrIo++ = sumT/* * mul_sum*/; // multiply only once with its square, during the processing of columns
				updateStackAndSums(pix, stack, stackIdx, sumIn, sumOut, sumT);
			}
		} else {
			col = limCol + 1U;
		}

		reportAvailableResults(resBuff, chunkStartCol, rowRangeStart, buffDestCol, ioBuff, fpSz);

		if(limCol == constantsDev.wm1)
			break;

		chunkStartCol += ROW_CHUNK_SZ;
		ptrIo = ioBuffRow; // no need to add here (col % ROW_CHUNK_SZ), since col is now a multiple of ROW_CHUNK_SZ
	}
}

/// Applies the Stack blur to the rows of the imgBuff and writes the result to resBuff
__global__
void stackBlurRows(const fp * const __restrict__ imgBuff, fp * const __restrict__ blurredRows,
					const unsigned fpSz) {
	__shared__ fp ioBuff[THREADS][ROW_CHUNK_SZ];
	extern __shared__ fp stackBuff[];
	fp * const stack = stackBuff + constantsDev.stackSz * threadIdx.x;

	fp sumIn = 0.f, sumOut = 0.f, sumT = 0.f;	
	unsigned chunkStartCol = 0U;

	const unsigned rowRangeStart = THREADS * blockIdx.x + ROW_CHUNK_SZ * (threadIdx.x / ROW_CHUNK_SZ),
					buffDestCol = threadIdx.x % ROW_CHUNK_SZ,
					row = THREADS * blockIdx.x + threadIdx.x;
	const bool enabled = (row <= constantsDev.hm1);

	prepareStackAndSums(&ioBuff[0][0], stack, sumIn, sumOut, sumT, chunkStartCol,
						imgBuff, rowRangeStart, buffDestCol,
						enabled, fpSz);

	tackleRows(imgBuff, blurredRows,
			   &ioBuff[0][0], stack, sumIn, sumOut, sumT,
			   row, rowRangeStart, buffDestCol, chunkStartCol,
			   enabled, fpSz);
}

/// Applies the Stack blur to the columns of the ioBuff
__global__
void stackBlurCols(fp * const ioBuff, fp mul_sum_sq) {
	const unsigned col = (__umul24(blockIdx.x, blockDim.x) + threadIdx.x);
	if(col <= constantsDev.wm1) {
		extern __shared__ fp stackBuff[];

		const fp *ptrIn = ioBuff + col;
		fp * const stack = stackBuff + constantsDev.stackSz * threadIdx.x;
		fp sumIn = 0.f, sumOut = *stack = *ptrIn, sumT = sumOut, pix = sumT, *ptrOut = const_cast<fp*>(ptrIn);
		
		unsigned row = 1U;
		for(; row <= constantsDev.minHm1R; ++row) {
			stack[row] = pix;
			sumT += pix * (row + 1U);  sumOut += pix;
			ptrIn = (fp*)((unsigned char*)ptrIn + constantsDev.stride);
			stack[row + constantsDev.r] = pix = *ptrIn;
			sumT += pix * (constantsDev.rp1 - row);  sumIn += pix;
		}
		if(row <= constantsDev.r) { // for a radius larger than image height
			const fp total = pix * (constantsDev.rp1 - row);
			sumOut += total;  sumIn += total;  sumT += constantsDev.rp2 * total;

			for(; row < constantsDev.rp1; ++row) {
				stack[row] = stack[row + constantsDev.r] = pix;
			}
		}

		unsigned stackIdx = constantsDev.r;
		for(row = 0U; row < constantsDev.complMinHm1R; ++row) {
			*ptrOut = sumT/* * mul_sum*/ * mul_sum_sq; // multiply only once with the square of mul_sum
			ptrOut = (fp*)((unsigned char*)ptrOut + constantsDev.stride);
			ptrIn = (fp*)((unsigned char*)ptrIn + constantsDev.stride);
			updateStackAndSums(*ptrIn, stack, stackIdx, sumIn, sumOut, sumT);
		}
		for(; row <= constantsDev.hm1; ++row) {
			*ptrOut = sumT/* * mul_sum*/ * mul_sum_sq; // multiply only once with the square of mul_sum
			ptrOut = (fp*)((unsigned char*)ptrOut + constantsDev.stride);
			updateStackAndSums(*ptrIn, stack, stackIdx, sumIn, sumOut, sumT);
		}
	}
}

/// Launches the 2 kernels (horizontal and vertical) and manipulates the input and the output
void stackBlur(const fp *imgBuff, fp *result, fp *toBlurDev, fp *blurredDev,
			   unsigned rows, unsigned stride, unsigned radius) {
	const size_t fpSz = sizeof(fp),
				buffSz = size_t(rows * stride);
	const unsigned cols = stride / unsigned(fpSz),
					stackSz = 2U * radius + 1U;
	const fp mul_sum = AbsStackBlurImpl::multipliers[radius] / fp(1 << AbsStackBlurImpl::shiftFactors[radius]),
			mul_sum_sq = mul_sum * mul_sum; // multiply only once, during the processing of columns

	static unsigned prevRows = 0U, prevStride = 0U, prevRadius = 0U;
	static bool initialized = false;
	bool newConstants = false;
	if(prevRows != rows) { newConstants = true; prevRows = rows; }
	if(prevStride != stride) { newConstants = true; prevStride = stride; }
	if(prevRadius != radius) { newConstants = true; prevRadius = radius; }

	if(newConstants || !initialized) {
		StackBlurConstants constantsHost; initStackBlurConstants(constantsHost, rows, cols, stride, radius);
		CHECK_OP(cudaMemcpyToSymbol(constantsDev, (void*)&constantsHost, sizeof(StackBlurConstants)));
		initialized = true;
	}

	// There's a CUDA stream for each CPU
	const auto cpuId = omp_get_thread_num();
	const auto streamId = StreamsManager::streams()[cpuId];

// 	CHECK_OP(cudaHostRegister((void*)imgBuff, buffSz, cudaHostRegisterDefault));
	CHECK_OP(cudaMemcpyAsync((void*)toBlurDev, (void*)imgBuff, buffSz, cudaMemcpyHostToDevice, streamId));
// 	CHECK_OP(cudaHostUnregister((void*)imgBuff));

	stackBlurRows<<<(rows + THREADS - 1) / THREADS, THREADS, fpSz * THREADS * stackSz, streamId>>>
		(toBlurDev, blurredDev, (unsigned)fpSz);
	CHECK_OP(cudaGetLastError());

	stackBlurCols<<<(cols + THREADS - 1) / THREADS, THREADS, fpSz * THREADS * stackSz, streamId>>>
		(blurredDev, mul_sum_sq);
	CHECK_OP(cudaGetLastError());

// 	CHECK_OP(cudaHostRegister((void*)result, buffSz, cudaHostRegisterDefault));
	CHECK_OP(cudaMemcpyAsync((void*)result, (void*)blurredDev, buffSz, cudaMemcpyDeviceToHost, streamId));
// 	CHECK_OP(cudaHostUnregister((void*)result));

	CHECK_OP(cudaStreamSynchronize(streamId));
}
