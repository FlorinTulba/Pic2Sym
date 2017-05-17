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
#include "boxBlurCUDA.h"
#include "streamsManager.h"
#include "util.h"

#include <cuda_runtime.h>

using namespace std;

/// Computes the initial rolling sums for each row / column
__global__
void initRollSums(fp *dataDev, unsigned rowsOrCols, unsigned providedColsOrRows, unsigned maskRadius) {
	const unsigned rowOrCol = blockIdx.x * blockDim.x + threadIdx.x,
		idxStartRowOrCol = rowOrCol,
		idxEndRowOrCol = (providedColsOrRows - 1U) * rowsOrCols + rowOrCol;
	const fp dataEndRowOrCol = dataDev[idxEndRowOrCol];
	if(rowOrCol < rowsOrCols) {
		fp rollingSum = dataDev[idxStartRowOrCol] * fp(maskRadius + 1U); // init the sum with (radius+1)*1st value
		for(unsigned colOrRow = 1U, idx = idxStartRowOrCol + rowsOrCols; colOrRow <= maskRadius;
				++colOrRow, idx += rowsOrCols)
			rollingSum += (idx < idxEndRowOrCol ? dataDev[idx] : dataEndRowOrCol); // add following radius values

		dataDev[idxStartRowOrCol] = rollingSum;
	}
}

void computeInitialRollingSums(fp *firstColOrRow, size_t firstColOrRowDataSz,
							   fp *nextRelevantColsOrRows, size_t nextRelevantColsOrRowsDataSz,
							   fp *dataDev, unsigned rowsOrCols, unsigned providedColsOrRows,
							   unsigned maskRadius, cudaStream_t streamId) {
	CHECK_OP(cudaMemcpyAsync((void*)&dataDev[rowsOrCols],
		(void*)nextRelevantColsOrRows,
		nextRelevantColsOrRowsDataSz, cudaMemcpyHostToDevice, streamId));

	// Launch kernel computing the rolling sum in dataDev[0 .. rowsOrCols]
	enum { ThreadsPerBlock = 64 };
	initRollSums<<<1 + (int(rowsOrCols)-1)/ThreadsPerBlock, ThreadsPerBlock, 0ULL, streamId>>>
		(dataDev, rowsOrCols, providedColsOrRows, maskRadius);
	CHECK_OP(cudaGetLastError());

	CHECK_OP(cudaMemcpyAsync((void*)firstColOrRow, (void*)dataDev,
		firstColOrRowDataSz, cudaMemcpyDeviceToHost, streamId));
}

/// Performs the box blur to the columns
__global__
void boxBlurCols(const fp* const __restrict__ imgBuf, fp* const __restrict__ blurredCols,
				const unsigned rows, const unsigned cols,
				const unsigned* const __restrict__ maskWidthsDev, const unsigned iterations) {
	extern __shared__ fp ioData[]; // 2*'rows' rows and blockDim.x columns
	const unsigned col = blockDim.x * blockIdx.x + threadIdx.x,
				maxInIdx = rows * blockDim.x;
	fp *inData = ioData, *outData = inData + maxInIdx, *aux = nullptr;

	// Copy data (columns accessed by this block from imgBuf) to shared memory
	if(col < cols) {
		for(unsigned row = 0U, inIdx = threadIdx.x, imgIdx = col; row < rows;
				++row, inIdx += blockDim.x, imgIdx += cols)
			inData[inIdx] = imgBuf[imgIdx];
	}
	__syncthreads();

	if(col < cols) {
		for(unsigned iter = 0U; iter < iterations;
				++iter, aux = inData, inData = outData, outData = aux) {
			const unsigned maskWidth = maskWidthsDev[iter],
						maskRadius = maskWidth >> 1;
			const fp colTop = inData[threadIdx.x],
					colBottom = inData[threadIdx.x + (rows-1U) * blockDim.x];

			/* Perform the box filtering (on each column) with a box of width maskWidth replicating the borders */

			// Setup of the rolling sum
			fp rollingSum = colTop * (1.f + maskRadius);
			for(unsigned row = 1U, inIdx = threadIdx.x + blockDim.x; row <= maskRadius; ++row, inIdx += blockDim.x)
				rollingSum += (inIdx < maxInIdx ? inData[inIdx] : colBottom);

			// Traversal of the column using the rolling sum
			int tailIdx = int(threadIdx.x - (int)maskRadius * blockDim.x);
			unsigned frontIdx = threadIdx.x + (1U + maskRadius) * blockDim.x;
			for(unsigned row = 0U, outIdx = threadIdx.x; row < rows;
					++row, outIdx += blockDim.x, tailIdx += (int)blockDim.x, frontIdx += blockDim.x) {
				outData[outIdx] = rollingSum;
				// There is a final rescaling at the end of 'boxBlurRows' (division by the product of all mask widths)

				rollingSum +=
					(frontIdx < maxInIdx ? inData[frontIdx] : colBottom) -
					(tailIdx > 0 ? inData[tailIdx] : colTop);
			}
		}
	}

	// Copy data (outData was swapped with inData) from shared memory to blurredCols
	__syncthreads();
	if(col < cols) {
		for(unsigned row = 0U, inIdx = threadIdx.x, outIdx = col; row < rows;
				++row, inIdx += blockDim.x, outIdx += cols)
			blurredCols[outIdx] = inData[inIdx];
	}
}

/// Performs the box blur to the rows. Affected by the uncoalesced global memory accesses
__global__
void boxBlurRows(fp* const __restrict__ ioDataGlob, const unsigned rows, const unsigned cols,
				const unsigned* const __restrict__ maskWidthsDev,
				const unsigned iterations, const fp scaler) {
	extern __shared__ fp prevVals[]; // maskRadius rows of blockDim.x columns. Each column is a circular buffer
	const unsigned row = blockDim.x * blockIdx.x + threadIdx.x,
				idxStartRow = row * cols,
				idxEndRow = idxStartRow + cols - 1U;
	if(row < rows) {
		// Data changes after each iteration, so the texture mechanism isn't practical, as it expect constant data.
		// The alternative is transposing the data matrix, applying boxBlurColumns on it and then transposing it back
		for(unsigned iter = 0U; iter < iterations; ++iter) {
			const fp startRow = ioDataGlob[idxStartRow],
					endRow = ioDataGlob[idxEndRow];
			const unsigned maskWidth = maskWidthsDev[iter],
						maskRadius = maskWidth >> 1,
						min_maskRadius_cols = min(maskRadius, cols);

			/* Perform the box filtering (on each row) with a box of width maskWidth replicating the borders */

			// Setup of the rolling sum
			fp rollingSum = startRow * (1.f + maskRadius);
			for(unsigned col = 1U, idx = idxStartRow + 1U; col <= maskRadius; ++col, ++idx)
				rollingSum += (idx < idxEndRow ? ioDataGlob[idx] : endRow);

			// Compute columns 0 .. min_maskRadius_cols-1
			unsigned frontIdx = idxStartRow + (1U + maskRadius),
					outIdx = idxStartRow,
					col = 0U;
			for(; col < min_maskRadius_cols; ++col, ++outIdx, ++frontIdx) {
				prevVals[threadIdx.x + col * blockDim.x] = ioDataGlob[outIdx];
				ioDataGlob[outIdx] = rollingSum;
				// There is a final rescaling at the end of this kernel (division by the product of all mask widths)

				rollingSum += (frontIdx < idxEndRow ? ioDataGlob[frontIdx] : endRow) - startRow;
			}

			// Compute columns min_maskRadius_cols .. cols-1
			for(unsigned tailIdxPrevVals = threadIdx.x + blockDim.x * (col % maskRadius); col < cols;
					++outIdx, ++frontIdx,
					tailIdxPrevVals = threadIdx.x + blockDim.x * (++col % maskRadius)) {
				fp temp = ioDataGlob[outIdx];
				ioDataGlob[outIdx] = rollingSum;
				// There is a final rescaling at the end of this kernel (division by the product of all mask widths)

				rollingSum +=
					(frontIdx < idxEndRow ? ioDataGlob[frontIdx] : endRow) - prevVals[tailIdxPrevVals];
				prevVals[tailIdxPrevVals] = temp;
			}
		}
	}

	// Wait for the completion of the iterations performing the blur by rows before scaling the result
	__syncthreads();
	for(unsigned row = blockDim.x * blockIdx.x, limRow = min(rows, row + blockDim.x); row < limRow; ++row)
		for(unsigned col = threadIdx.x, rowStart = row * cols; col < cols; col += blockDim.x)
			ioDataGlob[rowStart + col] *= scaler;
}

/// Launches the 2 kernels (horizontal and vertical) and manipulates the input and the output
void boxBlur(const fp *imgBuff, fp *result, fp *toBlurDev, fp *blurredDev,
			 unsigned rows, unsigned cols, size_t buffSz,
			 unsigned *maskWidthsDev, unsigned iterations,
			 unsigned largestMaskRadius, fp scaler,
			 cudaStream_t streamId) {
	static const size_t fpSz = sizeof(fp);

// 	CHECK_OP(cudaHostRegister((void*)imgBuff, buffSz, cudaHostRegisterDefault));
	CHECK_OP(cudaMemcpyAsync((void*)toBlurDev, (void*)imgBuff, buffSz, cudaMemcpyHostToDevice, streamId));
// 	CHECK_OP(cudaHostUnregister((void*)imgBuff));

	boxBlurCols<<<(cols + BoxBlurCUDA::BlockDimCols() - 1) / BoxBlurCUDA::BlockDimCols(),
				BoxBlurCUDA::BlockDimCols(),
				2U * BoxBlurCUDA::BlockDimCols() * rows * fpSz, // dynamic shared memory for in+out rows x blockDim tables of floats
				streamId>>>
		(toBlurDev, blurredDev, rows, cols, maskWidthsDev, iterations);
	CHECK_OP(cudaGetLastError());

	boxBlurRows<<<(rows + BoxBlurCUDA::BlockDimRows() - 1) / BoxBlurCUDA::BlockDimRows(),
				BoxBlurCUDA::BlockDimRows(),
				BoxBlurCUDA::BlockDimRows() * largestMaskRadius * fpSz, // dynamic shared memory for blockDim x largest_mask_radius tables of floats
				streamId>>>
		(blurredDev, rows, cols, maskWidthsDev, iterations, scaler);
	CHECK_OP(cudaGetLastError());

// 	CHECK_OP(cudaHostRegister((void*)result, buffSz, cudaHostRegisterDefault));
	CHECK_OP(cudaMemcpyAsync((void*)result, (void*)blurredDev, buffSz, cudaMemcpyDeviceToHost, streamId));
// 	CHECK_OP(cudaHostUnregister((void*)result));

	CHECK_OP(cudaStreamSynchronize(streamId));
}
