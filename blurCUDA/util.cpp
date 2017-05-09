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

#include "util.h"
#include "warnings.h"

#pragma warning ( push, 0 )

#include <iostream>
#include <sstream>

#include <cuda_runtime_api.h>

#pragma warning ( pop )

using namespace std;

void checkOp(cudaError_t opResult, const string &file, const int lineNo, bool doThrow/* = true*/) {
	if(cudaSuccess != opResult) {
		ostringstream oss;
		oss<<'['<<file<<':'<<lineNo<<"] CUDA error: "<<cudaGetErrorString(opResult);
		cerr<<oss.str()<<endl;
		if(doThrow)
			throw runtime_error(oss.str());
	}
}

bool cudaInitOk(cudaDeviceProp *pDeviceProps/* = nullptr*/) {
#pragma warning ( disable : WARN_THREAD_UNSAFE )
	static cudaDeviceProp deviceProps;
	static bool initialized = false, result = false;
#pragma warning ( default : WARN_THREAD_UNSAFE )
	if(!initialized) {
		int deviceCount = 0;
		cudaError_t err = cudaGetDeviceCount(&deviceCount);
		if(err != cudaSuccess) {
			cerr<<"Couldn't count CUDA-compatible devices! cudaGetDeviceCount returned "<<cudaGetErrorString(err)<<endl;
		} else if(deviceCount == 0) {
			cerr<<"There are no CUDA-compatible devices!"<<endl;
		} else {
			for(int dev = 0; dev < deviceCount; ++dev) {
				cudaGetDeviceProperties(&deviceProps, dev);
				if((deviceProps.major > CUDA_REQUIRED_CC_MAJOR) ||
						((deviceProps.major == CUDA_REQUIRED_CC_MAJOR) &&
						(deviceProps.minor >= CUDA_REQUIRED_CC_MINOR))) {
					cudaSetDevice(dev);
					result = true;
					break;
				}
			}

			if(!result)
				cerr<<"Existing CUDA devices have cc less than minimum "
					<<CUDA_REQUIRED_CC_MAJOR<<'.'<<CUDA_REQUIRED_CC_MINOR<<" required!"<<endl;
		}

		initialized = true;
	}

	if(pDeviceProps != nullptr && result) {
		static const size_t szCudaDeviceProp = sizeof(cudaDeviceProp);
		memcpy_s(pDeviceProps, szCudaDeviceProp, &deviceProps, szCudaDeviceProp);
	}

	return result;
}

