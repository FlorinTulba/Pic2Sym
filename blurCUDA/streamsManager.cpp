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

#include "streamsManager.h"
#include "util.h"
#include "misc.h"

#pragma warning ( push, 0 )

#include <cuda_runtime.h>

#include <omp.h>

#pragma warning ( pop )

using namespace std;

const StreamsManager& StreamsManager::streams() {
#pragma warning ( disable : WARN_THREAD_UNSAFE )
	static StreamsManager impl;
#pragma warning ( default : WARN_THREAD_UNSAFE )
	return impl;
}

StreamsManager::StreamsManager() :
		count_(cudaInitOk() ? omp_get_num_procs() : 0ULL) {
	if(count_ > MaxCPUsCount)
		THROW_WITH_VAR_MSG("Your system appears to have " + to_string(count_) +
			" CPUs, which more than currently configured - " + to_string(MaxCPUsCount) +
			". You may edit MaxCPUsCount from '/blurCUDA/streamManager.h' to overcome this issue.",
			out_of_range);

	for(size_t i = 0ULL; i < count_; ++i)
		CHECK_OP(cudaStreamCreate(&streams_[i]));
}

StreamsManager::~StreamsManager() {
	for(size_t i = 0ULL; i < count_; ++i)
		CHECK_OP_NO_THROW(cudaStreamDestroy(streams_[i]));
}

cudaStream_t StreamsManager::operator[](size_t idx) const {
	if(idx < count_)
		return streams_[idx];

	THROW_WITH_VAR_MSG(__FUNCTION__ " called with index " + to_string(idx) +
					   ", while there are only " + to_string(count_) + " CUDA streams!", out_of_range);
}

