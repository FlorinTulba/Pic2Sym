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

#ifndef H_STREAMS_MANAGER
#define H_STREAMS_MANAGER

#pragma warning ( push, 0 )

#include <driver_types.h>

#pragma warning ( pop )

/// Manages the CUDA streams
class StreamsManager {
protected:
	enum { MaxCPUsCount = 32}; ///< Limit for the streams count (it should be less than the CPU-s count)
	
	size_t count_;	///< same as CPU count

	/**
	One stream for each available CPU, which are hopefully less than MaxCPUsCount.

	Allocating this array dynamically based on the actual CPU count requires also
	releasing the memory in the destructor.
	However, in that case the destructor either crashes, or has to let the mentioned memory leak:
	http://stackoverflow.com/questions/16979982/cuda-streams-destruction-and-cudadevicereset
	*/
	cudaStream_t streams_[MaxCPUsCount];

	StreamsManager(const StreamsManager&) = delete;
	StreamsManager(StreamsManager&&) = delete;

	StreamsManager();

public:
	static const StreamsManager& streams();	/// Singleton to construct, provide & destroy the streams

	~StreamsManager();

	inline size_t count() const { return count_; }
	cudaStream_t operator[](size_t idx) const;
};

#endif // H_STREAMS_MANAGER