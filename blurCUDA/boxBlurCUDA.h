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

#ifndef H_BOX_BLUR_CUDA
#define H_BOX_BLUR_CUDA

#include "boxBlurBase.h"

/// Box blurring algorithm performing the operation using only the GPU
class BoxBlurCUDA : public TBoxBlur<BoxBlurCUDA> {
	friend class TBoxBlur<BoxBlurCUDA>; // for accessing nonTinySyms() and tinySyms() from below

protected:
	static AbsBoxBlurImpl& nonTinySyms();	///< handler for non-tiny symbols
	static AbsBoxBlurImpl& tinySyms();		///< handler for tiny symbols

public:
	/*
	Following 2 static methods provide each a simple configurable constant.
	Changing these constants directly in the interface costs lots of compile time.
	Using methods instead of basic constants:
	 - ensures the constants are initialized in all sources using them, no matter their compile order
	 - allows recompiling only a single file
	*/
	static unsigned BlockDimRows(); ///< CUDA thread block size when handling row blurring
	static unsigned BlockDimCols(); ///< CUDA thread block size when handling column blurring

	/**
	Preconditions for using the CUDA implementation:
	- there is a device with cc >= 1.1
	- the shared memory required by the algorithm is within the device bounds
	*/
	static bool preconditionsOk();

	/// Configure the filter through the mask width and the iterations count
	BoxBlurCUDA(unsigned boxWidth_ = 1U, unsigned iterations_ = 1U);
};


/// Box blurring algorithm performing only the initialization of the rolling sums on the GPU
class BoxBlurCUDAmin : public TBoxBlur<BoxBlurCUDAmin> {
	friend class TBoxBlur<BoxBlurCUDAmin>; // for accessing nonTinySyms() and tinySyms() from below

protected:
	static AbsBoxBlurImpl& nonTinySyms();	///< handler for non-tiny symbols
	static AbsBoxBlurImpl& tinySyms();		///< handler for tiny symbols

public:
	/// Configure the filter through the mask width and the iterations count
	BoxBlurCUDAmin(unsigned boxWidth_ = 1U, unsigned iterations_ = 1U);
};

#endif // H_BOX_BLUR_CUDA