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

#ifndef H_STACK_BLUR
#define H_STACK_BLUR

#include "stackBlurBase.h"

/**
Stack blurring algorithm

Note this is a different algorithm than Stacked Integral Image (SII).

Brought several modifications to:
	Stack Blur Algorithm by Mario Klingemann <mario@quasimondo.com>:
	http://www.codeproject.com/Articles/42192/Fast-Image-Blurring-with-CUDA
	under license: http://www.codeproject.com/info/cpol10.aspx

It was included in the project since it also presents a working version for CUDA.
*/
class StackBlur : public TStackBlur<StackBlur> {
	friend class TStackBlur<StackBlur>; // for accessing nonTinySyms() and tinySyms() from below

protected:
	static AbsStackBlurImpl& nonTinySyms();	///< handler for non-tiny symbols
	static AbsStackBlurImpl& tinySyms();	///< handler for tiny symbols

public:
	/// Configure the filter through the desired radius
	StackBlur(unsigned radius);
};

#endif // H_STACK_BLUR