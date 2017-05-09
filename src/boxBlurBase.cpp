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

#include "boxBlurBase.h"

using namespace std;
using namespace cv;

AbsBoxBlurImpl& AbsBoxBlurImpl::setSigma(double desiredSigma, unsigned iterations_/* = 1U*/) {
	assert(iterations_ > 0U);
	assert(desiredSigma > 0.);

	iterations = iterations_;
	const double common = 12. * desiredSigma * desiredSigma,
		idealBoxWidth = sqrt(1. + common / iterations_);
	wl = (((unsigned((idealBoxWidth-1.)/2.))<<1) | 1U);
	wu = wl + 2U; // next odd value

	countWl = (unsigned)(std::max)(0,
								   int(round((common - iterations_ * (3U + wl * (wl + 4U)))/(-4. * (wl + 1U)))));
	countWu = iterations_ - countWl;

	if(1U == wl) {
		iterations -= countWl;
		countWl = 0U;
	}

	return *this;
}

AbsBoxBlurImpl& AbsBoxBlurImpl::setWidth(unsigned boxWidth_) {
	assert(1U == (boxWidth_ & 1U)); // Parameter should be an odd value

	if(boxWidth_ == 1U)
		iterations = 0U;
	else
		wl = boxWidth_;

	countWl = iterations;
	countWu = 0U; // cancels any iterations for filters with width wu

	return *this;
}

AbsBoxBlurImpl& AbsBoxBlurImpl::setIterations(unsigned iterations_) {
	if(wl == 1U)
		iterations_ = 0U;

	iterations = countWl = iterations_;
	countWu = 0U; // cancels any iterations for filters with width wu

	return *this;
}
