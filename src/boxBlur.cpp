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

#include "boxBlur.h"

#pragma warning ( push, 0 )

#include <opencv2/imgproc/imgproc.hpp>

#pragma warning ( pop )

using namespace std;
using namespace cv;

// Handle class
class BoxBlurImpl : public AbsBoxBlurImpl {
	static const cv::Point midPoint; ///< Default anchor point for a given kernel (its center)

public:
	/// Actual implementation for the current configuration. toBlur is checked; blurred is initialized
	/// See http://www.web.uwa.edu.au/__data/assets/file/0008/826172/filterdesign.pdf for details
	void apply(const cv::Mat &toBlur, cv::Mat &blurred) override {
		if(iterations == 0U) {
			toBlur.copyTo(blurred);
			return;
		}

		bool applied = false;

		// The smaller mask (wl) can be a single-point mask (wl == 1), which can be skipped,
		// as it doesn't affect the result at all. In that case, countWl is 0
		if(countWl > 0U) {
			const Size boxL((int)wl, (int)wl);
			blur(toBlur, blurred, boxL, midPoint, BORDER_REPLICATE); // 1st time with mask wl

			// rest of the times with mask wu
			for(unsigned i = 1U; i < countWl; ++i)
				blur(blurred, blurred, boxL, midPoint, BORDER_REPLICATE);
			applied = true;
		}

		if(countWu > 0U) {
			const Size boxU((int)wu, (int)wu);

			// 1st time with mask wu
			if(!applied)
				blur(toBlur, blurred, boxU, midPoint, BORDER_REPLICATE);
			else
				blur(blurred, blurred, boxU, midPoint, BORDER_REPLICATE);

			// rest of the times with mask wu
			for(unsigned i = 1U; i < countWu; ++i)
				blur(blurred, blurred, boxU, midPoint, BORDER_REPLICATE);
		}
	}
};

const Point BoxBlurImpl::midPoint(-1, -1);

AbsBoxBlurImpl& BoxBlur::nonTinySyms() {
	static BoxBlurImpl impl;
	return impl;
}

AbsBoxBlurImpl& BoxBlur::tinySyms() {
	static BoxBlurImpl impl;
	return impl;
}

BoxBlur::BoxBlur(unsigned boxWidth_/* = 1U*/, unsigned iterations_/* = 1U*/) :
	TBoxBlur<BoxBlur>(boxWidth_, iterations_) {}
