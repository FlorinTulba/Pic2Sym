/****************************************************************************************
 The application Pic2Sym approximates images by a
 grid of colored symbols with colored backgrounds.

 This file belongs to the Pic2Sym project.

 Copyrights from the libraries used by the program:
 - (c) 2015 Boost (www.boost.org)
   License: <http://www.boost.org/LICENSE_1_0.txt>
            or doc/licenses/Boost.lic
 - (c) 2015 The FreeType Project (www.freetype.org)
   License: <http://git.savannah.gnu.org/cgit/freetype/freetype2.git/plain/docs/FTL.TXT>
	        or doc/licenses/FTL.txt
 - (c) 2015 OpenCV (www.opencv.org)
   License: <http://opencv.org/license.html>
            or doc/licenses/OpenCV.lic
 - (c) 1997-2002 OpenMP Architecture Review Board (www.openmp.org)
   (c) Microsoft Corporation (Visual C++ implementation for OpenMP C/C++ Version 2.0 March 2002)
   See: <https://msdn.microsoft.com/en-us/library/8y6825x5(v=vs.140).aspx>
 
 (c) 2016 Florin Tulba <florintulba@yahoo.com>

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
 ****************************************************************************************/

#include "boxBlur.h"
#include "misc.h"

#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

// Handle class
class BoxBlur::Impl {
	friend class BoxBlur;

	unsigned wl = 0U;	///< first odd width of the box mask less than the ideal width
	unsigned wu = 0U;	///< first odd width of the box mask greater than the ideal width
	unsigned countWl = 0U;	///< the number of times to repeat the filter of width wl
	unsigned countWu = 0U;	///< the number of times to repeat the filter of width wu

	Impl() {}

	/// Reconfigure the filter through a new desired standard deviation and a new repetitions count
	/// See http://www.web.uwa.edu.au/__data/assets/file/0008/826172/filterdesign.pdf for details
	Impl& setSigma(double desiredSigma, unsigned repetitions_ = 1U) {
		if(0U == repetitions_)
			THROW_WITH_CONST_MSG("Repetitions should be > 0 in " __FUNCTION__, invalid_argument);

		if(desiredSigma < 0.)
			THROW_WITH_CONST_MSG("desiredSigma should be > 0 in " __FUNCTION__, invalid_argument);

		const double common = 12. * desiredSigma * desiredSigma,
					idealBoxWidth = sqrt(1. + common / repetitions_);
		wl = (((unsigned((idealBoxWidth-1.)/2.))<<1) | 1U);
		wu = wl + 2U; // next odd value

		countWl = int(round((common - repetitions_ * (3U + wl * (wl + 4U)))/(-4. * (wl + 1U))));
		countWu = repetitions_ - countWl;

		return *this;
	}

	/// Reconfigure mask width (wl) and destroys the wu mask
	Impl& setWidth(unsigned boxWidth_) {
		if(0U == (boxWidth_ & 1U))
			THROW_WITH_CONST_MSG("Parameter should be an odd value in " __FUNCTION__, invalid_argument);

		wl = boxWidth_;
		countWu = 0U; // cancels any repetitions for filters with width wu

		return *this;
	}

	/// Reconfigure repetitions count for wl and destroys the wu mask
	Impl& setRepetitions(unsigned repetitions_) {
		if(0U == repetitions_)
			THROW_WITH_CONST_MSG("Repetitions should be > 0 in " __FUNCTION__, invalid_argument);

		countWl = repetitions_;
		countWu = 0U; // cancels any repetitions for filters with width wu
		
		return *this;
	}

	/// Actual implementation for the current configuration. toBlur is checked; blurred is initialized
	/// See http://www.web.uwa.edu.au/__data/assets/file/0008/826172/filterdesign.pdf for details
	void apply(const cv::Mat &toBlur, cv::Mat &blurred) {
		if(countWl == 0U && countWu == 0U) {
			toBlur.copyTo(blurred);
			return;
		}

		static const Point midPoint(-1, -1);
		
		bool applied = false;

		// The smaller mask (wl) can be a single-point mask (wl == 1), which can be skipped,
		// as it doesn't affect the result at all
		if(wl > 1U && countWl > 0U) {
			const Size boxL(wl, wl);
			blur(toBlur, blurred, boxL, midPoint, BORDER_REPLICATE); // 1st time with mask wl

			// rest of the times with mask wu
			for(unsigned i = 1U; i < countWl; ++i)
				blur(blurred, blurred, boxL, midPoint, BORDER_REPLICATE);
			applied = true;
		}

		if(countWu > 0U) {
			const Size boxU(wu, wu);

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

BoxBlur::Impl& BoxBlur::impl() {
	static BoxBlur::Impl implem;
	return implem;
}

BoxBlur::BoxBlur(unsigned boxWidth_/* = 1U*/, unsigned repetitions_/* = 1U*/) {
	impl().setWidth(boxWidth_).setRepetitions(repetitions_);
}

BoxBlur& BoxBlur::setSigma(double desiredSigma, unsigned repetitions_/* = 1U*/) {
	impl().setSigma(desiredSigma, repetitions_);
	return *this;
}

BoxBlur& BoxBlur::setWidth(unsigned boxWidth_) {
	impl().setWidth(boxWidth_);
	return *this;
}

BoxBlur& BoxBlur::setRepetitions(unsigned repetitions_) {
	impl().setRepetitions(repetitions_);
	return *this;
}

void BoxBlur::doProcess(const cv::Mat &toBlur, cv::Mat &blurred) const {
	impl().apply(toBlur, blurred);
}

const BoxBlur& BoxBlur::configuredInstance() {
	extern const double StructuralSimilarity_SIGMA;
	static BoxBlur result;
	static bool initialized = false;
	if(!initialized) {
		// Box blur with no repetitions and desired standard deviation
		result.setSigma(StructuralSimilarity_SIGMA);
		initialized = true;
	}
	return result;
}
