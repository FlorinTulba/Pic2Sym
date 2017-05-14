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

#ifndef H_BOX_BLUR_BASE
#define H_BOX_BLUR_BASE

#include "blur.h"
#include "warnings.h"

#pragma warning ( push, 0 )

#include <opencv2/core/core.hpp>

#pragma warning ( pop )


/// Base class for the Box blur implementations
class AbsBoxBlurImpl /*abstract*/ {
protected:
	unsigned wl = 1U;			///< first odd width of the box mask less than the ideal width
	unsigned wu = 3U;			///< first odd width of the box mask greater than the ideal width
	unsigned countWl = 0U;		///< the number of times to iterate the filter of width wl
	unsigned countWu = 0U;		///< the number of times to iterate the filter of width wu
	unsigned iterations = 0U;	///< desired number of iterations (countWl + countWu)

	virtual ~AbsBoxBlurImpl() {}

public:
	/// Reconfigure the filter through a new desired standard deviation and a new iterations count
	/// See http://www.web.uwa.edu.au/__data/assets/file/0008/826172/filterdesign.pdf for details
	virtual AbsBoxBlurImpl& setSigma(double desiredSigma, unsigned iterations_ = 1U);

	/// Reconfigure mask width (wl) for performing all iterations and destroys the wu mask
	virtual AbsBoxBlurImpl& setWidth(unsigned boxWidth_);

	/// Reconfigure iterations count for wl and destroys the wu mask
	virtual AbsBoxBlurImpl& setIterations(unsigned iterations_);

	/// Implementation of the blur
	virtual void apply(const cv::Mat &toBlur, cv::Mat &blurred) = 0;
};


/**
Adapter of the Box blur implementation.
Configures the implementation for tiny and normal symbols.

The CRTP from below ensures the use of the static methods tinySyms() and nonTinySyms()
from the derived classes (BoxBlurVersion).
*/
template<class BoxBlurVersion>
class TBoxBlur /*abstract*/ : public BlurEngine {

protected:
	static ConfInstRegistrator cir;	///< Registers the configured instance plus its name

	void doProcess(const cv::Mat &toBlur, cv::Mat &blurred, bool forTinySym) const override {
		if(forTinySym)
			BoxBlurVersion::tinySyms().apply(toBlur, blurred);
		else
			BoxBlurVersion::nonTinySyms().apply(toBlur, blurred);
	}

	/*virtual*/ ~TBoxBlur() {} // protected destructor, so no direct instance allowed

public:
	/// Configure the filter through the desired radius
	TBoxBlur(unsigned boxWidth_ = 1U, unsigned iterations_ = 1U) : BlurEngine() {
		setWidth(boxWidth_).setIterations(iterations_);
	}

	/// Returns a fully configured instance
	static const BoxBlurVersion& configuredInstance() {
#pragma warning ( disable : WARN_THREAD_UNSAFE )
		static BoxBlurVersion result;
		static bool initialized = false;
#pragma warning ( default : WARN_THREAD_UNSAFE )

		if(!initialized) {
			extern const double StructuralSimilarity_SIGMA;
			// Box blur with single iteration and desired standard deviation
			result.setSigma(StructuralSimilarity_SIGMA);
			initialized = true;
		}
		return result;
	}

	/// Reconfigure the filter through a new desired standard deviation and a new iterations count
	/// See http://www.web.uwa.edu.au/__data/assets/file/0008/826172/filterdesign.pdf for details
	TBoxBlur& setSigma(double desiredSigma, unsigned iterations_ = 1U) {
		BoxBlurVersion::nonTinySyms().setSigma(desiredSigma, iterations_);

		// Tiny symbols should use a sigma = desiredSigma/2.
		BoxBlurVersion::tinySyms().setSigma(desiredSigma * .5, iterations_);

		return *this;
	}

	/// Reconfigure mask width (wl) for performing all iterations and destroys the wu mask
	TBoxBlur& setWidth(unsigned boxWidth_) {
		BoxBlurVersion::nonTinySyms().setWidth(boxWidth_);

		// Tiny symbols should use a box whose width is next odd value >= boxWidth_/2.
		BoxBlurVersion::tinySyms().setWidth((boxWidth_>>1) | 1U);

		return *this;
	}

	/// Reconfigure iterations count for wl and destroys the wu mask
	TBoxBlur& setIterations(unsigned iterations_) {
		BoxBlurVersion::nonTinySyms().setIterations(iterations_);
		BoxBlurVersion::tinySyms().setIterations(iterations_);

		return *this;
	}
};

#endif // H_BOX_BLUR_BASE