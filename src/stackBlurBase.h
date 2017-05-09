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

#ifndef H_STACK_BLUR_BASE
#define H_STACK_BLUR_BASE

#include "blur.h"
#include "warnings.h"

/// Base class for the Stack blur implementations
class AbsStackBlurImpl /*abstract*/ {
protected:
	unsigned r = 1U;	///< filter radius (valid range: 1..254)

public:
	static const int multipliers[];		///< utilized multipliers array
	static const int shiftFactors[];	///< utilized shift factors array

	virtual ~AbsStackBlurImpl() {}

	/// Reconfigure the filter through a new desired standard deviation
	virtual AbsStackBlurImpl& setSigma(double desiredSigma);

	/// Reconfigure the filter through a new radius
	virtual AbsStackBlurImpl& setRadius(unsigned radius);

	/// Implementation of the blur
	virtual void apply(const cv::Mat &toBlur, cv::Mat &blurred) const = 0;
};


/**
Adapter of the Stack blur implementation.
Configures the implementation for tiny and normal symbols.

The CRTP from below ensures the use of the static methods tinySyms() and nonTinySyms()
from the derived classes (StackBlurVersion).
*/
template<class StackBlurVersion>
class TStackBlur /*abstract*/ : public BlurEngine {

protected:
	static ConfInstRegistrator cir;	///< Registers the configured instance plus its name

	void doProcess(const cv::Mat &toBlur, cv::Mat &blurred, bool forTinySym) const override {
		if(forTinySym)
			StackBlurVersion::tinySyms().apply(toBlur, blurred);
		else
			StackBlurVersion::nonTinySyms().apply(toBlur, blurred);
	}

	/*virtual*/ ~TStackBlur() {} // protected destructor, so no direct instance allowed

public:
	/// Configure the filter through the desired radius
	TStackBlur(unsigned radius) : BlurEngine() {
		setRadius(radius);
	}

	/// Returns a fully configured instance
	static const StackBlurVersion& configuredInstance() {
#pragma warning ( disable : WARN_THREAD_UNSAFE )
		static StackBlurVersion result(1U);
		static bool initialized = false;
#pragma warning ( default : WARN_THREAD_UNSAFE )

		if(!initialized) {
			// Stack blur with desired standard deviation
			extern const double StructuralSimilarity_SIGMA;
			result.setSigma(StructuralSimilarity_SIGMA);
			initialized = true;
		}
		return result;
	}

	/// Reconfigure the filter through a new desired standard deviation
	TStackBlur& setSigma(double desiredSigma) {
		StackBlurVersion::nonTinySyms().setSigma(desiredSigma);

		// Tiny symbols should use a sigma = desiredSigma/2.
		StackBlurVersion::tinySyms().setSigma(desiredSigma * .5);

		return *this;
	}

	/// Reconfigure radius
	TStackBlur& setRadius(unsigned radius) {
		StackBlurVersion::nonTinySyms().setRadius(radius);

		// Tiny symbols should use half the radius from normal symbols
		StackBlurVersion::tinySyms().setRadius(max((radius>>1), 1U));

		return *this;
	}
};

#endif // H_STACK_BLUR_BASE