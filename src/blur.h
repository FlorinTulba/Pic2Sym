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

#ifndef H_BLUR
#define H_BLUR

#include "blurBase.h"

#pragma warning ( push, 0 )

#include <string>
#include <map>

#pragma warning ( pop )

/**
Base class for various versions of blurring that can be configured within the application.

One possible matching aspect to be used during image approximation is
Structural Similarity (https://ece.uwaterloo.ca/~z70wang/research/ssim ).
It relies heavily on Gaussian blurring, whose implementation is already optimized in OpenCV for a sequential run.

This class addresses the issue that GaussianBlur function is the most time-consuming operation
during image approximation.

For the typical standard deviation of 1.5, GaussianBlur from OpenCV still remains the fastest
when compared to other tested sequential innovative algorithms:
- Young & van Vliet (implementation from CImg library - http://cimg.eu/)
- Deriche (implementation from CImg library - http://cimg.eu/)
- Stacked Integral Image (implementation from http://dev.ipol.im/~getreuer/code/doc/gaussian_20131215_doc/group__sii__gaussian.html)
- Stack blur (adaptation of the sequential algorithm from http://www.codeproject.com/Articles/42192/Fast-Image-Blurring-with-CUDA)

All those competitor algorithms are less accurate than Extended Box Blur configured with just 2 repetitions.
When applied only once, sequential Box-based blur techniques can be up to 3 times faster than GaussianBlur from OpenCV.
However, basic Box blurring with no repetitions has poor quality,
while Extended Box blurring incurs additional time costs for an improved quality.

An implementation of the Extended Box blurring exists at: http://dev.ipol.im/~getreuer/code/doc/gaussian_20131215_doc/group__ebox__gaussian.html
This project contains its own implementation of this blur technique (ExtBoxBlur).

The project includes following blur algorithms:
- GaussBlur - the reference blur, delegating to sequential GaussianBlur from OpenCV
- BoxBlur - for its versatility: quickest for no repetitions and slower, but increasingly accurate for more repetitions
  (Every repetition delegates to blur from OpenCV)
- ExtBoxBlur - for its accuracy, even for only a few repetitions. The sequential algorithm is highly parallelizable
- StackBlur - for its provided CUDA version that shows terrific time improvement compared to the sequential algorithm

All derived classes are expected to provide a static method that provides an instance of them
already configured for blurring serving structural similarity matching aspect:

	static const Derived& configuredInstance();

Besides, the derived classes will also declare a static field:
	
	static ConfInstRegistrator cir;

that will be initialized in varConfig.cpp unit like this:

	BlurEngine::ConfInstRegistrator Derived::cir("<blurTypeName_from_varConfig.txt>", Derived::configuredInstance());
*/
class BlurEngine : public IBlurEngine {
protected:
	/// Mapping type between blurTypes and corresponding configured blur instances
	typedef std::map<const std::string, const IBlurEngine*> ConfiguredInstances;

	/**
	Derived classes register themselves like: configuredInstances().insert(blurType, configuredInst)
	Instead of such a call, they simply declare a static field of type ConfInstRegistrator (see below)
	who performs the mentioned operation within its constructor.
	*/
	static ConfiguredInstances& configuredInstances();

	/**
	Derived classes from BlurEngine need to declare a static ConfInstRegistrator cir to self-register
	within BlurEngine::configuredInstances()
	*/
	struct ConfInstRegistrator {
		/// Provides the blur name and the instance to be registered within BlurEngine::configuredInstances()
		ConfInstRegistrator(const std::string &blurType, const IBlurEngine &configuredInstance);
	};

	/**
	Actual implementation of the blur algorithm

	@param toBlur is a single channel matrix with values of type double
	@param blurred the result of the blur (already initialized when method gets called)
	@param forTinySym demands generating a Gaussian blur with smaller window and standard deviation
			for tiny symbols
	*/
	virtual void doProcess(const cv::Mat &toBlur, cv::Mat &blurred, bool forTinySym) const = 0;

public:
	/// Provides a specific, completely configured blur engine. Throws invalid_argument for an unrecognized blurType
	static const IBlurEngine& byName(const std::string &blurType);

	/**
	Template method checking toBlur, initializing blurred and calling doProcess

	@param toBlur is a single channel matrix with values of type double
	@param blurred the result of the blur (not initialized when method gets called)
	@param forTinySym demands generating a Gaussian blur with smaller window and standard deviation
			for tiny symbols
	*/
	void process(const cv::Mat &toBlur, cv::Mat &blurred, bool forTinySym) const override;
};

#endif // H_BLUR
