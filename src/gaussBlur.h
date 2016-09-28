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

#ifndef H_GAUSS_BLUR
#define H_GAUSS_BLUR

#include "blur.h"

/*
Just forwarding to GaussianBlur from OpenCV.
Using only replicated borders.

This is the reference quality blur, but it's also the most time-expensive filter.
*/
class GaussBlur : public BlurEngine {
	static const GaussBlur& configuredInstance();	///< Returns a fully configured instance
	static ConfInstRegistrator cir;					///< Registers the configured instance plus its name

protected:
	double sigma;			///< desired standard deviation
	unsigned kernelWidth;	///< mask width (must be an odd value, or 0 - to let it be set from sigma)

	/// Forwarding to GaussianBlur. toBlur is checked; blurred is initialized
	void doProcess(const cv::Mat &toBlur, cv::Mat &blurred) const override;

public:
	/// Configure the filter through the desired standard deviation and the mask width
	GaussBlur(double desiredSigma, unsigned kernelWidth_ = 0U);

	/// Reconfigure the filter through a new desired standard deviation and a new mask width
	GaussBlur& configure(double desiredSigma, unsigned kernelWidth_ = 0U);
};

#endif // H_GAUSS_BLUR