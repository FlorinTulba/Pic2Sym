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

#include "extBoxBlur.h"
#include "misc.h"

#include <numeric>

#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

// Handle class
class ExtBoxBlur::Impl {
	friend class ExtBoxBlur;

	double sigma;			///< desired standard deviation
	unsigned times;			///< iterations count

	// Following fields all are derived from sigma and times. See extendedBoxKernel
	unsigned kernelWidth;	///< width of the mask (extended box)
	double boxHeight;		///< ceiling of the box (majority of extended box's values)
	double w;				///< values on the extension edges of the mask
	double w1;				///< boxHeight - w

	Impl() {}

	/// Reconfigures the kernel
	/// See http://www.mia.uni-saarland.de/Publications/gwosdek-ssvm11.pdf for details
	void extendedBoxKernel(double sigma_, unsigned times_) {
		sigma = sigma_; times = times_;

		const double sigmaSq = sigma_*sigma_,
			sigmaSqOverD = sigmaSq/times_,
			idealBoxSize = sqrt(1.+12.*sigmaSqOverD);
		const int l = int((idealBoxSize-1.)/2.),
			lp1 = l+1,
			L = (l<<1)|1;
		const double alpha = L * (l*lp1 - 3.*sigmaSqOverD) / (6. * (sigmaSqOverD - lp1*lp1)),
			lambda = L + alpha + alpha;

		kernelWidth = L + 2;
		boxHeight = 1./lambda;
		w = alpha / lambda;
		w1 = boxHeight - w;
	}

	/// Reconfigure the filter through a new desired standard deviation and a new iterations count
	Impl& setSigma(double desiredSigma, unsigned iterations_ = 1U) {
		if(0U == iterations_)
			THROW_WITH_CONST_MSG("iterations should be > 0 in " __FUNCTION__, invalid_argument);

		if(desiredSigma < 0.)
			THROW_WITH_CONST_MSG("desiredSigma should be > 0 in " __FUNCTION__, invalid_argument);

		extendedBoxKernel(desiredSigma, iterations_);

		return *this;
	}

	/// Reconfigure iterations count for wl and destroys the wu mask
	Impl& setIterations(unsigned iterations_) {
		if(0U == iterations_)
			THROW_WITH_CONST_MSG("iterations should be > 0 in " __FUNCTION__, invalid_argument);

		extendedBoxKernel(sigma, iterations_);

		return *this;
	}

	/// Actual implementation for the current configuration. toBlur is checked; blurred is initialized
	/// See http://www.mia.uni-saarland.de/Publications/gwosdek-ssvm11.pdf for details
	void apply(const cv::Mat &toBlur, cv::Mat &blurred) {
		const int origWidth = toBlur.cols,
			origHeight = toBlur.rows,
			kernelWidthM1 = (int)kernelWidth - 1,
			kernelRadius = (int)kernelWidth>>1,
			kernelRadiusM1 = kernelRadius - 1,
			kernelRadiusP1 = kernelRadius + 1;

		// Apply the number of times requested on the transposed original
		// Temp needs to be processed; blurred holds the outcome
		Mat temp = toBlur.t();
		blurred = Mat(origWidth, origHeight, CV_64FC1);
		int dataRows = temp.rows, dataCols = temp.cols, dataRowsM1 = dataRows - 1, dataColsM1 = dataCols - 1;

		// Lambda to be used for both traversal directions
		const auto applyKernelTimesHorizontally = [&] () {
			for(unsigned iteration = 0U; iteration < times; ++iteration) {
// Unfortunately, the CPU's are already busy in parallel parts calling this blur method,
// so OMP won't make this faster, given current context:
// #pragma omp parallel		
// #pragma omp for schedule(static, 1) nowait
				for(int row = 0; row < dataRows; ++row) {
					// Computations for 1st pixel on current row
					double *resultIt = blurred.ptr<double>(row);
					const double *dataItBegin = temp.ptr<double>(row);
					const double *frontEdge = dataItBegin + kernelRadiusP1;
					const double firstPixel = *dataItBegin,
								lastPixel = dataItBegin[dataColsM1];
					double prevFrontEdgePixel = dataItBegin[kernelRadius], frontEdgePixel,
							prevSum = *resultIt =
								w * (firstPixel + prevFrontEdgePixel) +
								boxHeight * (firstPixel * kernelRadius + accumulate(dataItBegin + 1, dataItBegin + kernelRadius, 0.));

					// Next few pixels use all first pixel from temp as back edge correction
					int col = 1;
					for(; col <= kernelRadius; prevFrontEdgePixel = frontEdgePixel, ++frontEdge, ++col) {
						frontEdgePixel = *frontEdge;
						prevSum += w * (frontEdgePixel - firstPixel) + w1 * (prevFrontEdgePixel - firstPixel);
						*++resultIt = prevSum;
					}

					// Most pixels have both front & back edges for correction
					const double *backEdge = dataItBegin + 1;
					double backEdgePixel, prevBackEdgePixel = firstPixel;
					for(; col < dataCols - kernelRadius; prevFrontEdgePixel = frontEdgePixel, ++frontEdge, prevBackEdgePixel = backEdgePixel, ++backEdge, ++col) {
						frontEdgePixel = *frontEdge;
						backEdgePixel = *backEdge;
						prevSum += w * (frontEdgePixel - prevBackEdgePixel) + w1 * (prevFrontEdgePixel - backEdgePixel);
						*++resultIt = prevSum;
					}

					// Last pixels use last col from temp as front edge correction
					for(; col < dataCols; prevBackEdgePixel = backEdgePixel, ++backEdge, ++col) {
						backEdgePixel = *backEdge;
						prevSum += w * (lastPixel - prevBackEdgePixel) + w1 * (lastPixel - backEdgePixel);
						*++resultIt = prevSum;
					}
				}

				swap(temp, blurred);
			}
		};

		applyKernelTimesHorizontally();

		// Apply the number of times requested on the temporary blurred transposed back
		// Temp needs to be processed; blurred holds the outcome
		temp = temp.t();
		if(blurred.isContinuous())
			blurred = blurred.reshape(1, origHeight);
		else
			blurred = Mat(toBlur.size(), toBlur.type());
		dataRows = temp.rows; dataCols = temp.cols; dataRowsM1 = dataRows - 1; dataColsM1 = dataCols - 1;
		applyKernelTimesHorizontally();

		blurred = temp;
	}
};

ExtBoxBlur::Impl& ExtBoxBlur::nonTinySyms() {
	static ExtBoxBlur::Impl implem;
	return implem;
}

ExtBoxBlur::Impl& ExtBoxBlur::tinySyms() {
	static ExtBoxBlur::Impl implem;
	return implem;
}

ExtBoxBlur::ExtBoxBlur(double desiredSigma, unsigned iterations_/* = 1U*/) {
	setSigma(desiredSigma, iterations_);
}

ExtBoxBlur& ExtBoxBlur::setSigma(double desiredSigma, unsigned iterations_/* = 1U*/) {
	nonTinySyms().setSigma(desiredSigma, iterations_);

	// Tiny symbols should use a sigma = desiredSigma/2.
	tinySyms().setSigma(desiredSigma * .5, iterations_);

	return *this;
}

ExtBoxBlur& ExtBoxBlur::setIterations(unsigned iterations_) {
	nonTinySyms().setIterations(iterations_);
	tinySyms().setIterations(iterations_);

	return *this;
}

void ExtBoxBlur::doProcess(const cv::Mat &toBlur, cv::Mat &blurred, bool forTinySym) const {
	if(forTinySym)
		tinySyms().apply(toBlur, blurred);
	else
		nonTinySyms().apply(toBlur, blurred);
}

const ExtBoxBlur& ExtBoxBlur::configuredInstance() {
	// Extended Box blur with no iterations and desired standard deviation
	extern const double StructuralSimilarity_SIGMA;
	static ExtBoxBlur result(StructuralSimilarity_SIGMA);
	return result;
}
