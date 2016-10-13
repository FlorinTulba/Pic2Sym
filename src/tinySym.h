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

#ifndef H_TINY_SYM
#define H_TINY_SYM

#include <opencv2/core/core.hpp>

struct PixMapSym; // Forward declaration

/// Data for tiny symbols
struct TinySym {
	/// Ratio between reference symbols and the shrunken symbol
	enum { RatioRefTiny = 8 };

	cv::Point2d mc = cv::Point2d(.5, .5);	///< reference mc (coordinates are within a unit-square: 0..1 x 0..1)
	double avgPixVal = 0.;					///< reference pixelSum divided by font area (0..1 range)

	cv::Mat mat;		///< grounded version of the small symbol

	// The average projections from below are for the grounded version, not the original
	cv::Mat hAvgProj, vAvgProj;						// horizontal and vertical projection
	cv::Mat backslashDiagAvgProj, slashDiagAvgProj;	// normal and inverse diagonal projections

	TinySym();
	TinySym(const PixMapSym &refSym);
	TinySym(const cv::Point2d &mc_, double avgPixVal_, const cv::Mat &mat_,
			const cv::Mat &hAvgProj_, const cv::Mat &vAvgProj_,
			const cv::Mat &backslashDiagAvgProj_, const cv::Mat &slashDiagAvgProj_);
};

#endif // H_TINY_SYM