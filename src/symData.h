/****************************************************************************************
 The application Pic2Sym approximates images by a
 grid of colored symbols with colored backgrounds.

 This file was created on 2016-4-10
 and belongs to the Pic2Sym project.

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

#ifndef H_SYM_DATA
#define H_SYM_DATA

#include <array>

#ifdef UNIT_TESTING
#	include <map>
#endif

#include <opencv2/core/core.hpp>

/// Most symbol information
struct SymData {
	const unsigned long code;		///< the code of the symbol
	const double minVal = 0.;		///< the value of darkest pixel, range 0..1
	const double diffMinMax = 1.;	///< difference between brightest and darkest pixels, each in 0..1
	const double pixelSum;			///< sum of the values of the pixels, each in 0..1
	const cv::Point2d mc;			///< mass center of the symbol given original fg & bg

	enum { // indices of each matrix type within a MatArray object
		FG_MASK_IDX,			///< mask isolating the foreground of the glyph
		BG_MASK_IDX,			///< mask isolating the background of the glyph 
		EDGE_MASK_IDX,			///< mask isolating the edge of the glyph (transition region fg-bg)
		NEG_SYM_IDX,			///< negative of the symbol (0..255 byte)
		GROUNDED_SYM_IDX,		///< symbol shifted (in brightness) to have black background (0..1)
		BLURRED_GR_SYM_IDX,		///< blurred version of the grounded symbol (0..1)
		VARIANCE_GR_SYM_IDX,	///< variance of the grounded symbol

		MATRICES_COUNT ///< KEEP THIS LAST and DON'T USE IT AS INDEX in MatArray objects!
	};

	// For each symbol from cmap, there'll be several additional helpful matrices to store
	// along with the one for the given glyph. The enum from above should be used for selection.
	typedef std::array< const cv::Mat, MATRICES_COUNT > MatArray;

	const MatArray symAndMasks;		///< symbol + other matrices & masks

	SymData(unsigned long code_, double minVal_, double diffMinMax_, double pixelSum_,
			const cv::Point2d &mc_, const MatArray &symAndMasks_) :
			code(code_), minVal(minVal_), diffMinMax(diffMinMax_),
			pixelSum(pixelSum_), mc(mc_), symAndMasks(symAndMasks_) {}

#ifdef UNIT_TESTING
	typedef std::map< int, const cv::Mat > IdxMatMap; ///< Used in the SymData constructor

	/// Constructor that allows filling only the relevant matrices from MatArray
	SymData(unsigned long code_, double minVal_, double diffMinMax_, double pixelSum_,
			const cv::Point2d &mc_, const IdxMatMap &relevantMats);
#endif
};

#endif