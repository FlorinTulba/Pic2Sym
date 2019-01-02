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
 
 (c) 2016-2019 Florin Tulba <florintulba@yahoo.com>

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

#ifndef H_SYM_DATA_BASE
#define H_SYM_DATA_BASE

#pragma warning ( push, 0 )

#include "std_memory.h"
#include <array>
#include <vector>

#include <opencv2/core/core.hpp>

#pragma warning ( pop )

/// Interface for symbol data
struct ISymData /*abstract*/ {
	enum { // indices of each matrix type within a MatArray object
		FG_MASK_IDX,			///< mask isolating the foreground of the glyph
		BG_MASK_IDX,			///< mask isolating the background of the glyph 
		EDGE_MASK_IDX,			///< mask isolating the edge of the glyph (transition region fg-bg)
		GROUNDED_SYM_IDX,		///< symbol shifted (in brightness) to have black background (0..1)
		BLURRED_GR_SYM_IDX,		///< blurred version of the grounded symbol (0..1)
		VARIANCE_GR_SYM_IDX,	///< variance of the grounded symbol

		MATRICES_COUNT ///< KEEP THIS LAST and DON'T USE IT AS INDEX in MatArray objects!
	};

	// For each symbol from cmap, there'll be several additional helpful matrices to store
	// along with the one for the given glyph. The enum from above should be used for selection.
	typedef std::array< cv::Mat, MATRICES_COUNT > MatArray;

	/// Retrieve specific mask
	const cv::Mat& getMask(size_t enumIdx) const {
		assert(enumIdx < MATRICES_COUNT);
		return getMasks()[enumIdx];
	}

	/// mass center of the symbol given original fg & bg (coordinates are within a unit-square: 0..1 x 0..1)
	virtual const cv::Point2d& getMc() const = 0;

	/// negative of the symbol (0..255 byte for normal symbols; double for tiny)
	virtual const cv::Mat& getNegSym() const = 0;

	/// various masks
	virtual const MatArray& getMasks() const = 0;

	/// symbol index within cmap
	virtual size_t getSymIdx() const = 0;
	
#ifdef UNIT_TESTING
	/// the value of darkest pixel, range 0..1
	virtual double getMinVal() const = 0;
#endif // UNIT_TESTING defined

	/// difference between brightest and darkest pixels, each in 0..1
	virtual double getDiffMinMax() const = 0;

	/// average pixel value, each pixel in 0..1
	virtual double getAvgPixVal() const = 0;

	/// the code of the symbol
	virtual unsigned long getCode() const = 0;

	/**
	Enabled symbol filters might mark this symbol as removable,
	but PreserveRemovableSymbolsForExamination from configuration might allow it to remain
	in the active symbol set used during image transformation.

	However, when removable == true && PreserveRemovableSymbolsForExamination,
	the symbol will appear as marked (inversed) in the cmap viewer

	This field doesn't need to be serialized, as filtering options might be different
	for distinct run sessions.
	*/
	virtual bool isRemovable() const = 0;

	virtual ~ISymData() = 0 {}
};

/// VSymData - vector with most information about each symbol
typedef std::vector<const std::uniquePtr<const ISymData>> VSymData;

#endif // H_SYM_DATA_BASE
