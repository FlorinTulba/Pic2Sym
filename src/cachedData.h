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
 ***********************************************************************************************/

#ifndef H_CACHED_DATA
#define H_CACHED_DATA

#pragma warning ( push, 0 )

#include <opencv2/core/core.hpp>

#pragma warning ( pop )

// Forward declarations
class MatchEngine;
class FontEngine;

/// Cached data for computing match parameters and evaluating match aspects
struct CachedData {
	/**
	Max possible std dev = 127.5  for foreground / background.
	Happens for an error matrix with a histogram with 2 equally large bins on 0 and 255.
	In that case, the mean is 127.5 and the std dev is:
	sqrt( ((-127.5)^2 * sz^2/2 + 127.5^2 * sz^2/2) /sz^2) = 127.5
	*/
	static inline const double sdevMaxFgBg() { return 127.5; }

	/**
	Max possible std dev for edge is 255.
	This happens in the following situation:
	a) Foreground and background masks cover an empty area of the patch =>
	approximated patch will be completely black
	b) Edge mask covers a full brightness (255) area of the patch =>
	every pixel from the patch covered by the edge mask has a deviation of 255 from
	the corresponding zone within the approximated patch.
	*/
	static inline const double sdevMaxEdge() { return 255.; }

	/// acceptable distance between mass centers (1/8)
	static inline const double preferredMaxMcDist() { return .125; }
		
	/// The center of a square with unit-length sides
	static const cv::Point2d& unitSquareCenter();

	/// 1 / max possible distance between mass centers: sqrt(2) - preferredMaxMcDist
	static const double invComplPrefMaxMcDist();

	// See comment from above the definitions of these static methods in cachedData.cpp, but also from DirectionalSmoothness::score
	static const double a_mcsOffsetFactor();	///< mcsOffsetFactor = a * mcsOffset + b
	static const double b_mcsOffsetFactor();	///< mcsOffsetFactor = a * mcsOffset + b

	cv::Mat consec;				///< row matrix with consecutive elements: 0..sz-1

	double sz_1;				///< double version of sz - 1
	double smallGlyphsCoverage;	///< max density for symbols considered small

	const bool forTinySyms;		///< Are all these values used for tiny symbols or normal ones?

	CachedData(bool forTinySyms_ = false);
	void operator=(const CachedData&) = delete;

	void update(unsigned sz_, const FontEngine &fe_);
	void update(const FontEngine &fe_);
	void useNewSymSize(unsigned sz_);
};

#endif