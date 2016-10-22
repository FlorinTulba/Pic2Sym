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

#ifndef H_CACHED_DATA
#define H_CACHED_DATA

#include <opencv2/core/core.hpp>

// forward declarations
class MatchEngine;
class FontEngine;

/// cached data for computing match parameters and evaluating match aspects
struct CachedData {
	static inline const cv::Point2d& unitSquareCenter() {
		static const cv::Point2d center(.5, .5);
		return center;
	}

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
	
	/// 1 / max possible distance between mass centers: sqrt(2) - preferredMaxMcDist
	static const double invComplPrefMaxMcDist() {
		static const double result = 1. / (sqrt(2) - preferredMaxMcDist());
		return result;
	}

	// See comment from above the definitions of these static methods in cachedData.cpp, but also from DirectionalSmoothness::score
	static const double a_mcsOffsetFactor();	///< mcsOffsetFactor = a * mcsOffset + b
	static const double b_mcsOffsetFactor();	///< mcsOffsetFactor = a * mcsOffset + b

	double sz_1;				///< double version of sz - 1
	double smallGlyphsCoverage;	///< max density for symbols considered small

	cv::Mat consec;				///< row matrix with consecutive elements: 0..sz-1

	const bool forTinySyms;		///< Are all these values used for tiny symbols or normal ones?

	CachedData(bool forTinySyms_ = false);

protected:
	friend class MatchEngine;
	void update(unsigned sz_, const FontEngine &fe_);
	void update(const FontEngine &fe_);

#ifdef UNIT_TESTING // UnitTesting project should have public access to 'useNewSymSize' method
public:
#endif
	void useNewSymSize(unsigned sz_);
};

#endif