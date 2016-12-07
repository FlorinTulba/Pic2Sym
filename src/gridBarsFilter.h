/************************************************************************************************
 The application Pic2Sym approximates images by a
 grid of colored symbols with colored backgrounds.

 This file belongs to the Pic2Sym project.

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

#ifndef H_GRID_BARS_FILTER
#define H_GRID_BARS_FILTER

#include "symFilter.h"

#pragma warning ( push, 0 )

#include <opencv2/core/core.hpp>

#pragma warning ( pop )

/**
Detects symbols typically used to generate a grid from glyphs.

Such characters are less desirable, since the image to be processed is already split as a grid,
so approximating patches with grid-like symbols produces the impression of further division.

These symbols are quite elusive:
- they might expand even towards the corners (when the borders they define are double-lines)
- they might not touch the borders of the glyph
- some of their branches might be thinner/thicker or single/double-lined
- the brightness of each branch isn't always constant, nor it has a constant profile

After lots of approaches I still miss many true positives and get numerous false positives.

It appears that supervised learning would be ideal here, instead of manually evolving a model.
It would be much easier just to provide a set of positives and negatives to a machine learning 
algorithm and then check its accuracy.
*/
struct GridBarsFilter : public TSymFilter<GridBarsFilter> {
	CHECK_ENABLED_SYM_FILTER(GridBarsFilter);

	static bool isDisposable(const PixMapSym &pms, const SymFilterCache &sfc); // static polymorphism

	GridBarsFilter(std::unique_ptr<ISymFilter> nextFilter_ = nullptr);
	GridBarsFilter(const GridBarsFilter&) = delete;
	void operator=(const GridBarsFilter&) = delete;

protected:
	static bool checkProjectionForGridSymbols(const cv::Mat &sums);
};

#endif