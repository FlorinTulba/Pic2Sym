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

#ifndef H_PRESENT_CMAP
#define H_PRESENT_CMAP

#include "matchEngine.h"
#include "controllerBase.h"

/**
Provides read-only access to Cmap data.
*/
struct IPresentCmap /*abstract*/ : virtual IController {
	virtual unsigned getFontSize() const = 0; ///< font size determines grid size

	/// Getting the fonts to fill currently displayed page
	virtual MatchEngine::VSymDataCItPair getFontFaces(unsigned from, unsigned maxCount) const = 0;

	/// Prepares for a new font resize operation
	virtual void resetCmapView() = 0;

	/// Attempts to display 1st cmap page, when full. Called after appending each symbol from charmap. 
	virtual void display1stPageIfFull(const std::vector<const PixMapSym> &syms) = 0;

	/// Updates the Cmap View status bar with the details about the symbols
	virtual void showUnofficialSymDetails(unsigned symsCount) const = 0;

	/// Reports the duration of the update of the symbols. elapsed is in seconds
	virtual void reportSymsUpdateDuration(double elapsed) const = 0;

	virtual ~IPresentCmap() = 0 {}
};

#endif