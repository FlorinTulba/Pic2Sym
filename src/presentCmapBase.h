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

#ifndef H_PRESENT_CMAP_BASE
#define H_PRESENT_CMAP_BASE

#include "cmapPerspectiveBase.h"

#pragma warning ( push, 0 )

#include <set>

#pragma warning ( pop )

/// Provides read-only access to Cmap data.
struct IPresentCmap /*abstract*/ {
	/// Getting the fonts to fill currently displayed page
	virtual ICmapPerspective::VPSymDataCItPair getFontFaces(unsigned from, unsigned maxCount) const = 0;

	/// Allows visualizing the symbol clusters within the Cmap View
	virtual const std::set<unsigned>& getClusterOffsets() const = 0;

	/**
	The viewer presents the identified clusters even when they're not used during the image transformation.
	In that case, the splits between the clusters use dashed line instead of a filled line.
	*/
	virtual bool areClustersUsed() const = 0;

	/// Updates the Cmap View status bar with the details about the symbols
	virtual void showUnofficialSymDetails(unsigned symsCount) const = 0;

	virtual ~IPresentCmap() = 0 {}
};

#endif // H_PRESENT_CMAP_BASE
