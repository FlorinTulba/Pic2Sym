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
 
 (c) 2016, 2017 Florin Tulba <florintulba@yahoo.com>

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

#ifndef UNIT_TESTING

#ifndef H_CMAP_INSPECT_BASE
#define H_CMAP_INSPECT_BASE

#include "viewsBase.h"
#include "updateSymsActions.h"

#pragma warning ( push, 0 )

#include <vector>
#include <atomic>

#include <opencv2/core/core.hpp>

#pragma warning ( pop )

/**
Interface for displaying the symbols from the current charmap (cmap).

When there are lots of symbols, they are divided into pages which
can be browsed using the page slider.
*/
struct ICmapInspect /*abstract*/ : virtual ICvWin {
	virtual unsigned getCellSide() const = 0;
	virtual unsigned getSymsPerRow() const = 0;
	virtual unsigned getSymsPerPage() const = 0;
	virtual unsigned getPageIdx() const = 0;
	virtual bool isBrowsable() const = 0;
	virtual void setBrowsable(bool readyToBrowse_ = true) = 0;

	/// Display an 'early' (unofficial) version of the 1st page from the Cmap view, if the official version isn't available yet
	virtual void showUnofficial1stPage(std::vector<const cv::Mat> &symsOn1stPage,
									   std::atomic_flag &updating1stCmapPage,
									   LockFreeQueue &updateSymsActionsQueue) = 0;

	virtual void clear() = 0;								///< clears the grid, the status bar and updates required fields

	virtual void updatePagesCount(unsigned cmapSize) = 0;	///< puts also the slider on 0
	virtual void updateGrid() = 0;							///< Changing font size must update also the grid

	virtual void showPage(unsigned pageIdx) = 0;			///< displays page 'pageIdx'

	virtual ~ICmapInspect() = 0 {}
};

#endif // H_CMAP_INSPECT_BASE

#endif // UNIT_TESTING not defined
