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

#ifdef UNIT_TESTING
#	include "../test/mockUi.h"

#else // UNIT_TESTING not defined

#ifndef H_VIEWS
#define H_VIEWS

#include "comparatorBase.h"
#include "cmapInspectBase.h"
#include "cmapPerspectiveBase.h"
#include "warnings.h"

/**
CvWin - base class for Comparator & CmapInspect from below.
Allows setting title, overlay, status, location, size and resizing properties.
*/
class CvWin /*abstract*/ : public virtual ICvWin {
protected:
	const cv::String winName;	///< window's handle
	cv::Mat content;			///< what to display 

	CvWin(const cv::String &winName_);
	void operator=(const CvWin&) = delete;

public:
	virtual ~CvWin() = 0 {}

	void setTitle(const std::stringType &title) const override;
	void setOverlay(const std::stringType &overlay, int timeoutMs = 0) const override;
	void setStatus(const std::stringType &status, int timeoutMs = 0) const override;

	void setPos(int x, int y) const override;
	void permitResize(bool allow = true) const override;
	void resize(int w, int h) const override;
};

#pragma warning( disable : WARN_INHERITED_VIA_DOMINANCE )
/**
View which permits comparing the original image with the transformed one.

A slider adjusts the transparency of the resulted image,
so that the original can be more or less visible.
*/
class Comparator : public CvWin, public virtual IComparator {
protected:
	static const cv::Mat noImage;				///< image to display initially (not for processing)

	cv::Mat initial, result;	///< Compared items
	int trackPos = 0;			///< Transparency value

	void setTransparency(double transparency);	///< called from updateTransparency

public:	
	Comparator(); ///< Creating a Comparator window.
	void operator=(const Comparator&) = delete;

	static void updateTransparency(int newTransp, void *userdata); ///< slider's callback

	void setReference(const cv::Mat &reference_) override;
	void setResult(const cv::Mat &result_,
				   int transparency =
						(int)round(Comparator_defaultTransparency * Comparator_trackMax)) override;
	
	using CvWin::resize; // to remain visible after declaring the overload below
	void resize() const override;
};
#pragma warning( default : WARN_INHERITED_VIA_DOMINANCE )

// Forward declarations
struct IPresentCmap;
struct ISelectSymbols;

#pragma warning( disable : WARN_INHERITED_VIA_DOMINANCE )
/**
Class for displaying the symbols from the current charmap (cmap).

When there are lots of symbols, they are divided into pages which
can be browsed using the page slider.
*/
class CmapInspect : public CvWin, public virtual ICmapInspect {
protected:
	std::sharedPtr<const IPresentCmap> cmapPresenter;	///< presents the cmap window
	std::sharedPtr<const ISelectSymbols> symsSelector;	///< allows saving a selection of symbols

	cv::Mat grid;				///< the symbols' `hive`
	int page = 0;				///< page slider position
	unsigned pagesCount = 0U;	///< used for dividing the cmap
	const unsigned &fontSz;		///< font size
	unsigned cellSide = 0U;		///< used for dividing the cmap
	unsigned symsPerRow = 0U;	///< used for dividing the cmap
	unsigned symsPerPage = 0U;	///< used for dividing the cmap

	/**
	hack field

	The max of the page slider won't update unless
	issuing an additional slider move, which has to be ignored.
	*/
	bool updatingPageMax = false;

	bool readyToBrowse = false;		///< set to true only when all pages are ready

	cv::Mat createGrid();			///< generates the grid that separates the glyphs

	/// content = grid + glyphs for current page specified by a pair of iterators
	void populateGrid(const ICmapPerspective::VPSymDataCItPair &itPair,
					  const std::set<unsigned> &clusterOffsets,
					  unsigned idxOfFirstSymFromPage);

public:
	CmapInspect(std::sharedPtr<const IPresentCmap> cmapPresenter_,
				std::sharedPtr<const ISelectSymbols> symsSelector_,
				const unsigned &fontSz_);
	void operator=(const CmapInspect&) = delete;

	static void updatePageIdx(int newPage, void *userdata); ///< slider's callback

	unsigned getCellSide() const override final { return cellSide; }
	unsigned getSymsPerRow() const override final { return symsPerRow; }
	unsigned getSymsPerPage() const override final { return symsPerPage; }
	unsigned getPageIdx() const override final { return (unsigned)page; }
	bool isBrowsable() const override final { return readyToBrowse; }
	void setBrowsable(bool readyToBrowse_ = true) override final { readyToBrowse = readyToBrowse_; }

	/// Display an 'early' (unofficial) version of the 1st page from the Cmap view, if the official version isn't available yet
	void showUnofficial1stPage(std::vector<const cv::Mat> &symsOn1stPage,
							   std::atomic_flag &updating1stCmapPage,
							   LockFreeQueue &updateSymsActionsQueue) override;

	void clear() override;								///< clears the grid, the status bar and updates required fields

	void updatePagesCount(unsigned cmapSize) override;	///< puts also the slider on 0
	void updateGrid() override;							///< Changing font size must update also the grid

	void showPage(unsigned pageIdx) override;			///< displays page 'pageIdx'
};
#pragma warning( default : WARN_INHERITED_VIA_DOMINANCE )

#endif // H_VIEWS

#endif // UNIT_TESTING not defined
