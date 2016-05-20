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

#ifdef UNIT_TESTING
#	include "../test/mockUi.h"

#else // UNIT_TESTING not defined

#ifndef H_VIEWS
#define H_VIEWS

#include "matchEngine.h"
#include "img.h"

#include <opencv2/core.hpp>

/**
CvWin - base class for Comparator & CmapInspect from below.
Allows setting title, overlay, status, location, size and resizing properties.
*/
class CvWin /*abstract*/ {
protected:
	const cv::String winName;	///< window's handle
	cv::Mat content;			///< what to display 

	CvWin(const cv::String &winName_);

public:
	virtual ~CvWin() = 0 {}

	void setTitle(const std::string &title) const;
	void setOverlay(const std::string &overlay, int timeoutMs = 0) const;
	void setStatus(const std::string &status, int timeoutMs = 0) const;

	void setPos(int x, int y) const;
	virtual void permitResize(bool allow = true) const;
	void resize(int w, int h) const;
};

extern const int Comparator_trackMax;
extern const double Comparator_defaultTransparency;

/**
View which permits comparing the original image with the transformed one.

A slider adjusts the transparency of the resulted image,
so that the original can be more or less visible.
*/
class Comparator : public CvWin {
protected:
	static const cv::Mat noImage;				///< image to display initially (not for processing)

	cv::Mat initial, result;	///< Compared items
	int trackPos = 0;			///< Transparency value

	void setTransparency(double transparency);	///< called from updateTransparency

public:
	/**
	Creating a Comparator window.

	The parameter just supports a macro mechanism that creates several object types
	with variable number of parameters.

	For Comparator, instead of 'Comparator field;', it would generate 'Comparator field();'
	which is interpreted as a function declaration.

	Adding this extra param generates no harm in the rest of the project,
	but allows the macro to see it as object 'Comparator field(nullptr);', not a function.
	*/
	Comparator(void** /*hackParam*/ =nullptr);

	static void updateTransparency(int newTransp, void *userdata); ///< slider's callback

	void setReference(const cv::Mat &reference_);
	void setResult(const cv::Mat &result_,
				   int transparency =
						(int)round(Comparator_defaultTransparency * Comparator_trackMax));
	
	using CvWin::resize; // to remain visible after declaring an overload below
	void resize() const;
};

struct IPresentCmap; // forward declaration

/**
Class for displaying the symbols from the current charmap (cmap).

When there are lots of symbols, they are divided into pages which
can be browsed using the page slider.
*/
class CmapInspect : public CvWin {
protected:
	const IPresentCmap &cmapPresenter;	///< window manager

	cv::Mat grid;				///< the symbols' `hive`
	int page = 0;				///< page slider position
	unsigned pagesCount = 0U;	///< used for dividing the cmap
	unsigned symsPerPage = 0U;	///< used for dividing the cmap

	/**
	hack field

	The max of the page slider won't update unless
	issuing an additional slider move, which has to be ignored.
	*/
	bool updatingPageMax = false;

	unsigned computeSymsPerPage() const;	///< determines how many symbols fit on a single page

	cv::Mat createGrid() const;				///< generates the grid that separates the glyphs

	/// content = grid + glyphs for current page specified by a pair of iterators
	void populateGrid(const MatchEngine::VSymDataCItPair &itPair);

public:
	CmapInspect(const IPresentCmap &cmapPresenter_);

	static void updatePageIdx(int newPage, void *userdata); ///< slider's callback

	void reset();

	unsigned getSymsPerPage() const { return symsPerPage; }

	/// Display an 'early' (unofficial) version of the 1st page from the Cmap view
	void showUnofficial1stPage(const std::vector<const cv::Mat> &&symsOn1stPage);

	void updatePagesCount(unsigned cmapSize);	///< puts also the slider on 0
	void updateGrid();							///< Changing font size must update also the grid

	void showPage(unsigned pageIdx);			///< displays page <pageIdx>
};

#endif // H_VIEWS

#endif // UNIT_TESTING not defined
