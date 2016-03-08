/****************************************************************************************
 The application Pic2Sym approximates images by a
 grid of colored symbols with colored backgrounds.

 This file was created on 2016-1-22
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

#ifndef H_UI
#define H_UI

#include "match.h"
#include "img.h"

#include <opencv2/core.hpp>

class Settings;		// global settings
class Controller;	// The views defined below interact with this class

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

/**
View which permits comparing the original image with the transformed one.

A slider adjusts the transparency of the resulted image,
so that the original can be more or less visible.
*/
class Comparator final : public CvWin {
	static const cv::String transpTrackName;	///< slider's handle
	static const double defaultTransparency;	///< used transparency when the result appears
	static const int trackMax = 100;			///< transparency range 0..100
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
				   int transparency = (int)round(defaultTransparency * trackMax));
};

/**
Class for displaying the symbols from the current charmap (cmap).

When there are lots of symbols, they are divided into pages which
can be browsed using the page slider.
*/
class CmapInspect final : public CvWin {
	static const cv::String pageTrackName;	///< page slider handle
	static const cv::Size pageSz;			///< 640x480

	const Controller &ctrler;	///< window manager

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

	cv::Mat createGrid() const;				///< generates the grid that separates the glyphs

	/// content = grid + glyphs for current page specified by a pair of iterators
	void populateGrid(const MatchEngine::VSymDataCItPair &itPair);

	unsigned computeSymsPerPage() const;	///< determines how many symbols fit on a single page

public:
	CmapInspect(const Controller &ctrler_);

	static void updatePageIdx(int newPage, void *userdata); ///< slider's callback

	void updatePagesCount(unsigned cmapSize);	///< puts also the slider on 0
	void updateGrid();							///< Changing font size must update also the grid

	void showPage(unsigned pageIdx);			///< displays page <pageIdx>
};

/**
Configures the transformation settings, chooses which image to process and with which cmap.

The range of the slider used for selecting the encoding is updated automatically
based on the selected font family.

The sliders from OpenCV are quite basic, so ranges must be integer, start from 0 and include 1.
Thus, range 0..1 could also mean there's only one value (0) and the 1 will be ignored.
Furthermore, range 0..x can be invalid if the actual range is [x+1 .. y].
In this latter case an error message will report the problem and the user needs to correct it.
*/
class ControlPanel final {
	/**
	Helper to convert settings from actual ranges to the ones used by the sliders.

	For now, flexibility in choosing the conversion rules is more important than design,
	so lots of redundancies appear below.
	*/
	struct Converter {
		/**
		One possible conversion function 'proportionRule':
			x in 0..xMax;  y in 0..yMax  =>
			y = x*yMax/xMax
		*/
		static double proportionRule(double x, double xMax, double yMax);

		/// used for the slider controlling the structural similarity
		struct StructuralSim {
			static const int maxSlider = 100;
			static const double maxReal;
			static int toSlider(double ssim);
			static double fromSlider(int ssim);
		};

		/// used for the 3 sliders controlling the correctness
		struct Correctness {
			static const int maxSlider = 100;
			static const double maxReal;
			static int toSlider(double correctness);
			static double fromSlider(int correctness);
		};

		/// used for the slider controlling the contrast
		struct Contrast {
			static const int maxSlider = 100;
			static const double maxReal;
			static int toSlider(double contrast);
			static double fromSlider(int contrast);
		};

		/// used for the slider controlling the 'gravitational' smoothness
		struct Gravity {
			static const int maxSlider = 100;
			static const double maxReal;
			static int toSlider(double gravity);
			static double fromSlider(int gravity);
		};

		/// used for the slider controlling the directional smoothness
		struct Direction {
			static const int maxSlider = 100;
			static const double maxReal;
			static int toSlider(double direction);
			static double fromSlider(int direction);
		};

		/// used for the slider controlling the preference for larger symbols
		struct LargerSym {
			static const int maxSlider = 100;
			static const double maxReal;
			static int toSlider(double largerSym);
			static double fromSlider(int largerSym);
		};
	};

	// Configuration sliders' handles
	static const cv::String fontSzTrName, encodingTrName, outWTrName, outHTrName;
	static const cv::String hybridResultTrName, structuralSimTrName,
		underGlyphCorrectnessTrName, glyphEdgeCorrectnessTrName, asideGlyphCorrectnessTrName,
		moreContrastTrName;
	static const cv::String gravityTrName, directionTrName, largerSymTrName, thresh4BlanksTrName;

	Controller &ctrler;	// window manager

	// Configuration sliders' positions
	int maxHSyms, maxVSyms;
	int encoding, fontSz;
	int hybridResult;
	int structuralSim, underGlyphCorrectness, glyphEdgeCorrectness, asideGlyphCorrectness;
	int moreContrast, gravity, direction, largerSym, thresh4Blanks;

	/** 
	hack field

	The max of the encodings slider won't update unless
	issuing an additional slider move, which has to be ignored.
	*/
	bool updatingEncMax = false;

public:
	ControlPanel(Controller &ctrler_, const Settings &cfg);

	void updateEncodingsCount(unsigned uniqueEncodings);	///< puts also the slider on 0
	bool encMaxHack() const { return updatingEncMax; }		///< used for the hack above

	/// updates font size & encoding sliders, if necessary
	void updateSymSettings(unsigned encIdx, unsigned fontSz_);
	void updateImgSettings(const ImgSettings &is); ///< updates sliders concerning ImgSettings items
	void updateMatchSettings(const MatchSettings &ms); ///< updates sliders concerning MatchSettings items
};

#endif // H_UI

#endif // UNIT_TESTING not defined
