/**********************************************************
 Project:     Pic2Sym
 File:        ui.h

 Author:      Florin Tulba
 Created on:  2016-1-22
 
 Copyright (c) 2016 Florin Tulba
 **********************************************************/

#ifndef H_UI
#define H_UI

#include "transform.h"

#include <memory>
#include <opencv2/core.hpp>

class Controller; // The views defined below interact with this class

#ifdef UNIT_TESTING
class Comparator {
public:
	Comparator(const Controller&) {}
	void setTitle(const std::string&) const {}
	void setOverlay(const std::string&, int = 0) const {}
	void setStatus(const std::string&, int = 0) const {}
	void setPos(int, int) const {}
	virtual void permitResize(bool = true) const {}
	void resize(int, int) const {}
	static void updateTransparency(int, void*) {}
	void setReference(const cv::Mat&) {}
	void setResult(const cv::Mat&) {}
};

class CmapInspect {
public:
	CmapInspect(const Controller&) {}
	void setTitle(const std::string&) const {}
	void setOverlay(const std::string&, int = 0) const {}
	void setStatus(const std::string&, int = 0) const {}
	void setPos(int, int) const {}
	virtual void permitResize(bool = true) const {}
	void resize(int, int) const {}
	static void updatePageIdx(int, void*) {}
	void updatePagesCount(unsigned) {}
	void updateGrid() {}
	void showPage(unsigned) {}
};

class ControlPanel {
public:
	ControlPanel(Controller&, const Config&) {}
	void updateEncodingsCount(unsigned) {}
	bool encMaxHack() const { return false; }
};

#else // UNIT_TESTING not defined

/*
CvWin - base class for Comparator & CmapInspect from below.
Allows setting title, overlay, status, location, size and resizing properties.
*/
class CvWin abstract {
protected:
	const Controller &ctrler;			// window manager

	const cv::String winName;	// window's handle
	cv::Mat content;			// what to display 

public:
	CvWin(const Controller &ctrler_, const cv::String &winName_);
	virtual ~CvWin() = 0 {}

	void setTitle(const std::string &title) const;
	void setOverlay(const std::string &overlay, int timeoutMs = 0) const;
	void setStatus(const std::string &status, int timeoutMs = 0) const;

	void setPos(int x, int y) const;
	virtual void permitResize(bool allow = true) const;
	void resize(int w, int h) const;
};

/*
View which permits comparing the original image with the transformed one.
A slider adjusts the transparency of the resulted image,
so that the original can be more or less visible.
*/
class Comparator : public CvWin {
	static const cv::String transpTrackName;	// slider's handle
	static const double defaultTransparency;	// used transparency when the result appears
	static const int trackMax = 100;			// transparency range 0..100
	static const cv::Mat noImage;				// image to display initially (not for processing)

	cv::Mat initial, result;	// Compared items
	int trackPos = 0;			// Transparency value

	void setTransparency(double transparency);	// called from updateTransparency

public:
	Comparator(const Controller &ctrler_);

	static void updateTransparency(int newTransp, void *userdata); // slider's callback

	void setReference(const cv::Mat &reference_);
	void setResult(const cv::Mat &result_);
};

/*
Class for displaying the symbols from the current charmap (cmap).
When there are lots of symbols, they are divided into pages which
can be browsed using the page slider.
*/
class CmapInspect : public CvWin {
	static const cv::String pageTrackName;	// page slider handle
	static const cv::Size pageSz;			// 640x480

	cv::Mat grid;				// the symbols' `hive`
	int page = 0;				// page slider position
	unsigned pagesCount = 0U, symsPerPage = 0U; // used for dividing the cmap

	// The max of the page slider won't update unless
	// issuing an additional slider move, which has to be ignored.
	// 'updatingPageMax' is used for this hack
	bool updatingPageMax = false;

	cv::Mat createGrid() const;				// generates the grid that separates the glyphs

	// content = grid + glyphs for current page specified by a pair of iterators
	void populateGrid(const MatchEngine::VSymDataCItPair &itPair);

	unsigned computeSymsPerPage() const;	// determines how many symbols fit on a single page

public:
	CmapInspect(const Controller &ctrler_);

	static void updatePageIdx(int newPage, void *userdata); // slider's callback

	void updatePagesCount(unsigned cmapSize);	// puts also the slider on 0
	void updateGrid();							// Changing font size must update also the grid

	void showPage(unsigned pageIdx);			// displays page <pageIdx>
};

/*
Configures the transformation settings, chooses which image to process and with which cmap.
The range of the slider used for selecting the encoding is updated automatically
based on the selected font family.

The sliders from OpenCV are quite basic, so ranges must be integer, start from 0 and include 1.
Thus, range 0..1 could also mean there's only one value (0) and the 1 will be ignored.
Furthermore, range 0..x can be invalid if the actual range is [x+1 .. y].
In this latter case an error message will report the problem and the user needs to correct it.
*/
class ControlPanel {
	/*
	Helper to convert settings from actual ranges to the ones used by the sliders.
	For now, flexibility in choosing the conversion rules is more important than design,
	so lots of redundancies appear below.
	*/
	struct Converter {
		/*
		One possible conversion function 'proportionRule':
			x in 0..xMax;  y in 0..yMax  =>
			y = x*yMax/xMax
		*/
		static double proportionRule(double x, double xMax, double yMax);

		// used for the 3 sliders controlling the correctness
		struct Correctness {
			static const int maxSlider = 100;
			static const double maxReal;
			static int toSlider(double correctness);
			static double fromSlider(int correctness);
		};

		// used for the slider controlling the contrast
		struct Contrast {
			static const int maxSlider = 100;
			static const double maxReal;
			static int toSlider(double contrast);
			static double fromSlider(int contrast);
		};

		// used for the slider controlling the 'gravitational' smoothness
		struct Gravity {
			static const int maxSlider = 100;
			static const double maxReal;
			static int toSlider(double gravity);
			static double fromSlider(int gravity);
		};

		// used for the slider controlling the directional smoothness
		struct Direction {
			static const int maxSlider = 100;
			static const double maxReal;
			static int toSlider(double direction);
			static double fromSlider(int direction);
		};

		// used for the slider controlling the preference for larger symbols
		struct LargerSym {
			static const int maxSlider = 100;
			static const double maxReal;
			static int toSlider(double largerSym);
			static double fromSlider(int largerSym);
		};
	};

	// Configuration sliders' handles
	static const cv::String fontSzTrName, encodingTrName, outWTrName, outHTrName;
	static const cv::String underGlyphCorrectnessTrName, glyphEdgeCorrectnessTrName, asideGlyphCorrectnessTrName, moreContrastTrName;
	static const cv::String gravityTrName, directionTrName, largerSymTrName, thresh4BlanksTrName;

	Controller &ctrler;	// window manager

	// Configuration sliders' positions
	int maxHSyms, maxVSyms;
	int encoding, fontSz;
	int underGlyphCorrectness, glyphEdgeCorrectness, asideGlyphCorrectness;
	int moreContrast, gravity, direction, largerSym, thresh4Blanks;

	// The max of the encodings slider won't update unless
	// issuing an additional slider move, which has to be ignored.
	// 'updatingEncMax' is used for this hack
	bool updatingEncMax = false;

public:
	ControlPanel(Controller &ctrler_, const Config &cfg);

	void updateEncodingsCount(unsigned uniqueEncodings);	// puts also the slider on 0
	bool encMaxHack() const { return updatingEncMax; }		// used for the hack above
};

#endif // UNIT_TESTING

#endif