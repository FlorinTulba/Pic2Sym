/**********************************************************
 Project:     UnitTesting
 File:        mockUi.h

 Author:      Florin Tulba
 Created on:  2016-2-7
 
 Copyright (c) 2016 Florin Tulba
 **********************************************************/

#ifndef H_MOCK_UI
#define H_MOCK_UI

#ifndef UNIT_TESTING
#	error Shouldn't include headers from UnitTesting project unless UNIT_TESTING is defined
#endif

#include "../src/config.h"

#include <opencv2/core.hpp>

class Controller; // The views defined below interact with this class

class CvWin abstract {
protected:
	CvWin(const Controller &ctrler_, const cv::String&) {}

public:
	void setTitle(const std::string&) const {}
	void setOverlay(const std::string&, int = 0) const {}
	void setStatus(const std::string&, int = 0) const {}
	void setPos(int, int) const {}
	void permitResize(bool = true) const {}
	void resize(int, int) const {}
};

class Comparator final : public CvWin {
public:
	Comparator(const Controller &c) : CvWin(c, "") {}
	static void updateTransparency(int, void*) {}
	void setReference(const cv::Mat&) {}
	void setResult(const cv::Mat&) {}
};

class CmapInspect final : public CvWin {
public:
	CmapInspect(const Controller &c) : CvWin(c, "") {}
	static void updatePageIdx(int, void*) {}
	void updatePagesCount(unsigned) {}
	void updateGrid() {}
	void showPage(unsigned) {}
};

class ControlPanel final {
public:
	ControlPanel(Controller&, const Config&) {}
	void updateEncodingsCount(unsigned) {}
	bool encMaxHack() const { return false; }
};

#endif