/****************************************************************************************
 The application Pic2Sym approximates images by a
 grid of colored symbols with colored backgrounds.

 This file belongs to the UnitTesting project.

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

#ifndef H_MOCK_UI
#define H_MOCK_UI

#ifndef UNIT_TESTING
#	error Shouldn't include headers from UnitTesting project unless UNIT_TESTING is defined
#endif

#include "../src/settings.h"
#include "../src/img.h"
#include "../src/controller.h"

#include <opencv2/core.hpp>

class Controller; // The views defined below interact with this class

class CvWin /*abstract*/ {
protected:
	CvWin(const cv::String&) {}

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
	Comparator(void** /*hackParam*/ = nullptr) : CvWin("") {}
	static void updateTransparency(int, void*) {}
	void setReference(const cv::Mat&) {}
	void setResult(const cv::Mat&,int=0) {}
	using CvWin::resize; // to remain visible after declaring an overload below
	void resize() const {}
};

class CmapInspect final : public CvWin {
public:
	CmapInspect(const Controller &c) : CvWin("") {}
	static void updatePageIdx(int, void*) {}
	void updatePagesCount(unsigned) {}
	void updateGrid() {}
	void showPage(unsigned) {}
};

class ControlPanel final {
public:
	ControlPanel(Controller&, const Settings&) {}
	void updateEncodingsCount(unsigned) {}
	bool encMaxHack() const { return false; }
	void updateMatchSettings(const MatchSettings&) {}
	void updateImgSettings(const ImgSettings&) {}
	void updateSymSettings(unsigned, unsigned) {}
};

#endif