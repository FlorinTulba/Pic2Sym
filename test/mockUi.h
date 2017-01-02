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

#ifndef H_MOCK_UI
#define H_MOCK_UI

#ifndef UNIT_TESTING
#	error Shouldn't include headers from UnitTesting project unless UNIT_TESTING is defined
#endif // UNIT_TESTING not defined

class CvWin /*abstract*/ {
protected:
	CvWin(...) {}

public:
	void setTitle(...) const {}
	void setOverlay(...) const {}
	void setStatus(...) const {}
	void setPos(...) const {}
	void permitResize(...) const {}
	void resize(...) const {}
};

class Comparator : public CvWin {
public:
	Comparator(...) : CvWin(nullptr) {}
	static void updateTransparency(...) {}
	void setReference(...) {}
	void setResult(...) {}
	using CvWin::resize; // to remain visible after declaring an overload below
	void resize(int, int) const {}
};

class CmapInspect : public CvWin {
public:
	CmapInspect(...) : CvWin(nullptr) {}
	static void updatePageIdx(...) {}
	void updatePagesCount(...) {}
	void updateGrid() {}
	void showPage(...) {}
	unsigned getCellSide() const { return 0U; }
	unsigned getSymsPerRow() const { return 0U; }
	unsigned getSymsPerPage() const { return 0U; }
	unsigned getPageIdx() const { return 0U; }
	bool isBrowsable() const { return false; }
	void setBrowsable(bool = true) {}
	void showUnofficial1stPage(...) {}
};

struct ActionPermit {};

class ControlPanel {
public:
	ControlPanel(...) {}
	void updateEncodingsCount(...) {}
	bool encMaxHack() const { return false; }
	void updateMatchSettings(...) {}
	void updateImgSettings(...) {}
	void updateSymSettings(...) {}
	std::unique_ptr<ActionPermit> actionDemand(...) { return std::move(std::make_unique<ActionPermit>()); }
	void restoreSliderValue(...) {}
};

#endif