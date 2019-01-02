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

#ifndef H_MOCK_UI
#define H_MOCK_UI

#ifndef UNIT_TESTING
#	error Shouldn't include headers from UnitTesting project unless UNIT_TESTING is defined
#endif // UNIT_TESTING not defined

#include "warnings.h"

struct ICvWin /*abstract*/ {
	virtual void setTitle(...) const = 0;
	virtual void setOverlay(...) const = 0;
	virtual void setStatus(...) const = 0;
	virtual void setPos(...) const = 0;
	virtual void permitResize(...) const = 0;
	virtual void resize(...) const = 0;

	virtual ~ICvWin() = 0 {}
};

class CvWin /*abstract*/ : public virtual ICvWin {
protected:
	CvWin() = default;

public:
	void setTitle(...) const {}
	void setOverlay(...) const {}
	void setStatus(...) const {}
	void setPos(...) const {}
	void permitResize(...) const {}
	void resize(...) const {}
};

struct IComparator : virtual ICvWin {
	virtual void setReference(...) = 0;
	virtual void setResult(...) = 0;

	virtual ~IComparator() = 0 {}
};

struct ICmapInspect /*abstract*/ : virtual ICvWin {
	virtual unsigned getCellSide() const = 0;
	virtual unsigned getSymsPerRow() const = 0;
	virtual unsigned getSymsPerPage() const = 0;
	virtual unsigned getPageIdx() const = 0;
	virtual bool isBrowsable() const = 0;
	virtual void setBrowsable(...) = 0;

	/// Display an 'early' (unofficial) version of the 1st page from the Cmap view, if the official version isn't available yet
	virtual void showUnofficial1stPage(...) = 0;

	virtual void clear() = 0;								///< clears the grid, the status bar and updates required fields

	virtual void updatePagesCount(...) = 0;	///< puts also the slider on 0
	virtual void updateGrid() = 0;							///< Changing font size must update also the grid

	virtual void showPage(...) = 0;			///< displays page 'pageIdx'

	virtual ~ICmapInspect() = 0 {}
};

#pragma warning( disable : WARN_INHERITED_VIA_DOMINANCE )
class Comparator : public CvWin, public virtual IComparator {
public:
	static void updateTransparency(...) {}
	void setReference(...) {}
	void setResult(...) {}
	void resize(...) const {}
};

class CmapInspect : public CvWin, public virtual ICmapInspect {
public:
	CmapInspect(...) : CvWin() {}
	static void updatePageIdx(...) {}
	void updatePagesCount(...) {}
	void updateGrid() {}
	void clear() {}
	void showPage(...) {}
	unsigned getCellSide() const { return 0U; }
	unsigned getSymsPerRow() const { return 0U; }
	unsigned getSymsPerPage() const { return 0U; }
	unsigned getPageIdx() const { return 0U; }
	bool isBrowsable() const { return false; }
	void setBrowsable(...) {}
	void showUnofficial1stPage(...) {}
};
#pragma warning( default : WARN_INHERITED_VIA_DOMINANCE )

struct ActionPermit {};

/// Interface of ControlPanel
struct IControlPanel /*abstract*/ {
	virtual void restoreSliderValue(...) = 0;
	virtual std::unique_ptr<const ActionPermit> actionDemand(...) = 0;
	virtual void updateEncodingsCount(...) = 0;
	virtual bool encMaxHack() const = 0;
	virtual void updateSymSettings(...) = 0;
	virtual void updateImgSettings(...) = 0;
	virtual void updateMatchSettings(...) = 0;

	virtual ~IControlPanel() = 0 {}
};

class ControlPanel : public IControlPanel {
public:
	ControlPanel(...) {}
	void updateEncodingsCount(...) {}
	bool encMaxHack() const { return false; }
	void updateMatchSettings(...) {}
	void updateImgSettings(...) {}
	void updateSymSettings(...) {}
	std::unique_ptr<const ActionPermit> actionDemand(...) { return std::move(std::make_unique<const ActionPermit>()); }
	void restoreSliderValue(...) {}
};

#endif // H_MOCK_UI
