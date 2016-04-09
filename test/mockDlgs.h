/****************************************************************************************
 The application Pic2Sym approximates images by a
 grid of colored symbols with colored backgrounds.

 This file was created on 2016-2-7
 and belongs to the UnitTesting project.

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

#ifndef H_MOCK_DLGS
#define H_MOCK_DLGS

#ifndef UNIT_TESTING
#	error Shouldn't include headers from UnitTesting project unless UNIT_TESTING is defined
#endif

#include <tchar.h>
#include <string>

// Dlg is the base class for the standard Windows dialogs from below
class Dlg /*abstract*/ {
protected:
	Dlg() {}

public:
	bool promptForUserChoice() { return true; }
	const std::string& selection() const { static std::string result; return result; }
	void reset() {}
};

class OpenSave /*abstract*/ : public Dlg {
protected:
	OpenSave(const TCHAR * const = nullptr, const TCHAR * const = nullptr,
			 const TCHAR * const  = nullptr,
			 bool = true) : Dlg() {}
};

class ImgSelector final : public OpenSave {
public:
	ImgSelector() : OpenSave() {}
};

class SettingsSelector final : public OpenSave {
public:
	SettingsSelector(bool = true) : OpenSave() {}
};

class SelectFont final : public Dlg {
public:
	SelectFont() : Dlg() {}
	bool bold() const { return false; }
	bool italic() const { return false; }
};

#endif