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
#	include "../test/mockDlgs.h"

#else // UNIT_TESTING not defined

#ifndef H_DLGS
#define H_DLGS

#include <Windows.h>
#include <tchar.h>
#include <string>

/// Dlg is the base class for the standard Windows dialogs from below
class Dlg /*abstract*/ {
protected:
	std::string result = ""; ///< the result to be returned
	Dlg() {}

public:
	virtual ~Dlg() = 0 {}

	/// Displays the dialog and stores the selection or returns false when canceled
	virtual bool promptForUserChoice() = 0; 
	const std::string& selection() const { return result; }
	virtual void reset() { result.clear(); }
};

/// OpenSave class controls a FileOpenDialog / FileSaveDialog.
class OpenSave /*abstract*/ : public Dlg {
protected:
	OPENFILENAME ofn;		///< structure used by the FileOpenDialog
	TCHAR fNameBuf[1024];	///< buffer for the selected image file
	const bool toOpen = true;	///< most derived classes want Open File Dialog (not Save)

	/// Prepares the dialog
	OpenSave(const TCHAR * const title,		///< displayed title of the dialog
			 const TCHAR * const filter,	///< expected extensions
			 const TCHAR * const defExtension = nullptr,	///< default extension
			 bool toOpen_ = true			///< open or save dialog
			 );

public:
	bool promptForUserChoice() override;
};

/// Selecting an image to transform
class ImgSelector : public OpenSave {
public:
	ImgSelector();
};

/// Selecting a settings file to load / be saved
class SettingsSelector : public OpenSave {
public:
	SettingsSelector(bool toOpen_ = true);
};

/// SelectFont class controls a ChooseFont Dialog.
class SelectFont : public Dlg {
protected:
	CHOOSEFONT cf;		///< structure used by the ChooseFont Dialog
	LOGFONT lf;			///< structure filled with Font information
	bool isBold = false;
	bool isItalic = false;

public:
	SelectFont(); ///< Prepares the dialog

	bool promptForUserChoice() override;

	void reset() override { Dlg::reset(); isBold = isItalic = false; }

	bool bold() const { return isBold; }
	bool italic() const { return isItalic; }
};

#endif // H_DLGS

#endif // UNIT_TESTING not defined
