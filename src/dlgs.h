/**********************************************************
 Project:     Pic2Sym
 File:        dlgs.h

 Author:      Florin Tulba
 Created on:  2015-12-21
 
 Copyright (c) 2016 Florin Tulba
 **********************************************************/

#ifndef H_DLGS
#define H_DLGS

#include <Windows.h>
#include <tchar.h>

#include <string>

// Dlg is the base class for the standard Windows dialogs from below
class Dlg abstract {
protected:
	std::string result = ""; // the result to be returned
	Dlg() {}

public:
	virtual ~Dlg() = 0 {}

	virtual bool promptForUserChoice() = 0; // Displays the dialog and stores the selection or returns false when canceled
	const std::string& selection() const { return result; }
	virtual void reset() { result.clear(); }
};

// FileOpen class controls a FileOpenDialog.
class FileOpen final : public Dlg {
	OPENFILENAME ofn;		// structure used by the FileOpenDialog
	TCHAR fNameBuf[1024];	// buffer for the selected image file

public:
	FileOpen(); // Prepares the dialog

	bool promptForUserChoice() override;
};

// SelectFont class controls a ChooseFont Dialog.
class SelectFont final : public Dlg {
	CHOOSEFONT cf;		// structure used by the ChooseFont Dialog
	LOGFONT lf;			// structure filled with Font information
	bool isBold = false;
	bool isItalic = false;

public:
	SelectFont(); // Prepares the dialog

	bool promptForUserChoice() override;

	void reset() override { Dlg::reset(); isBold = isItalic = false; }

	bool bold() const { return isBold; }
	bool italic() const { return isItalic; }
};

#endif