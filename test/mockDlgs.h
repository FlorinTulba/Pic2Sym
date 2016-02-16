/**********************************************************
 Project:     UnitTesting
 File:        mockDlgs.h

 Author:      Florin Tulba
 Created on:  2016-2-7
 
 Copyright (c) 2016 Florin Tulba
 **********************************************************/

#ifndef H_MOCK_DLGS
#define H_MOCK_DLGS

#ifndef UNIT_TESTING
#	error Shouldn't include headers from UnitTesting project unless UNIT_TESTING is defined
#endif

#include <tchar.h>
#include <string>

// Dlg is the base class for the standard Windows dialogs from below
class Dlg abstract {
protected:
	Dlg() {}

public:
	bool promptForUserChoice() { return true; }
	const std::string& selection() const { static std::string result; return result; }
	void reset() {}
};

class OpenSave abstract : public Dlg {
public:
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