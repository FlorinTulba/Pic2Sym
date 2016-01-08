/**********************************************************
 Project:     Pic2Sym
 File:        config.h

 Author:      Florin Tulba
 Created on:  2015-12-20
 
 Copyright (c) 2016 Florin Tulba
 **********************************************************/

#ifndef H_CONFIG
#define H_CONFIG

#include "fontEngine.h"

#include <string>

/*
Config class controls the parameters for transforming one or more images.
*/
class Config final {
	std::string curDir;	// Folder where the application was launched

	unsigned fontSz				= 0U; // Using font height fontSz
	unsigned outW				= 0U; // Count of resulted horizontal characters
	unsigned outH				= 0U; // Count of resulted vertical characters
	unsigned threshold4Blank	= 0U; // Using Blank character replacement under this threshold

	bool parseCfg(); // Parse the edited cfg.txt and update settings if parsing is successful

public:
	Config(const std::string &appLaunchPath); // Initial Parse of default cfg.txt

	void update(); // Prompts for changing the existing config and validates the changes

	std::string getCurDir() const { return curDir; }
	unsigned getFontSz() const { return fontSz; }
	unsigned getOutW() const { return outW; }
	unsigned getOutH() const { return outH; }
	unsigned getBlankThreshold() const { return threshold4Blank; }
};

#endif
