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

#include <boost/filesystem.hpp>

/*
Config class controls the parameters for transforming one or more images.
*/
class Config final {
	boost::filesystem::path workDir;	// Folder where the application was launched
	boost::filesystem::path cfgPath;	// Path of the configuration file

	unsigned fontSz				= 0U;	// Using font height fontSz
	unsigned outW				= 0U;	// Count of resulted horizontal characters
	unsigned outH				= 0U;	// Count of resulted vertical characters
	unsigned threshold4Blank	= 0U;	// Using Blank character replacement under this threshold

	// powers of used factors; set to 0 to ignore specific factor
	double kSdevFg = 1, kSdevBg = 1, kContrast=1;	// powers of factors for glyph correlation
	double kCosAngleMCs = 1., kMCsOffset = 1.;		// powers of factors targeting smoothness
	double kGlyphWeight = 1.;			// power of factor aiming fanciness, not correctness

	bool parseCfg(); // Parse the edited cfg.txt and update settings if parsing is successful

public:
	Config(const std::string &appLaunchPath); // Initial Parse of default cfg.txt
	~Config(); // Cleanup

	// Prompts for changing the existing config and validates the changes
	// Returns true if the settings have changed
	bool update();

	const boost::filesystem::path& getWorkDir() const { return workDir; }
	unsigned getFontSz() const { return fontSz; }
	unsigned getOutW() const { return outW; }
	unsigned getOutH() const { return outH; }
	unsigned getBlankThreshold() const { return threshold4Blank; }

	double get_kSdevFg() const { return kSdevFg; }
	double get_kSdevBg() const { return kSdevBg; }
	double get_kContrast() const { return kContrast; }
	double get_kCosAngleMCs() const { return kCosAngleMCs; }
	double get_kMCsOffset() const { return kMCsOffset; }
	double get_kGlyphWeight() const { return kGlyphWeight; }

	const std::string joined() const; // returns the settings joined by underscores
};

#endif
