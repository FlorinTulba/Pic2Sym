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

#ifdef UNIT_TESTING
	Config(unsigned fontSz_, unsigned outW_, unsigned outH_, unsigned threshold4Blank_,
		   double kSdevFg_, double kSdevBg_, double kContrast_,
		   double kCosAngleMCs_, double kMCsOffset_, double kGlyphWeight_) :
		   fontSz(fontSz_), outW(outW_), outH(outH_), threshold4Blank(threshold4Blank_),
		   kSdevFg(kSdevFg_), kSdevBg(kSdevBg_), kContrast(kContrast_),
		   kCosAngleMCs(kCosAngleMCs_), kMCsOffset(kMCsOffset_), kGlyphWeight(kGlyphWeight_) {}
	Config() {}

	void setFontSz(unsigned fontSz_) { fontSz = fontSz_; }
	void set_kSdevFg(double kSdevFg_) { kSdevFg = kSdevFg_; }
	void set_kSdevBg(double kSdevBg_) { kSdevBg = kSdevBg_; }
	void set_kContrast(double kContrast_) { kContrast = kContrast_; }
	void set_kCosAngleMCs(double kCosAngleMCs_) { kCosAngleMCs = kCosAngleMCs_; }
	void set_kMCsOffset(double kMCsOffset_) { kMCsOffset = kMCsOffset_; }
	void set_kGlyphWeight(double kGlyphWeight_) { kGlyphWeight = kGlyphWeight_; }
#endif
};

#endif
