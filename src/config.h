/**********************************************************
 Project:     Pic2Sym
 File:        config.h

 Author:      Florin Tulba
 Created on:  2015-12-20
 
 Copyright (c) 2016 Florin Tulba
 **********************************************************/

#ifndef H_CONFIG
#define H_CONFIG

#include <string>

#include <boost/filesystem/path.hpp>

/*
Config class controls the parameters for transforming one or more images.
*/
class Config final {
	boost::filesystem::path workDir;	// Folder where the application was launched
	boost::filesystem::path cfgPath;	// Path of the configuration file

	unsigned fontSz = 0U;			// Using font height fontSz
	double kSdevFg = 1, kSdevEdge = 1, kSdevBg = 1,
		kContrast = 1;				// powers of factors for glyph correlation
	double kMCsOffset = 1., kCosAngleMCs = 1.; // powers of factors targeting smoothness
	double kGlyphWeight = 1.;		// power of factor aiming fanciness, not correctness
	unsigned threshold4Blank = 0U;	// Using Blank character replacement under this threshold
	unsigned hMaxSyms = 0U;			// Count of resulted horizontal symbols
	unsigned vMaxSyms = 0U;			// Count of resulted vertical symbols

	bool parseCfg(); // Parse res/defaultCfg.txt

public:
	static const unsigned // Limits
		MIN_FONT_SIZE = 7U, MAX_FONT_SIZE = 50U,
		MIN_H_SYMS = 3U, MAX_H_SYMS = 1024U,
		MIN_V_SYMS = 3U, MAX_V_SYMS = 768U,
		MAX_THRESHOLD_FOR_BLANKS = 50U;

	static bool isFontSizeOk(unsigned fs) { return fs>=MIN_FONT_SIZE && fs<=MAX_FONT_SIZE; }
	static bool isHmaxSymsOk(unsigned syms) { return syms>=MIN_H_SYMS && syms<=MAX_H_SYMS; }
	static bool isVmaxSymsOk(unsigned syms) { return syms>=MIN_V_SYMS && syms<=MAX_V_SYMS; }
	static bool isBlanksThresholdOk(unsigned t) { return t < MAX_THRESHOLD_FOR_BLANKS; }

	Config(const std::string &appLaunchPath); // using defaultCfg.txt

	const boost::filesystem::path& getWorkDir() const { return workDir; }

	unsigned getFontSz() const { return fontSz; }
	void setFontSz(unsigned fontSz_) { fontSz = fontSz_; }

	unsigned getMaxHSyms() const { return hMaxSyms; }
	void setMaxHSyms(unsigned syms) { hMaxSyms = syms; }

	unsigned getMaxVSyms() const { return vMaxSyms; }
	void setMaxVSyms(unsigned syms) { vMaxSyms = syms; }

	unsigned getBlankThreshold() const { return threshold4Blank; }
	void setBlankThreshold(unsigned threshold4Blank_) { threshold4Blank = threshold4Blank_; }

	const double& get_kSdevFg() const { return kSdevFg; }
	void set_kSdevFg(double kSdevFg_) { kSdevFg = kSdevFg_; }

	const double& get_kSdevEdge() const { return kSdevEdge; }
	void set_kSdevEdge(double kSdevEdge_) { kSdevEdge = kSdevEdge_; }

	const double& get_kSdevBg() const { return kSdevBg; }
	void set_kSdevBg(double kSdevBg_) { kSdevBg = kSdevBg_; }

	const double& get_kContrast() const { return kContrast; }
	void set_kContrast(double kContrast_) { kContrast = kContrast_; }

	const double& get_kCosAngleMCs() const { return kCosAngleMCs; }
	void set_kCosAngleMCs(double kCosAngleMCs_) { kCosAngleMCs = kCosAngleMCs_; }

	const double& get_kMCsOffset() const { return kMCsOffset; }
	void set_kMCsOffset(double kMCsOffset_) { kMCsOffset = kMCsOffset_; }

	const double& get_kGlyphWeight() const { return kGlyphWeight; }
	void set_kGlyphWeight(double kGlyphWeight_) { kGlyphWeight = kGlyphWeight_; }

#ifdef UNIT_TESTING
	// Constructors used during Unit Testing
	Config(unsigned fontSz_ = 0U,
		   double kSdevFg_ = 0., double kSdevEdge_ = 0., double kSdevBg_ = 0.,
		   double kContrast_ = 0., double kMCsOffset_ = 0., double kCosAngleMCs_ = 0.,
		   double kGlyphWeight_ = 0., unsigned threshold4Blank_ = 0U,
		   unsigned hMaxSyms_ = 0U, unsigned vMaxSyms_ = 0U) :
		fontSz(fontSz_),
		kSdevFg(kSdevFg_), kSdevEdge(kSdevEdge_), kSdevBg(kSdevBg_), kContrast(kContrast_),
		kMCsOffset(kMCsOffset_), kCosAngleMCs(kCosAngleMCs_), kGlyphWeight(kGlyphWeight_),
		threshold4Blank(threshold4Blank_), hMaxSyms(hMaxSyms_), vMaxSyms(vMaxSyms_) {}
#endif
};

#endif
