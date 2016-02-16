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
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/serialization/split_member.hpp>
#include <boost/serialization/version.hpp>

// MatchSettings class controls the matching parameters for transforming one or more images.
class MatchSettings {
	boost::filesystem::path workDir;	// Folder where the application was launched
	boost::filesystem::path cfgPath;	// Path of the configuration file

	double kSdevFg = 0., kSdevEdge = 0., kSdevBg = 0.; // powers of factors for glyph correlation
	double kContrast = 0.;						// power of factor for the resulted glyph contrast
	double kMCsOffset = 0., kCosAngleMCs = 0.;	// powers of factors targeting smoothness
	double kGlyphWeight = 0.;		// power of factor aiming fanciness, not correctness
	unsigned threshold4Blank = 0U;	// Using Blank character replacement under this threshold

	template<class Archive>
	void load(Archive &ar, const unsigned version) {
		// It is useful to see which settings changed when loading
		MatchSettings defSettings(*this); // create as copy of previous values

		// read user default match settings
		ar&defSettings.kSdevFg; ar&defSettings.kSdevEdge; ar&defSettings.kSdevBg;
		ar&defSettings.kContrast;
		ar&defSettings.kMCsOffset; ar&defSettings.kCosAngleMCs;
		ar&defSettings.kGlyphWeight;
		ar&defSettings.threshold4Blank;

		// these show message when there are changes
		set_kSdevFg(defSettings.kSdevFg);
		set_kSdevEdge(defSettings.kSdevEdge);
		set_kSdevBg(defSettings.kSdevBg);
		set_kContrast(defSettings.kContrast);
		set_kMCsOffset(defSettings.kMCsOffset);
		set_kCosAngleMCs(defSettings.kCosAngleMCs);
		set_kGlyphWeight(defSettings.kGlyphWeight);
		setBlankThreshold(defSettings.threshold4Blank);
	}
	template<class Archive>
	void save(Archive &ar, const unsigned version) const {
		ar&kSdevFg; ar&kSdevEdge; ar&kSdevBg;
		ar&kContrast;
		ar&kMCsOffset; ar&kCosAngleMCs;
		ar&kGlyphWeight;
		ar&threshold4Blank;
	}
	BOOST_SERIALIZATION_SPLIT_MEMBER();
	friend class boost::serialization::access;

public:
	/*
	Initializes the fields from initMatchSettings.cfg when this file exists,
	otherwise from res/defaultMatchSettings.txt.
	The latter file is used to conveniently alter the defaults during development.

	The parameter appLaunchPath is the path to 'Pic2Sym.exe' and is used to determine
	the folder where to look for 'res/defaultMatchSettings.txt' and 'initMatchSettings.cfg'
	*/
	MatchSettings(const std::string &appLaunchPath);

	const boost::filesystem::path& getWorkDir() const { return workDir; }

	unsigned getBlankThreshold() const { return threshold4Blank; }
	void setBlankThreshold(unsigned threshold4Blank_);

	const double& get_kSdevFg() const { return kSdevFg; }
	void set_kSdevFg(double kSdevFg_);

	const double& get_kSdevEdge() const { return kSdevEdge; }
	void set_kSdevEdge(double kSdevEdge_);

	const double& get_kSdevBg() const { return kSdevBg; }
	void set_kSdevBg(double kSdevBg_);

	const double& get_kContrast() const { return kContrast; }
	void set_kContrast(double kContrast_);

	const double& get_kCosAngleMCs() const { return kCosAngleMCs; }
	void set_kCosAngleMCs(double kCosAngleMCs_);

	const double& get_kMCsOffset() const { return kMCsOffset; }
	void set_kMCsOffset(double kMCsOffset_);

	const double& get_kGlyphWeight() const { return kGlyphWeight; }
	void set_kGlyphWeight(double kGlyphWeight_);

	bool parseCfg(const boost::filesystem::path &cfgFile); // Loads the settings provided in cfgFile
	void loadUserDefaults(); // Overwrites current settings with those from initMatchSettings.cfg
	void saveUserDefaults() const; // Overwrites initMatchSettings.cfg with current settings

	friend std::ostream& operator<<(std::ostream &os, const MatchSettings &c);
#ifdef UNIT_TESTING
	// Constructor available only within UnitTesting project
	MatchSettings(
		   double kSdevFg_ = 0., double kSdevEdge_ = 0., double kSdevBg_ = 0.,
		   double kContrast_ = 0., double kMCsOffset_ = 0., double kCosAngleMCs_ = 0.,
		   double kGlyphWeight_ = 0., unsigned threshold4Blank_ = 0U);
#endif
};

BOOST_CLASS_VERSION(MatchSettings, 0)

#endif
