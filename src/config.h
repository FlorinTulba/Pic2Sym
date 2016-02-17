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
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/serialization/split_member.hpp>
#include <boost/serialization/version.hpp>

// MatchSettings class controls the matching parameters for transforming one or more images.
class MatchSettings {
public:
	static const unsigned VERSION = 0U;

private:
	boost::filesystem::path workDir;	// Folder where the application was launched
	boost::filesystem::path defCfgPath;	// Path of the original configuration file
	boost::filesystem::path cfgPath;	// Path of the user configuration file

	double kSdevFg = 0., kSdevEdge = 0., kSdevBg = 0.; // powers of factors for glyph correlation
	double kContrast = 0.;						// power of factor for the resulted glyph contrast
	double kMCsOffset = 0., kCosAngleMCs = 0.;	// powers of factors targeting smoothness
	double kGlyphWeight = 0.;		// power of factor aiming fanciness, not correctness
	unsigned threshold4Blank = 0U;	// Using Blank character replacement under this threshold

	/*
	MatchSettings is considered correctly initialized if its data is read from
	'res/defaultMatchSettings.txt'(most up-to-date file, which always exists) or
	'initMatchSettings.cfg'.

	First launch of the application will generate the second file from above and
	further launches will check directly for 'initMatchSettings.cfg'.

	Anytime MatchSettings::VERSION is increased, 'initMatchSettings.cfg' becomes
	obsolete, so it must be overwritten with the fresh data from 'res/defaultMatchSettings.txt'.
	*/
	bool initialized = false;		// set to true at the end of constructor

	template<class Archive>
	void load(Archive &ar, const unsigned version) {
		if(version < VERSION) {
			if(!initialized) // can happen only when loading an obsolete 'initMatchSettings.cfg'
				throw invalid_argument("Obsolete version of 'initMatchSettings.cfg'!");

			// Point reachable while reading Settings with an older version of MatchSettings field
		}

		// It is useful to see which settings changed when loading
		MatchSettings defSettings(*this); // create as copy of previous values

		// read user default match settings
		ar >> defSettings.kSdevFg >> defSettings.kSdevEdge >> defSettings.kSdevBg
			>> defSettings.kContrast
			>> defSettings.kMCsOffset >> defSettings.kCosAngleMCs
			>> defSettings.kGlyphWeight
			>> defSettings.threshold4Blank;

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
		ar << kSdevFg << kSdevEdge << kSdevBg
			<< kContrast
			<< kMCsOffset << kCosAngleMCs
			<< kGlyphWeight
			<< threshold4Blank;
	}
	BOOST_SERIALIZATION_SPLIT_MEMBER();
	friend class boost::serialization::access;

	// creates 'initMatchSettings.cfg' with data from 'res/defaultMatchSettings.txt'
	void createUserDefaults();

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

BOOST_CLASS_VERSION(MatchSettings, MatchSettings::VERSION);

#endif
