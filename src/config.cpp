/**********************************************************
 Project:     Pic2Sym
 File:        config.cpp

 Author:      Florin Tulba
 Created on:  2015-12-20
 
 Copyright (c) 2016 Florin Tulba
 **********************************************************/

#include "config.h"
#include "misc.h"
#include "controller.h"

#include <fstream>
#include <ctime>

#include <boost/filesystem/operations.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/info_parser.hpp>

using namespace std;
using namespace boost::filesystem;
using namespace boost::property_tree;
using namespace boost::archive;

MatchSettings::MatchSettings(const string &appLaunchPath) {
	boost::filesystem::path executablePath(absolute(appLaunchPath));
	defCfgPath = cfgPath = workDir = executablePath.remove_filename();
	defCfgPath = defCfgPath.append("res").append("defaultMatchSettings.txt");
	cfgPath = cfgPath.append("initMatchSettings.cfg");
	
	if(exists(cfgPath))
		try {
			initialized = false;
			loadUserDefaults();
			initialized = true;
			return;
		} catch(invalid_argument&) {
			// Renaming the obsolete file
			rename(cfgPath, boost::filesystem::path(cfgPath)
				   .concat(".").concat(to_string(time(nullptr))).concat(".bak"));
		}

	// Create a fresh 'initMatchSettings.cfg' with data from 'res/defaultMatchSettings.txt'
	createUserDefaults();
	initialized = true;
}

void MatchSettings::createUserDefaults() {
	if(!exists(defCfgPath)) {
		cerr<<"There's no "<<cfgPath<<", neither "<<defCfgPath<<endl;
		throw runtime_error("There's no initMatchSettings.cfg, neither res/defaultMatchSettings.txt");
	}

	if(!parseCfg(defCfgPath))
		throw runtime_error("Invalid Configuration!");

	saveUserDefaults();
}

void MatchSettings::loadUserDefaults() {
	ifstream ifs(cfgPath.string(), ios::binary);
	binary_iarchive ia(ifs);
	ia>>*this; // throws invalid_argument for obsolete 'initMatchSettings.cfg'
}

void MatchSettings::saveUserDefaults() const {
	ofstream ofs(cfgPath.string(), ios::binary);
	binary_oarchive oa(ofs);
	oa<<*this;
}

bool MatchSettings::parseCfg(const boost::filesystem::path &cfgFile) {
	bool correct = false;
	double new_kSsim = 0.,
		new_kSdevFg = 0., new_kSdevEdge = 0., new_kSdevBg = 0.,
		new_kMCsOffset = 0., new_kCosAngleMCs = 0.,
		new_kContrast = 0., new_kGlyphWeight = 0.;
	unsigned newThreshold4Blank = 0U;
	string prop;
	try {
		ptree theCfg;
		read_info(cfgFile.string(), theCfg);

		new_kSsim			= theCfg.get<double>(prop = "STRUCTURAL_SIMILARITY");
		new_kSdevFg			= theCfg.get<double>(prop = "UNDER_GLYPH_CORRECTNESS");
		new_kSdevEdge		= theCfg.get<double>(prop = "GLYPH_EDGE_CORRECTNESS");
		new_kSdevBg			= theCfg.get<double>(prop = "ASIDE_GLYPH_CORRECTNESS");
		new_kContrast		= theCfg.get<double>(prop = "MORE_CONTRAST_PREF");
		new_kMCsOffset		= theCfg.get<double>(prop = "GRAVITATIONAL_SMOOTHNESS");
		new_kCosAngleMCs	= theCfg.get<double>(prop = "DIRECTIONAL_SMOOTHNESS");
		new_kGlyphWeight	= theCfg.get<double>(prop = "LARGER_GLYPHS_PREF");
		newThreshold4Blank	= theCfg.get<unsigned>(prop = "THRESHOLD_FOR_BLANK");

		if(!Settings::isBlanksThresholdOk(newThreshold4Blank) ||
		   new_kSsim < 0. || new_kSdevFg < 0. || new_kSdevEdge < 0. || new_kSdevBg < 0. ||
		   new_kContrast < 0. || new_kMCsOffset < 0. || new_kCosAngleMCs < 0. ||
		   new_kGlyphWeight < 0.)
			cerr<<"One or more properties in the configuration file are out of their range!"<<endl;
		else {
			correct = true;
			kSsim = new_kSsim;
			kSdevFg = new_kSdevFg;  kSdevEdge = new_kSdevEdge; kSdevBg = new_kSdevBg;
			kContrast = new_kContrast; 
			kMCsOffset = new_kMCsOffset; kCosAngleMCs = new_kCosAngleMCs;
			kGlyphWeight = new_kGlyphWeight; threshold4Blank = newThreshold4Blank;

			cout<<"Initial config values:"<<endl<<*this<<endl;
		}

	} catch(info_parser_error&) {
		cerr<<"Couldn't read "<<cfgFile<<endl;
	} catch(ptree_bad_path&) {
		cerr<<"Property '"<<prop<<"' is missing from configuration file!"<<endl;
	} catch(ptree_bad_data&) {
		cerr<<"Property '"<<prop<<"' cannot be converted to its required type!"<<endl;
	}

	return correct;
}

void MatchSettings::set_kSsim(double kSsim_) {
	if(kSsim == kSsim_)
		return;
	cout<<"kSsim"<<" : "<<kSsim<<" -> "<<kSsim_<<endl;
	kSsim = kSsim_;
}

void MatchSettings::set_kSdevFg(double kSdevFg_) {
	if(kSdevFg == kSdevFg_)
		return;
	cout<<"kSdevFg"<<" : "<<kSdevFg<<" -> "<<kSdevFg_<<endl;
	kSdevFg = kSdevFg_;
}

void MatchSettings::set_kSdevEdge(double kSdevEdge_) {
	if(kSdevEdge == kSdevEdge_)
		return;
	cout<<"kSdevEdge"<<" : "<<kSdevEdge<<" -> "<<kSdevEdge_<<endl;
	kSdevEdge = kSdevEdge_;
}

void MatchSettings::set_kSdevBg(double kSdevBg_) {
	if(kSdevBg == kSdevBg_)
		return;
	cout<<"kSdevBg"<<" : "<<kSdevBg<<" -> "<<kSdevBg_<<endl;
	kSdevBg = kSdevBg_;
}

void MatchSettings::set_kContrast(double kContrast_) {
	if(kContrast == kContrast_)
		return;
	cout<<"kContrast"<<" : "<<kContrast<<" -> "<<kContrast_<<endl;
	kContrast = kContrast_;
}

void MatchSettings::set_kCosAngleMCs(double kCosAngleMCs_) {
	if(kCosAngleMCs == kCosAngleMCs_)
		return;
	cout<<"kCosAngleMCs"<<" : "<<kCosAngleMCs<<" -> "<<kCosAngleMCs_<<endl;
	kCosAngleMCs = kCosAngleMCs_;
}

void MatchSettings::set_kMCsOffset(double kMCsOffset_) {
	if(kMCsOffset == kMCsOffset_)
		return;
	cout<<"kMCsOffset"<<" : "<<kMCsOffset<<" -> "<<kMCsOffset_<<endl;
	kMCsOffset = kMCsOffset_;
}

void MatchSettings::set_kGlyphWeight(double kGlyphWeight_) {
	if(kGlyphWeight == kGlyphWeight_)
		return;
	cout<<"kGlyphWeight"<<" : "<<kGlyphWeight<<" -> "<<kGlyphWeight_<<endl;
	kGlyphWeight = kGlyphWeight_;
}

void MatchSettings::setBlankThreshold(unsigned threshold4Blank_) {
	if(threshold4Blank == threshold4Blank_)
		return;
	cout<<"threshold4Blank"<<" : "<<threshold4Blank<<" -> "<<threshold4Blank_<<endl;
	threshold4Blank = threshold4Blank_;
}

ostream& operator<<(ostream &os, const MatchSettings &c) {
	os<<"kSsim"<<" : "<<c.kSsim<<endl;
	os<<"kSdevFg"<<" : "<<c.kSdevFg<<endl;
	os<<"kSdevEdge"<<" : "<<c.kSdevEdge<<endl;
	os<<"kSdevBg"<<" : "<<c.kSdevBg<<endl;
	os<<"kContrast"<<" : "<<c.kContrast<<endl;
	os<<"kMCsOffset"<<" : "<<c.kMCsOffset<<endl;
	os<<"kCosAngleMCs"<<" : "<<c.kCosAngleMCs<<endl;
	os<<"kGlyphWeight"<<" : "<<c.kGlyphWeight<<endl;
	os<<"threshold4Blank"<<" : "<<c.threshold4Blank<<endl;
	return os;
}