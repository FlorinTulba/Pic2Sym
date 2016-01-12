/**********************************************************
 Project:     Pic2Sym
 File:        config.cpp

 Author:      Florin Tulba
 Created on:  2015-12-20
 
 Copyright (c) 2016 Florin Tulba
 **********************************************************/

#include "config.h"
#include "misc.h"

#include <sstream>
#include <set>

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/info_parser.hpp>

using namespace std;
using namespace boost::filesystem;
using namespace boost::property_tree;

Config::Config(const string &appLaunchPath) {
	boost::filesystem::path origCfgPath,
		executablePath(absolute(appLaunchPath));
	cfgPath = origCfgPath = workDir =
		executablePath.remove_filename();
	cfgPath /= "cfg.txt";

	if(!exists(origCfgPath.append("res").append("defaultCfg.txt"))) {
		cerr<<"There's no "<<origCfgPath<<endl;
		throw runtime_error("There's no defaultCfg.txt");
	}

	// Keeping a local copy (cfg.txt) of the original configuration file (res/defaultCfg.txt)
	copy_file(origCfgPath, cfgPath, copy_option::overwrite_if_exists);

	if(!parseCfg())
		throw invalid_argument("Invalid Default Config!");
}

Config::~Config() {
	remove(cfgPath); // Deletes the local configuration file cfg.txt
}

bool Config::parseCfg() {
	bool correct = false;
	unsigned newFontSz = 0U, newOutW = 0U, newOutH = 0U, newThreshold4Blank = 0U;
	double new_kSdevFg = 0., new_kSdevBg = 0.,
		new_kCosAngleCogs = 0., new_kCogOffset = 0.,
		new_kContrast = 0., new_kGlyphWeight = 0.;
	string prop;
	try {
		ptree theCfg;
		read_info(cfgPath.string(), theCfg);

		newFontSz			= theCfg.get<unsigned>(prop = "FONT_HEIGHT");
		newOutW				= theCfg.get<unsigned>(prop = "RESULT_WIDTH");
		newOutH				= theCfg.get<unsigned>(prop = "RESULT_HEIGHT");
		newThreshold4Blank	= theCfg.get<unsigned>(prop = "THRESHOLD_FOR_BLANK");
		new_kContrast		= theCfg.get<double>(prop = "MORE_CONTRAST_PREF");
		new_kGlyphWeight	= theCfg.get<double>(prop = "LARGER_GLYPHS_PREF");
		new_kSdevFg			= theCfg.get<double>(prop = "UNDER_GLYPH_CORRECTNESS");
		new_kSdevBg			= theCfg.get<double>(prop = "ASIDE_GLYPH_CORRECTNESS");
		new_kCosAngleCogs	= theCfg.get<double>(prop = "DIRECTIONAL_SMOOTHNESS");
		new_kCogOffset		= theCfg.get<double>(prop = "GRAVITATIONAL_SMOOTHNESS");

		if(newFontSz < 7U || newFontSz > 50U ||
		   newOutW < 3U || newOutW > 1024 ||
		   newOutH < 3U || newOutH > 768 ||
		   newThreshold4Blank > 50U ||
		   new_kSdevFg < 0. || new_kSdevBg < 0. ||
		   new_kCosAngleCogs < 0. || new_kCogOffset < 0. ||
		   new_kContrast < 0. || new_kGlyphWeight < 0.)
			cerr<<"One or more properties in the configuration file are out of their range!"<<endl;
		else {
			correct = true;
			fontSz = newFontSz; threshold4Blank = newThreshold4Blank;
			outW = newOutW; outH = newOutH;
			kContrast = new_kContrast; kGlyphWeight = new_kGlyphWeight;
			kSdevFg = new_kSdevFg; kSdevBg = new_kSdevBg;
			kCosAngleCogs = new_kCosAngleCogs; kCogOffset = new_kCogOffset;
		}

	} catch(info_parser_error&) {
		cerr<<"Couldn't read "<<cfgPath<<endl;
	} catch(ptree_bad_path&) {
		cerr<<"Property '"<<prop<<"' is missing from configuration file!"<<endl;
	} catch(ptree_bad_data&) {
		cerr<<"Property '"<<prop<<"' cannot be converted to its required type!"<<endl;
	}

	return correct;
}

bool Config::update() {
	ostringstream oss;
	oss<<endl<<"Current configuration is:"<<endl<<endl
		<<"FONT_HEIGHT = "<<fontSz<<endl
		<<"RESULT_WIDTH = "<<outW<<endl
		<<"RESULT_HEIGHT = "<<outH<<endl
		<<"THRESHOLD_FOR_BLANK = "<<threshold4Blank<<endl
		<<"MORE_CONTRAST_PREF = "<<kContrast<<endl
		<<"LARGER_GLYPHS_PREF = "<<kGlyphWeight<<endl
		<<"UNDER_GLYPH_CORRECTNESS = "<<kSdevFg<<endl
		<<"ASIDE_GLYPH_CORRECTNESS = "<<kSdevBg<<endl
		<<"DIRECTIONAL_SMOOTHNESS = "<<kCosAngleCogs<<endl
		<<"GRAVITATIONAL_SMOOTHNESS = "<<kCogOffset<<endl
		<<endl;
	oss<<"Keep these settings?";
	bool someChanges = false;

	if(!boolPrompt(oss.str())) {
		unsigned oldFontSz = fontSz, oldOutW = outW, oldOutH = outH,
			oldThreshold4Blank = threshold4Blank;
		double old_kContrast = kContrast, old_kGlyphWeight = kGlyphWeight,
			old_kSdevFg = kSdevFg, old_kSdevBg = kSdevBg,
			old_kCosAngleCogs = kCosAngleCogs, old_kCogOffset = kCogOffset;

		oss.str(""); oss.clear(); oss<<"notepad.exe "<<cfgPath;
		const string openCfgWithNotepad(oss.str());
		for(;;) {
			system(openCfgWithNotepad.c_str());
			if(parseCfg())
				break;
			cerr<<"Problems within cfg.txt, please correct them!"<<endl;
		}

		if(oldFontSz != fontSz) {
			cout<<"New FONT_HEIGHT is "<<fontSz<<endl;
			someChanges = true;
		}
		if(oldOutW != outW) {
			cout<<"New RESULT_WIDTH is "<<outW<<endl;
			someChanges = true;
		}
		if(oldOutH != outH) {
			cout<<"New RESULT_HEIGHT is "<<outH<<endl;
			someChanges = true;
		}
		if(oldThreshold4Blank != threshold4Blank) {
			cout<<"New THRESHOLD_FOR_BLANK is "<<threshold4Blank<<endl;
			someChanges = true;
		}
		if(old_kContrast != kContrast) {
			cout<<"New MORE_CONTRAST_PREF is "<<kContrast<<endl;
			someChanges = true;
		}
		if(old_kGlyphWeight != kGlyphWeight) {
			cout<<"New LARGER_GLYPHS_PREF is "<<kGlyphWeight<<endl;
			someChanges = true;
		}
		if(old_kSdevFg != kSdevFg) {
			cout<<"New UNDER_GLYPH_CORRECTNESS is "<<kSdevFg<<endl;
			someChanges = true;
		}
		if(old_kSdevBg != kSdevBg) {
			cout<<"New ASIDE_GLYPH_CORRECTNESS is "<<kSdevBg<<endl;
			someChanges = true;
		}
		if(old_kCosAngleCogs != kCosAngleCogs) {
			cout<<"New DIRECTIONAL_SMOOTHNESS is "<<kCosAngleCogs<<endl;
			someChanges = true;
		}
		if(old_kCogOffset != kCogOffset) {
			cout<<"New GRAVITATIONAL_SMOOTHNESS is "<<kCogOffset<<endl;
			someChanges = true;
		}
	}

	return someChanges;
}
