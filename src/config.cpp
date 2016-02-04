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

#include <boost/filesystem/operations.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/info_parser.hpp>

using namespace std;
using namespace boost::filesystem;
using namespace boost::property_tree;

Config::Config(const string &appLaunchPath) {
	boost::filesystem::path executablePath(absolute(appLaunchPath));
	cfgPath = workDir =
		executablePath.remove_filename();
	cfgPath = cfgPath.append("res").append("defaultCfg.txt");

	if(!exists(cfgPath)) {
		cerr<<"There's no "<<cfgPath<<endl;
		throw runtime_error("There's no defaultCfg.txt");
	}

	if(!parseCfg())
		throw invalid_argument("Invalid Default Config!");
}

bool Config::parseCfg() {
	bool correct = false;
	unsigned newFontSz = 0U, newHmaxSyms = 0U, newVmaxSyms = 0U, newThreshold4Blank = 0U;
	double new_kSdevFg = 0., new_kSdevEdge = 0., new_kSdevBg = 0.,
		new_kMCsOffset = 0., new_kCosAngleMCs = 0.,
		new_kContrast = 0., new_kGlyphWeight = 0.;
	string prop;
	try {
		ptree theCfg;
		read_info(cfgPath.string(), theCfg);

		newFontSz			= theCfg.get<unsigned>(prop = "FONT_HEIGHT");
		new_kSdevFg			= theCfg.get<double>(prop = "UNDER_GLYPH_CORRECTNESS");
		new_kSdevEdge		= theCfg.get<double>(prop = "GLYPH_EDGE_CORRECTNESS");
		new_kSdevBg			= theCfg.get<double>(prop = "ASIDE_GLYPH_CORRECTNESS");
		new_kContrast		= theCfg.get<double>(prop = "MORE_CONTRAST_PREF");
		new_kMCsOffset		= theCfg.get<double>(prop = "GRAVITATIONAL_SMOOTHNESS");
		new_kCosAngleMCs	= theCfg.get<double>(prop = "DIRECTIONAL_SMOOTHNESS");
		new_kGlyphWeight	= theCfg.get<double>(prop = "LARGER_GLYPHS_PREF");
		newThreshold4Blank	= theCfg.get<unsigned>(prop = "THRESHOLD_FOR_BLANK");
		newHmaxSyms			= theCfg.get<unsigned>(prop = "RESULT_WIDTH");
		newVmaxSyms			= theCfg.get<unsigned>(prop = "RESULT_HEIGHT");

		if(!isFontSizeOk(newFontSz) ||
		   !isHmaxSymsOk(newHmaxSyms) ||
		   !isVmaxSymsOk(newVmaxSyms) ||
		   !isBlanksThresholdOk(newThreshold4Blank) ||
		   new_kSdevFg < 0. || new_kSdevEdge < 0. || new_kSdevBg < 0. || new_kContrast < 0. ||
		   new_kMCsOffset < 0. || new_kCosAngleMCs < 0. ||
		   new_kGlyphWeight < 0.)
			cerr<<"One or more properties in the configuration file are out of their range!"<<endl;
		else {
			correct = true;
			fontSz = newFontSz;
			kSdevFg = new_kSdevFg;  kSdevEdge = new_kSdevEdge; kSdevBg = new_kSdevBg;
			kContrast = new_kContrast; 
			kMCsOffset = new_kMCsOffset; kCosAngleMCs = new_kCosAngleMCs;
			kGlyphWeight = new_kGlyphWeight; threshold4Blank = newThreshold4Blank;
			hMaxSyms = newHmaxSyms; vMaxSyms = newVmaxSyms;
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
