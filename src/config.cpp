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

	cout<<"Initial config values:"<<endl<<*this<<endl;
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

void Config::setFontSz(unsigned fontSz_) {
	cout<<"fontSz"<<" : "<<fontSz<<" -> "<<fontSz_<<endl;
	fontSz = fontSz_;
}

void Config::setMaxHSyms(unsigned syms) {
	cout<<"hMaxSyms"<<" : "<<hMaxSyms<<" -> "<<syms<<endl;
	hMaxSyms = syms;
}

void Config::setMaxVSyms(unsigned syms) {
	cout<<"vMaxSyms"<<" : "<<vMaxSyms<<" -> "<<syms<<endl;
	vMaxSyms = syms;
}

void Config::setBlankThreshold(unsigned threshold4Blank_) {
	cout<<"threshold4Blank"<<" : "<<threshold4Blank<<" -> "<<threshold4Blank_<<endl;
	threshold4Blank = threshold4Blank_;
}

void Config::set_kSdevFg(double kSdevFg_) {
	cout<<"kSdevFg"<<" : "<<kSdevFg<<" -> "<<kSdevFg_<<endl;
	kSdevFg = kSdevFg_;
}

void Config::set_kSdevEdge(double kSdevEdge_) {
	cout<<"kSdevEdge"<<" : "<<kSdevEdge<<" -> "<<kSdevEdge_<<endl;
	kSdevEdge = kSdevEdge_;
}

void Config::set_kSdevBg(double kSdevBg_) {
	cout<<"kSdevBg"<<" : "<<kSdevBg<<" -> "<<kSdevBg_<<endl;
	kSdevBg = kSdevBg_;
}

void Config::set_kContrast(double kContrast_) {
	cout<<"kContrast"<<" : "<<kContrast<<" -> "<<kContrast_<<endl;
	kContrast = kContrast_;
}

void Config::set_kCosAngleMCs(double kCosAngleMCs_) {
	cout<<"kCosAngleMCs"<<" : "<<kCosAngleMCs<<" -> "<<kCosAngleMCs_<<endl;
	kCosAngleMCs = kCosAngleMCs_;
}

void Config::set_kMCsOffset(double kMCsOffset_) {
	cout<<"kMCsOffset"<<" : "<<kMCsOffset<<" -> "<<kMCsOffset_<<endl;
	kMCsOffset = kMCsOffset_;
}

void Config::set_kGlyphWeight(double kGlyphWeight_) {
	cout<<"kGlyphWeight"<<" : "<<kGlyphWeight<<" -> "<<kGlyphWeight_<<endl;
	kGlyphWeight = kGlyphWeight_;
}

ostream& operator<<(ostream &os, const Config &c) {
	os<<"fontSz"<<" : "<<c.fontSz<<endl;
	os<<"kSdevFg"<<" : "<<c.kSdevFg<<endl;
	os<<"kSdevEdge"<<" : "<<c.kSdevEdge<<endl;
	os<<"kSdevBg"<<" : "<<c.kSdevBg<<endl;
	os<<"kContrast"<<" : "<<c.kContrast<<endl;
	os<<"kMCsOffset"<<" : "<<c.kMCsOffset<<endl;
	os<<"kCosAngleMCs"<<" : "<<c.kCosAngleMCs<<endl;
	os<<"kGlyphWeight"<<" : "<<c.kGlyphWeight<<endl;
	os<<"threshold4Blank"<<" : "<<c.threshold4Blank<<endl;
	os<<"hMaxSyms"<<" : "<<c.hMaxSyms<<endl;
	os<<"vMaxSyms"<<" : "<<c.vMaxSyms<<endl;
	return os;
}