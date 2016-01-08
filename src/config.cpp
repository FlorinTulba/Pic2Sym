/**********************************************************
 Project:     Pic2Sym
 File:        config.cpp

 Author:      Florin Tulba
 Created on:  2015-12-20
 
 Copyright (c) 2016 Florin Tulba
 **********************************************************/

#include "config.h"
#include "misc.h"

#include <fstream>
#include <sstream>
#include <set>

#include <boost/filesystem.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/info_parser.hpp>

using namespace std;
using namespace boost::filesystem;
using namespace boost::property_tree;

Config::Config(const string &appLaunchPath) {
	boost::filesystem::path executablePath(absolute(appLaunchPath));
	current_path(executablePath.parent_path()); // Set as working directory the one where the currently launched exe resides
	curDir = current_path().string();
	ifstream ifs("res/defaultCfg.txt");
	if(!ifs) {
		cerr<<"There's no defaultCfg.txt"<<endl;
		throw runtime_error("There's no defaultCfg.txt");
	}
	ofstream ofs("cfg.txt");
	ofs<<ifs.rdbuf();
	ofs.close();
	ifs.close();

	if(!parseCfg())
		throw invalid_argument("Invalid Default Config!");
}

bool Config::parseCfg() {
	bool correct = false;
	unsigned newFontSz = 0U, newOutW = 0U, newOutH = 0U, newThreshold4Blank = 0U;
	string prop;
	try {
		ptree theCfg;
		read_info("cfg.txt", theCfg);

		newFontSz			= theCfg.get<unsigned>(prop = "FONT_HEIGHT");
		newOutW				= theCfg.get<unsigned>(prop = "RESULT_WIDTH");
		newOutH				= theCfg.get<unsigned>(prop = "RESULT_HEIGHT");
		newThreshold4Blank	= theCfg.get<unsigned>(prop = "THRESHOLD_FOR_BLANK");

		if(newFontSz < 7U || newFontSz > 50U ||
		   newOutW < 3U || newOutW > 1024 ||
		   newOutH < 3U || newOutH > 768 ||
		   newThreshold4Blank > 50U)
			cerr<<"One or more properties in the configuration file are out of their range!"<<endl;
		else {
			correct = true;
			fontSz = newFontSz; threshold4Blank = newThreshold4Blank;
			outW = newOutW; outH = newOutH;
		}

	} catch(info_parser_error&) {
		cerr<<"Couldn't read cfg.txt"<<endl;
	} catch(ptree_bad_path&) {
		cerr<<"Property '"<<prop<<"' is missing from configuration file!"<<endl;
	} catch(ptree_bad_data&) {
		cerr<<"Property '"<<prop<<"' cannot be converted to its required type!"<<endl;
	}

	return correct;
}

void Config::update() {
	current_path(curDir);
	ifstream ifs("cfg.txt");
	ostringstream oss;
	oss<<endl<<"Current configuration is:"<<endl<<endl
		<<"FONT_HEIGHT = "<<fontSz<<endl
		<<"RESULT_WIDTH = "<<outW<<endl
		<<"RESULT_HEIGHT = "<<outH<<endl
		<<"THRESHOLD_FOR_BLANK = "<<threshold4Blank<<endl
		<<endl;
	oss<<"Keep these settings?";
	if(!boolPrompt(oss.str())) {
		unsigned oldFontSz = fontSz, oldOutW = outW, oldOutH = outH,
			oldThreshold4Blank = threshold4Blank;

		for(;;) {
			system("notepad.exe cfg.txt");
			if(parseCfg())
				break;
			cerr<<"Problems within cfg.txt, please correct them!"<<endl;
		}

		if(oldFontSz != fontSz)
			cout<<"New FONT_HEIGHT is "<<fontSz<<endl;
		if(oldOutW != outW)
			cout<<"New RESULT_WIDTH is "<<outW<<endl;
		if(oldOutH != outH)
			cout<<"New RESULT_HEIGHT is "<<outH<<endl;
		if(oldThreshold4Blank != threshold4Blank)
			cout<<"New THRESHOLD_FOR_BLANK is "<<threshold4Blank<<endl;
	}
}
