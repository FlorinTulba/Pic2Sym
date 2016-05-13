/****************************************************************************************
 The application Pic2Sym approximates images by a
 grid of colored symbols with colored backgrounds.

 This file was created on 2016-1-8
 and belongs to the Pic2Sym project.

 Copyrights from the libraries used by the program:
 - (c) 2015 Boost (www.boost.org)
   License: <http://www.boost.org/LICENSE_1_0.txt>
            or doc/licenses/Boost.lic
 - (c) 2015 The FreeType Project (www.freetype.org)
   License: <http://git.savannah.gnu.org/cgit/freetype/freetype2.git/plain/docs/FTL.TXT>
	        or doc/licenses/FTL.txt
 - (c) 2015 OpenCV (www.opencv.org)
   License: <http://opencv.org/license.html>
            or doc/licenses/OpenCV.lic
 
 (c) 2016 Florin Tulba <florintulba@yahoo.com>

 This program is free software: you can use its results,
 redistribute it and/or modify it under the terms of the GNU
 Affero General Public License version 3 as published by the
 Free Software Foundation.

 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 GNU Affero General Public License for more details.

 You should have received a copy of the GNU Affero General Public License
 along with this program ('agpl-3.0.txt').
 If not, see <http://www.gnu.org/licenses/agpl-3.0.txt>.
 ****************************************************************************************/

#include "controller.h"
#include "misc.h"

#include <fstream>
#include <ctime>

#include <boost/filesystem/operations.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/info_parser.hpp>
#include <boost/scope_exit.hpp>

using namespace std;
using namespace boost::filesystem;
using namespace boost::property_tree;
using namespace boost::archive;

MatchSettings::MatchSettings(const string &appLaunchPath) {
	boost::filesystem::path executablePath(absolute(appLaunchPath));

	defCfgPath = cfgPath = workDir = executablePath.remove_filename();
	
	if(!exists(defCfgPath.append("res").append("defaultMatchSettings.txt"))) {
		cerr<<"There's no "<<defCfgPath<<endl;
		throw runtime_error("There's no res/defaultMatchSettings.txt");
	}

	// Ensure initialized is true when leaving the constructor
	BOOST_SCOPE_EXIT(this_) {
		if(!this_->initialized)
			this_->initialized = true;
	} BOOST_SCOPE_EXIT_END;
	
	if(exists(cfgPath.append("initMatchSettings.cfg"))) {
		if(last_write_time(cfgPath) > last_write_time(defCfgPath)) { // newer
			try {
				initialized = false;
				loadUserDefaults(); // throws invalid_argument for older versions
				
				return;

			} catch(invalid_argument&) {} // newer, but still obsolete due to its version
		}

		// Renaming the obsolete file
		rename(cfgPath, boost::filesystem::path(cfgPath)
			   .concat(".").concat(to_string(time(nullptr))).concat(".bak"));
	}

	// Create a fresh 'initMatchSettings.cfg' with data from 'res/defaultMatchSettings.txt'
	createUserDefaults();
}

void MatchSettings::createUserDefaults() {
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
		new_kContrast = 0., new_kSymDensity = 0.;
	unsigned newThreshold4Blank = 0U;
	string prop;
	try {
		ptree theCfg;
		read_info(cfgFile.string(), theCfg);

		new_kSsim			= theCfg.get<double>(prop = "STRUCTURAL_SIMILARITY");
		new_kSdevFg			= theCfg.get<double>(prop = "UNDER_SYM_CORRECTNESS");
		new_kSdevEdge		= theCfg.get<double>(prop = "SYM_EDGE_CORRECTNESS");
		new_kSdevBg			= theCfg.get<double>(prop = "ASIDE_SYM_CORRECTNESS");
		new_kContrast		= theCfg.get<double>(prop = "MORE_CONTRAST_PREF");
		new_kMCsOffset		= theCfg.get<double>(prop = "GRAVITATIONAL_SMOOTHNESS");
		new_kCosAngleMCs	= theCfg.get<double>(prop = "DIRECTIONAL_SMOOTHNESS");
		new_kSymDensity		= theCfg.get<double>(prop = "LARGER_SYM_PREF");
		newThreshold4Blank	= theCfg.get<unsigned>(prop = "THRESHOLD_FOR_BLANK");

		if(!Settings::isBlanksThresholdOk(newThreshold4Blank) ||
		   new_kSsim < 0. || new_kSdevFg < 0. || new_kSdevEdge < 0. || new_kSdevBg < 0. ||
		   new_kContrast < 0. || new_kMCsOffset < 0. || new_kCosAngleMCs < 0. ||
		   new_kSymDensity < 0.)
			cerr<<"One or more properties in the configuration file are out of their range!"<<endl;
		else {
			correct = true;
			kSsim = new_kSsim;
			kSdevFg = new_kSdevFg;  kSdevEdge = new_kSdevEdge; kSdevBg = new_kSdevBg;
			kContrast = new_kContrast; 
			kMCsOffset = new_kMCsOffset; kCosAngleMCs = new_kCosAngleMCs;
			kSymDensity = new_kSymDensity; threshold4Blank = newThreshold4Blank;

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

void MatchSettings::set_kSymDensity(double kSymDensity_) {
	if(kSymDensity == kSymDensity_)
		return;
	cout<<"kSymDensity"<<" : "<<kSymDensity<<" -> "<<kSymDensity_<<endl;
	kSymDensity = kSymDensity_;
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
	os<<"kSymDensity"<<" : "<<c.kSymDensity<<endl;
	os<<"threshold4Blank"<<" : "<<c.threshold4Blank<<endl;
	return os;
}