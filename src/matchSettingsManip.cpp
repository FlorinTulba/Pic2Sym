/****************************************************************************************
 The application Pic2Sym approximates images by a
 grid of colored symbols with colored backgrounds.

 This file was created on 2016-4-8
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

#ifndef UNIT_TESTING

#include "matchSettingsManip.h"
#include "settings.h"
#include "propsReader.h"

#include <fstream>
#include <iostream>

#include <boost/filesystem/operations.hpp>
#include <boost/scope_exit.hpp>

using namespace std;
using namespace boost::filesystem;
using namespace boost::archive;

MatchSettingsManip* MatchSettingsManip::inst = nullptr;

void MatchSettingsManip::init(const string &appLaunchPath) {
	if(nullptr != inst)
		throw logic_error(__FUNCTION__ " should be called only once!");

	static MatchSettingsManip theInstance(appLaunchPath);
	inst = &theInstance;
}

MatchSettingsManip& MatchSettingsManip::instance() {
	if(nullptr != inst)
		return *inst;

	throw logic_error(__FUNCTION__ " called before MatchSettingsManip::init()!");
}

void MatchSettingsManip::initMatchSettings(MatchSettings &ms) {
	// Ensure ms.initialized is true when leaving the method
	BOOST_SCOPE_EXIT(&ms) {
		if(!ms.initialized)
			ms.initialized = true;
	} BOOST_SCOPE_EXIT_END;

	if(exists(cfgPath.append("initMatchSettings.cfg"))) {
		if(last_write_time(cfgPath) > last_write_time(defCfgPath)) { // newer
			try {
				loadUserDefaults(ms); // throws invalid_argument for older versions

				return;

			} catch(invalid_argument&) {} // newer, but still obsolete due to its version
		}

		// Renaming the obsolete file
		rename(cfgPath, boost::filesystem::path(cfgPath)
			   .concat(".").concat(to_string(time(nullptr))).concat(".bak"));
	}

	// Create a fresh 'initMatchSettings.cfg' with data from 'res/defaultMatchSettings.txt'
	createUserDefaults(ms);
}

MatchSettingsManip::MatchSettingsManip(const string &appLaunchPath) {
	boost::filesystem::path executablePath(absolute(appLaunchPath));

	defCfgPath = cfgPath = workDir = executablePath.remove_filename();

	if(!exists(defCfgPath.append("res").append("defaultMatchSettings.txt"))) {
		cerr<<"There's no "<<defCfgPath<<endl;
		throw runtime_error("There's no res/defaultMatchSettings.txt");
	}
}

void MatchSettingsManip::createUserDefaults(MatchSettings &ms) {
	if(!parseCfg(ms, defCfgPath))
		throw runtime_error("Invalid Configuration!");

	saveUserDefaults(ms);
}

void MatchSettingsManip::loadUserDefaults(MatchSettings &ms) {
	ifstream ifs(cfgPath.string(), ios::binary);
	binary_iarchive ia(ifs);
	ia>>ms; // when ms.initialized==false, throws invalid_argument for obsolete 'initMatchSettings.cfg'
}

void MatchSettingsManip::saveUserDefaults(const MatchSettings &ms) const {
	ofstream ofs(cfgPath.string(), ios::binary);
	binary_oarchive oa(ofs);
	oa<<ms;
}

bool MatchSettingsManip::parseCfg(MatchSettings &ms, const boost::filesystem::path &cfgFile) {
	static PropsReader parser(cfgFile);

	bool correct = false;
	const bool newResultMode = parser.read<bool>("HYBRID_RESULT");
	const double new_kSsim = parser.read<double>("STRUCTURAL_SIMILARITY"),
				new_kSdevFg = parser.read<double>("UNDER_SYM_CORRECTNESS"),
				new_kSdevEdge = parser.read<double>("SYM_EDGE_CORRECTNESS"),
				new_kSdevBg = parser.read<double>("ASIDE_SYM_CORRECTNESS"),
				new_kMCsOffset = parser.read<double>("MORE_CONTRAST_PREF"), 
				new_kCosAngleMCs = parser.read<double>("GRAVITATIONAL_SMOOTHNESS"),
				new_kContrast = parser.read<double>("DIRECTIONAL_SMOOTHNESS"),
				new_kSymDensity = parser.read<double>("LARGER_SYM_PREF");
	const unsigned newThreshold4Blank = parser.read<unsigned>("THRESHOLD_FOR_BLANK");

	if(!Settings::isBlanksThresholdOk(newThreshold4Blank) ||
	   new_kSsim < 0. || new_kSdevFg < 0. || new_kSdevEdge < 0. || new_kSdevBg < 0. ||
	   new_kContrast < 0. || new_kMCsOffset < 0. || new_kCosAngleMCs < 0. ||
	   new_kSymDensity < 0.)
	   cerr<<"One or more properties in the configuration file are out of their range!"<<endl;
	else {
		correct = true;
		ms.setResultMode(newResultMode);
		ms.set_kSsim(new_kSsim);
		ms.set_kSdevFg(new_kSdevFg);
		ms.set_kSdevEdge(new_kSdevEdge);
		ms.set_kSdevBg(new_kSdevBg);
		ms.set_kContrast(new_kContrast);
		ms.set_kMCsOffset(new_kMCsOffset);
		ms.set_kCosAngleMCs(new_kCosAngleMCs);
		ms.set_kSymDensity(new_kSymDensity);
		ms.setBlankThreshold(newThreshold4Blank);

		cout<<"Initial config values:"<<endl<<ms<<endl;
	}

	return correct;
}

#endif