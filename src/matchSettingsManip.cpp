/************************************************************************************************
 The application Pic2Sym approximates images by a
 grid of colored symbols with colored backgrounds.

 Copyrights from the libraries used by the program:
 - (c) 2016 Boost (www.boost.org)
		License: <http://www.boost.org/LICENSE_1_0.txt>
			or doc/licenses/Boost.lic
 - (c) 2015 OpenCV (www.opencv.org)
		License: <http://opencv.org/license.html>
            or doc/licenses/OpenCV.lic
 - (c) 2015 The FreeType Project (www.freetype.org)
		License: <http://git.savannah.gnu.org/cgit/freetype/freetype2.git/plain/docs/FTL.TXT>
	        or doc/licenses/FTL.txt
 - (c) 1997-2002 OpenMP Architecture Review Board (www.openmp.org)
   (c) Microsoft Corporation (Visual C++ implementation for OpenMP C/C++ Version 2.0 March 2002)
		See: <https://msdn.microsoft.com/en-us/library/8y6825x5(v=vs.140).aspx>
 - (c) 1995-2013 zlib software (Jean-loup Gailly and Mark Adler - see: www.zlib.net)
		License: <http://www.zlib.net/zlib_license.html>
            or doc/licenses/zlib.lic
 
 (c) 2016, 2017 Florin Tulba <florintulba@yahoo.com>

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
 ***********************************************************************************************/

#ifndef UNIT_TESTING

#include "matchSettingsManip.h"
#include "appStart.h"
#include "settings.h"
#include "propsReader.h"
#include "misc.h"

#pragma warning ( push, 0 )

#include <fstream>
#include <iostream>

#include "boost_filesystem_operations.h"

#ifndef AI_REVIEWER_CHECK
#include <boost/scope_exit.hpp>
#endif // AI_REVIEWER_CHECK

#pragma warning ( pop )

using namespace std;
using namespace boost::filesystem;

#ifndef AI_REVIEWER_CHECK
using namespace boost::archive;
#endif // AI_REVIEWER_CHECK

MatchSettingsManip& MatchSettingsManip::instance() {
#pragma warning ( disable : WARN_THREAD_UNSAFE )
	static MatchSettingsManip inst;
#pragma warning ( default : WARN_THREAD_UNSAFE )

	return inst;
}

void MatchSettingsManip::initMatchSettings(MatchSettings &ms) {
#ifndef AI_REVIEWER_CHECK
	// Ensure ms.initialized is true when leaving the method
#pragma warning ( disable : WARN_CANNOT_GENERATE_ASSIGN_OP )
	BOOST_SCOPE_EXIT(&ms) {
		if(!ms.initialized)
			ms.initialized = true;
	} BOOST_SCOPE_EXIT_END;
#pragma warning ( default : WARN_CANNOT_GENERATE_ASSIGN_OP )
#endif // AI_REVIEWER_CHECK

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

MatchSettingsManip::MatchSettingsManip() {
	defCfgPath = cfgPath = AppStart::dir();

	if(!exists(defCfgPath.append("res").append("defaultMatchSettings.txt")))
		THROW_WITH_VAR_MSG(__FUNCTION__" : There's no " + defCfgPath.string(), runtime_error);
}

void MatchSettingsManip::createUserDefaults(MatchSettings &ms) {
	if(!parseCfg(ms, defCfgPath))
		THROW_WITH_CONST_MSG(__FUNCTION__" : Invalid Configuration!", runtime_error);

	saveUserDefaults(ms);
}

void MatchSettingsManip::loadUserDefaults(MatchSettings &ms) {
	ifstream ifs(cfgPath.string(), ios::binary);
#ifndef AI_REVIEWER_CHECK
	binary_iarchive ia(ifs);
	ia>>ms; // when ms.initialized==false, throws invalid_argument for obsolete 'initMatchSettings.cfg'
#endif // AI_REVIEWER_CHECK
}

void MatchSettingsManip::saveUserDefaults(const MatchSettings &ms) const {
	ofstream ofs(cfgPath.string(), ios::binary);
#ifndef AI_REVIEWER_CHECK
	binary_oarchive oa(ofs);
	oa<<ms;
#endif // AI_REVIEWER_CHECK
}

bool MatchSettingsManip::parseCfg(MatchSettings &ms, const boost::filesystem::path &cfgFile) {
#pragma warning ( disable : WARN_THREAD_UNSAFE )
	static PropsReader parser(cfgFile);
#pragma warning ( default : WARN_THREAD_UNSAFE )

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

#endif // UNIT_TESTING not defined