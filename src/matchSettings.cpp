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
 
 (c) 2016-2019 Florin Tulba <florintulba@yahoo.com>

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

#include "matchSettings.h"

unsigned MatchSettings::VERSION_FROM_LAST_IO_OP = UINT_MAX;

#pragma warning ( push, 0 )

#include <iostream>

#pragma warning ( pop )

#ifndef UNIT_TESTING

#include "appStart.h"
#include "propsReader.h"
#include "settingsBase.h"

#include "boost_filesystem_operations.h"

#ifndef AI_REVIEWER_CHECK

#include <boost/scope_exit.hpp>

using namespace boost::archive;

#endif // AI_REVIEWER_CHECK

using namespace std;
using namespace boost::filesystem;

path MatchSettings::defCfgPath;
path MatchSettings::cfgPath;

void MatchSettings::configurePaths() {
	if(defCfgPath.empty()) {
		defCfgPath = cfgPath = AppStart::dir();

		if(!exists(defCfgPath.append("res").append("defaultMatchSettings.txt")))
			THROW_WITH_VAR_MSG(__FUNCTION__" : There's no " + defCfgPath.string(), runtime_error);

		cfgPath.append("initMatchSettings.cfg");
	}
}

void MatchSettings::replaceByUserDefaults() {
	ifstream ifs(cfgPath.string(), ios::binary);
#ifndef AI_REVIEWER_CHECK
	binary_iarchive ia(ifs);
	ia>>*this; // when initialized==false, throws invalid_argument for obsolete 'initMatchSettings.cfg'
#endif // AI_REVIEWER_CHECK
}

void MatchSettings::saveAsUserDefaults() const {
	ofstream ofs(cfgPath.string(), ios::binary);
#ifndef AI_REVIEWER_CHECK
	binary_oarchive oa(ofs);
	oa<<*this;
#endif // AI_REVIEWER_CHECK
}

void MatchSettings::createUserDefaults() {
	if(!parseCfg())
		THROW_WITH_CONST_MSG(__FUNCTION__" : Invalid Configuration!", runtime_error);

	saveAsUserDefaults();
}

bool MatchSettings::parseCfg() {
#pragma warning ( disable : WARN_THREAD_UNSAFE )
	static PropsReader parser(defCfgPath);
#pragma warning ( default : WARN_THREAD_UNSAFE )

	bool correct = false;
	const bool newResultMode = parser.read<bool>("HYBRID_RESULT");
	const double new_kSsim = parser.read<double>("STRUCTURAL_SIMILARITY"),
		new_kCorrel = parser.read<double>("CORRELATION_CORRECTNESS"),
		new_kSdevFg = parser.read<double>("UNDER_SYM_CORRECTNESS"),
		new_kSdevEdge = parser.read<double>("SYM_EDGE_CORRECTNESS"),
		new_kSdevBg = parser.read<double>("ASIDE_SYM_CORRECTNESS"),
		new_kMCsOffset = parser.read<double>("MORE_CONTRAST_PREF"),
		new_kCosAngleMCs = parser.read<double>("GRAVITATIONAL_SMOOTHNESS"),
		new_kContrast = parser.read<double>("DIRECTIONAL_SMOOTHNESS"),
		new_kSymDensity = parser.read<double>("LARGER_SYM_PREF");
	const unsigned newThreshold4Blank = parser.read<unsigned>("THRESHOLD_FOR_BLANK");

	if(!ISettings::isBlanksThresholdOk(newThreshold4Blank) ||
	   new_kSsim < 0. || new_kCorrel < 0. ||
	   new_kSdevFg < 0. || new_kSdevEdge < 0. || new_kSdevBg < 0. ||
	   new_kContrast < 0. || new_kMCsOffset < 0. || new_kCosAngleMCs < 0. ||
	   new_kSymDensity < 0.)
	   cerr<<"One or more properties in the configuration file are out of their range!"<<endl;
	else {
		correct = true;
		setResultMode(newResultMode);
		set_kSsim(new_kSsim);
		set_kCorrel(new_kCorrel);
		set_kSdevFg(new_kSdevFg);
		set_kSdevEdge(new_kSdevEdge);
		set_kSdevBg(new_kSdevBg);
		set_kContrast(new_kContrast);
		set_kMCsOffset(new_kMCsOffset);
		set_kCosAngleMCs(new_kCosAngleMCs);
		set_kSymDensity(new_kSymDensity);
		setBlankThreshold(newThreshold4Blank);

		cout<<"Initial config values:"<<endl<<*this<<endl;
	}

	return correct;
}

MatchSettings::MatchSettings() {
	configurePaths();

#ifndef AI_REVIEWER_CHECK
	// Ensure initialized is true when leaving the method
#pragma warning ( disable : WARN_CANNOT_GENERATE_ASSIGN_OP )
	BOOST_SCOPE_EXIT(&initialized) {
		if(!initialized)
			initialized = true;
	} BOOST_SCOPE_EXIT_END;
#pragma warning ( default : WARN_CANNOT_GENERATE_ASSIGN_OP )
#endif // AI_REVIEWER_CHECK

	if(exists(cfgPath)) {
		if(last_write_time(cfgPath) > last_write_time(defCfgPath)) { // newer
			try {
				replaceByUserDefaults(); // throws invalid files or older versions

				return;

			} catch(...) {} // invalid files or older versions
		}

		// Renaming the obsolete file
		rename(cfgPath, boost::filesystem::path(cfgPath)
			   .concat(".").concat(to_string(time(nullptr))).concat(".bak"));
	}

	// Create a fresh 'initMatchSettings.cfg' with data from 'res/defaultMatchSettings.txt'
	createUserDefaults();
}

#endif // UNIT_TESTING not defined

using namespace std;

MatchSettings& MatchSettings::setResultMode(bool hybridResultMode_) {
	if(hybridResultMode != hybridResultMode_) {
		cout<<"hybridResultMode"<<" : "<<hybridResultMode<<" -> "<<hybridResultMode_<<endl;
		hybridResultMode = hybridResultMode_;
	}
	return *this;
}

MatchSettings& MatchSettings::set_kSsim(double kSsim_) {
	if(kSsim != kSsim_) {
		cout<<"kSsim"<<" : "<<kSsim<<" -> "<<kSsim_<<endl;
		kSsim = kSsim_;
	}
	return *this;
}

MatchSettings& MatchSettings::set_kCorrel(double kCorrel_) {
	if(kCorrel != kCorrel_) {
		cout<<"kCorrel"<<" : "<<kCorrel<<" -> "<<kCorrel_<<endl;
		kCorrel = kCorrel_;
	}
	return *this;
}

MatchSettings& MatchSettings::set_kSdevFg(double kSdevFg_) {
	if(kSdevFg != kSdevFg_) {
		cout<<"kSdevFg"<<" : "<<kSdevFg<<" -> "<<kSdevFg_<<endl;
		kSdevFg = kSdevFg_;
	}
	return *this;
}

MatchSettings& MatchSettings::set_kSdevEdge(double kSdevEdge_) {
	if(kSdevEdge != kSdevEdge_) {
		cout<<"kSdevEdge"<<" : "<<kSdevEdge<<" -> "<<kSdevEdge_<<endl;
		kSdevEdge = kSdevEdge_;
	}
	return *this;
}

MatchSettings& MatchSettings::set_kSdevBg(double kSdevBg_) {
	if(kSdevBg != kSdevBg_) {
		cout<<"kSdevBg"<<" : "<<kSdevBg<<" -> "<<kSdevBg_<<endl;
		kSdevBg = kSdevBg_;
	}
	return *this;
}

MatchSettings& MatchSettings::set_kContrast(double kContrast_) {
	if(kContrast != kContrast_) {
		cout<<"kContrast"<<" : "<<kContrast<<" -> "<<kContrast_<<endl;
		kContrast = kContrast_;
	}
	return *this;
}

MatchSettings& MatchSettings::set_kCosAngleMCs(double kCosAngleMCs_) {
	if(kCosAngleMCs != kCosAngleMCs_) {
		cout<<"kCosAngleMCs"<<" : "<<kCosAngleMCs<<" -> "<<kCosAngleMCs_<<endl;
		kCosAngleMCs = kCosAngleMCs_;
	}
	return *this;
}

MatchSettings& MatchSettings::set_kMCsOffset(double kMCsOffset_) {
	if(kMCsOffset != kMCsOffset_) {
		cout<<"kMCsOffset"<<" : "<<kMCsOffset<<" -> "<<kMCsOffset_<<endl;
		kMCsOffset = kMCsOffset_;
	}
	return *this;
}

MatchSettings& MatchSettings::set_kSymDensity(double kSymDensity_) {
	if(kSymDensity != kSymDensity_) {
		cout<<"kSymDensity"<<" : "<<kSymDensity<<" -> "<<kSymDensity_<<endl;
		kSymDensity = kSymDensity_;
	}
	return *this;
}

MatchSettings& MatchSettings::setBlankThreshold(unsigned threshold4Blank_) {
	if(threshold4Blank != threshold4Blank_) {
		cout<<"threshold4Blank"<<" : "<<threshold4Blank<<" -> "<<threshold4Blank_<<endl;
		threshold4Blank = threshold4Blank_;
	}
	return *this;
}

uniquePtr<IMatchSettings> MatchSettings::clone() const {
	return makeUnique<MatchSettings>(*this);
}

bool MatchSettings::olderVersionDuringLastIO() {
	return VERSION_FROM_LAST_IO_OP < VERSION;
}
