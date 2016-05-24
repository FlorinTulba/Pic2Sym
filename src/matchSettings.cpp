/****************************************************************************************
 The application Pic2Sym approximates images by a
 grid of colored symbols with colored backgrounds.

 This file was created on 2016-4-10
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
 - (c) 1997-2002 OpenMP Architecture Review Board (www.openmp.org)
   (c) Microsoft Corporation (Visual C++ implementation for OpenMP C/C++ Version 2.0 March 2002)
   See: <https://msdn.microsoft.com/en-us/library/8y6825x5(v=vs.140).aspx>
 
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

#include "matchSettingsManip.h"

#include <iostream>

using namespace std;

#ifndef UNIT_TESTING
MatchSettings::MatchSettings() {
	MatchSettingsManip::instance().initMatchSettings(*this);
}
#endif

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
