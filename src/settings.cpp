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

#include "settings.h"

using namespace std;

extern const unsigned Settings_MAX_THRESHOLD_FOR_BLANKS;
extern const unsigned Settings_MIN_H_SYMS;
extern const unsigned Settings_MAX_H_SYMS;
extern const unsigned Settings_MIN_V_SYMS;
extern const unsigned Settings_MAX_V_SYMS;
extern const unsigned Settings_MIN_FONT_SIZE;
extern const unsigned Settings_MAX_FONT_SIZE;
extern const unsigned Settings_DEF_FONT_SIZE;

bool ISettings::isBlanksThresholdOk(unsigned t) {
	return t < Settings_MAX_THRESHOLD_FOR_BLANKS;
}

bool ISettings::isHmaxSymsOk(unsigned syms) {
	return syms>=Settings_MIN_H_SYMS && syms<=Settings_MAX_H_SYMS;
}

bool ISettings::isVmaxSymsOk(unsigned syms) {
	return syms>=Settings_MIN_V_SYMS && syms<=Settings_MAX_V_SYMS;
}

bool ISettings::isFontSizeOk(unsigned fs) {
	return fs>=Settings_MIN_FONT_SIZE && fs<=Settings_MAX_FONT_SIZE;
}

Settings::Settings(const MatchSettings &ms_) :
	ss(Settings_DEF_FONT_SIZE), is(Settings_MAX_H_SYMS, Settings_MAX_V_SYMS), ms(ms_) {}

Settings::Settings() :
	ss(Settings_DEF_FONT_SIZE), is(Settings_MAX_H_SYMS, Settings_MAX_V_SYMS), ms() {}

const SymSettings& Settings::getSS() const {
	return ss;
}

const ImgSettings& Settings::getIS() const {
	return is;
}

const MatchSettings& Settings::getMS() const {
	return ms;
}

SymSettings& Settings::SS() {
	return ss;
}

ImgSettings& Settings::IS() {
	return is;
}

MatchSettings& Settings::MS() {
	return ms;
}

ostream& operator<<(ostream &os, const ISettings &s) {
	os<<s.getSS()<<s.getIS()<<s.getMS()<<endl;
	return os;
}

ostream& operator<<(ostream &os, const ImgSettings &is) {
	os<<"hMaxSyms"<<" : "<<is.hMaxSyms<<endl;
	os<<"vMaxSyms"<<" : "<<is.vMaxSyms<<endl;
	return os;
}

ostream& operator<<(ostream &os, const SymSettings &ss) {
	os<<"fontFile"<<" : "<<ss.fontFile<<endl;
	os<<"encoding"<<" : "<<ss.encoding<<endl;
	os<<"fontSz"<<" : "<<ss.fontSz<<endl;
	return os;
}

ostream& operator<<(ostream &os, const MatchSettings &ms) {
	if(ms.hybridResultMode)
		os<<"hybridResultMode"<<" : "<<boolalpha<<ms.hybridResultMode<<endl;
	if(ms.kSsim > 0.)
		os<<"kSsim"<<" : "<<ms.kSsim<<endl;
	if(ms.kSdevFg > 0.)
		os<<"kSdevFg"<<" : "<<ms.kSdevFg<<endl;
	if(ms.kSdevEdge > 0.)
		os<<"kSdevEdge"<<" : "<<ms.kSdevEdge<<endl;
	if(ms.kSdevBg > 0.)
		os<<"kSdevBg"<<" : "<<ms.kSdevBg<<endl;
	if(ms.kContrast > 0.)
		os<<"kContrast"<<" : "<<ms.kContrast<<endl;
	if(ms.kMCsOffset > 0.)
		os<<"kMCsOffset"<<" : "<<ms.kMCsOffset<<endl;
	if(ms.kCosAngleMCs > 0.)
		os<<"kCosAngleMCs"<<" : "<<ms.kCosAngleMCs<<endl;
	if(ms.kSymDensity > 0.)
		os<<"kSymDensity"<<" : "<<ms.kSymDensity<<endl;
	if(ms.threshold4Blank > 0.)
		os<<"threshold4Blank"<<" : "<<ms.threshold4Blank<<endl;
	return os;
}
