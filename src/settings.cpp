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
#include "symSettings.h"
#include "imgSettings.h"
#include "matchSettings.h"

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

Settings::Settings(const IMatchSettings &ms_) :
	ss(new SymSettings(Settings_DEF_FONT_SIZE)),
	is(new ImgSettings(Settings_MAX_H_SYMS, Settings_MAX_V_SYMS)),
	ms(move(ms_.clone())) {}

Settings::Settings() :
	ss(new SymSettings(Settings_DEF_FONT_SIZE)),
	is(new ImgSettings(Settings_MAX_H_SYMS, Settings_MAX_V_SYMS)),
	ms(new MatchSettings) {}

const ISymSettings& Settings::getSS() const {
	assert(ss);
	return *ss;
}

const IfImgSettings& Settings::getIS() const {
	assert(is);
	return *is;
}

const IMatchSettings& Settings::getMS() const {
	assert(ms);
	return *ms;
}

ISymSettings& Settings::refSS() {
	assert(ss);
	return *ss;
}

IfImgSettings& Settings::refIS() {
	assert(is);
	return *is;
}

IMatchSettings& Settings::refMS() {
	assert(ms);
	return *ms;
}

ostream& operator<<(ostream &os, const ISettings &s) {
	os<<s.getSS()<<s.getIS()<<s.getMS()<<endl;
	return os;
}

ostream& operator<<(ostream &os, const IfImgSettings &is) {
	os<<"hMaxSyms"<<" : "<<is.getMaxHSyms()<<endl;
	os<<"vMaxSyms"<<" : "<<is.getMaxVSyms()<<endl;
	return os;
}

ostream& operator<<(ostream &os, const ISymSettings &ss) {
	os<<"fontFile"<<" : "<<ss.getFontFile()<<endl;
	os<<"encoding"<<" : "<<ss.getEncoding()<<endl;
	os<<"fontSz"<<" : "<<ss.getFontSz()<<endl;
	return os;
}

ostream& operator<<(ostream &os, const IMatchSettings &ms) {
	os<<ms.toString(true);
	return os;
}
