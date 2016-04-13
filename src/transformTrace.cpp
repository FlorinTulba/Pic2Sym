/****************************************************************************************
 The application Pic2Sym approximates images by a
 grid of colored symbols with colored backgrounds.

 This file was created on 2016-4-11
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

#if defined _DEBUG || defined UNIT_TESTING

#include "matchParams.h"

#include <string>
#include <boost/optional/optional_io.hpp>

static const std::wstring COMMA(L",\t");

std::wostream& operator<<(std::wostream &os, const MatchParams &mp) {
	os<<mp.ssim<<COMMA
		<<mp.sdevFg<<COMMA<<mp.sdevEdge<<COMMA<<mp.sdevBg<<COMMA
		<<mp.fg<<COMMA<<mp.bg<<COMMA;

	if(mp.mcPatchApprox)
		os<<mp.mcPatchApprox->x<<COMMA<<mp.mcPatchApprox->y<<COMMA;
	else
		os<<boost::none<<COMMA<<boost::none<<COMMA;

	if(mp.mcPatch)
		os<<mp.mcPatch->x<<COMMA<<mp.mcPatch->y<<COMMA;
	else
		os<<boost::none<<COMMA<<boost::none<<COMMA;

	os<<mp.symDensity;

	return os;
}

std::wostream& operator<<(std::wostream &os, const BestMatch &bm) {
	if(!bm.symCode)
		os<<boost::none;
	else {
		unsigned long symCode = *bm.symCode;
		if(bm.unicode) {
			switch(symCode) {
				case (unsigned long)',':
					os<<L"COMMA"; break;
				case (unsigned long)'(':
					os<<L"OPEN_PAR"; break;
				case (unsigned long)')':
					os<<L"CLOSE_PAR"; break;
				default:
					// for other characters, check if they can be displayed on the current console
					if(os<<(wchar_t)symCode) {
						// when they can be displayed, add in () their code
						os<<'('<<symCode<<')';
					} else { // when they can't be displayed, show just their code
						os.clear(); // clear the error first
						os<<symCode;
					}
			}
		} else
			os<<symCode;
	}

	os<<COMMA<<bm.score<<COMMA<<bm.bestVariant.params;
	return os;
}

#endif // defined _DEBUG || defined UNIT_TESTING


#if defined _DEBUG && !defined UNIT_TESTING

#include "transformTrace.h"
#include "matchSettingsManip.h"

#include <boost/filesystem/path.hpp>

using namespace std;
using namespace boost::filesystem;

TransformTrace::TransformTrace(const string &studiedCase_, unsigned sz_, bool isUnicode_) :
		studiedCase(studiedCase_), sz(sz_), isUnicode(isUnicode_) {
	path traceFile(MatchSettingsManip::instance().getWorkDir());
	traceFile.append("data_").concat(studiedCase).
		concat(".csv"); // generating a CSV trace file

	ofs = wofstream(traceFile.c_str());
	ofs<<"#Row"<<COMMA<<"#Col"<<COMMA<<BestMatch::HEADER<<endl;
}

TransformTrace::~TransformTrace() {
	ofs.close();
}

void TransformTrace::newEntry(unsigned r, unsigned c, const BestMatch &best) {
	ofs<<r/sz<<COMMA<<c/sz<<COMMA<<const_cast<BestMatch&>(best).setUnicode(isUnicode)<<endl;

	// flush after every row fully transformed
	if(r > transformingRow) {
		ofs.flush();
		transformingRow = r;
	}
}

#endif // _DEBUG && !UNIT_TESTING