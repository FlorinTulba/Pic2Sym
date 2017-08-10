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

#if defined _DEBUG || defined UNIT_TESTING

#include "matchParamsBase.h"
#include "bestMatchBase.h"

#pragma warning ( push, 0 )

#include <string>
#include <boost/optional/optional_io.hpp>

#pragma warning ( pop )

static const std::wstring COMMA(L",\t");

std::wostream& operator<<(std::wostream &os, const IMatchParams &mp) {
	os<<mp.getSsim()<<COMMA
		<<mp.getSdevFg()<<COMMA<<mp.getSdevEdge()<<COMMA<<mp.getSdevBg()<<COMMA
		<<mp.getFg()<<COMMA<<mp.getBg()<<COMMA;

	if(mp.getMcPatchApprox())
		os<<mp.getMcPatchApprox()->x<<COMMA<<mp.getMcPatchApprox()->y<<COMMA;
	else
		os<<boost::none<<COMMA<<boost::none<<COMMA;

	if(mp.getMcPatch())
		os<<mp.getMcPatch()->x<<COMMA<<mp.getMcPatch()->y<<COMMA;
	else
		os<<boost::none<<COMMA<<boost::none<<COMMA;

	os<<mp.getSymDensity();

	return os;
}

std::wostream& operator<<(std::wostream &os, const IBestMatch &bm) {
	if(!bm.getSymCode())
		os<<boost::none;
	else {
		const unsigned long symCode = *bm.getSymCode();
		if(bm.isUnicode()) {
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

	os<<COMMA<<bm.getScore();
	if(bm.getParams())
		os<<COMMA<<*bm.getParams();
	return os;
}

#endif // _DEBUG || UNIT_TESTING


#if defined _DEBUG && !defined UNIT_TESTING

#include "transformTrace.h"
#include "appStart.h"

using namespace std;
using namespace boost::filesystem;

TransformTrace::TransformTrace(const string &studiedCase_, unsigned sz_, bool isUnicode_) :
		studiedCase(studiedCase_), sz(sz_), isUnicode(isUnicode_) {
	path traceFile(AppStart::dir());
	traceFile.append("data_").concat(studiedCase).
		concat(".csv"); // generating a CSV trace file

	extern const wstring BestMatch_HEADER;
	ofs = wofstream(traceFile.c_str());
	ofs<<"#Row"<<COMMA<<"#Col"<<COMMA<<BestMatch_HEADER<<endl;
}

TransformTrace::~TransformTrace() {
	ofs.close();
}

void TransformTrace::newEntry(unsigned r, unsigned c, const IBestMatch &best) {
	ofs<<r/sz<<COMMA<<c/sz<<COMMA<<const_cast<IBestMatch&>(best).setUnicode(isUnicode)<<endl;

	// flush after every row fully transformed
	if(r > transformingRow) {
		ofs.flush();
		transformingRow = r;
	}
}

#endif // _DEBUG && !UNIT_TESTING