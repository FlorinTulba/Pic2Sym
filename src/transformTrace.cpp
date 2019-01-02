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

#if defined _DEBUG || defined UNIT_TESTING

#include "matchParamsBase.h"
#include "bestMatchBase.h"
#include "warnings.h"

#pragma warning ( push, 0 )

#include <boost/optional/optional_io.hpp>

#pragma warning ( pop )

const std::wstringType& COMMA() {
#pragma warning ( disable : WARN_THREAD_UNSAFE )
	static const std::wstringType res(L",\t");
#pragma warning ( default : WARN_THREAD_UNSAFE )
	return res;
}

std::wostream& operator<<(std::wostream &wos, const IMatchParams &mp) {
	wos<<mp.toWstring();
	return wos;
}

std::wostream& operator<<(std::wostream &wos, const IBestMatch &bm) {
	wos<<bm.toWstring();
	return wos;
}

#endif // _DEBUG || UNIT_TESTING


#if defined _DEBUG && !defined UNIT_TESTING

#include "transformTrace.h"
#include "appStart.h"

using namespace std;
using namespace boost::filesystem;

TransformTrace::TransformTrace(const stringType &studiedCase_, unsigned sz_, bool isUnicode_) :
		studiedCase(studiedCase_), sz(sz_), isUnicode(isUnicode_) {
	path traceFile(AppStart::dir());
	traceFile.append("data_").concat(studiedCase).
		concat(".csv"); // generating a CSV trace file

	extern const wstringType BestMatch_HEADER;
	wofs = wofstream(traceFile.c_str());
	wofs<<"#Row"<<COMMA()<<"#Col"<<COMMA()<<BestMatch_HEADER<<endl;
}

TransformTrace::~TransformTrace() {
	wofs.close();
}

void TransformTrace::newEntry(unsigned r, unsigned c, const IBestMatch &best) {
	wofs<<r/sz<<COMMA()<<c/sz<<COMMA()<<const_cast<IBestMatch&>(best).setUnicode(isUnicode)<<endl;

	// flush after every row fully transformed
	if(r > transformingRow) {
		wofs.flush();
		transformingRow = r;
	}
}

#endif // _DEBUG && !UNIT_TESTING
