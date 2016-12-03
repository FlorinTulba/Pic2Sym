/************************************************************************************************
 The application Pic2Sym approximates images by a
 grid of colored symbols with colored backgrounds.

 This file belongs to the Pic2Sym project.

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
 ***********************************************************************************************/

#include "fontErrorsHelper.h"

/// Pair describing a FreeType error - the code and the string message
struct FtError {
	int code;			///< error code
	const char* msg;	///< error message
};

#include <ft2build.h>
#include FT_TYPES_H
#include FT_ERRORS_H

#undef __FTERRORS_H__
#define FT_ERRORDEF( e, v, s )  { e, s },
#define FT_ERROR_START_LIST     {
#define FT_ERROR_END_LIST       { 0, NULL } };

static std::vector<const std::string>&& initFtErrors() {
	const FtError ft_errors[] =
#include FT_ERRORS_H

	size_t registeredErrors = sizeof(ft_errors) / sizeof(FtError);
	int maxErrCode = INT_MIN;
	for(const auto &err : ft_errors) {
		if(err.code > maxErrCode)
			maxErrCode = err.code;
	}

	using namespace std;

	static vector<const string> _FtErrors(maxErrCode + 1);
	
	for(const auto &err : ft_errors) {
		if(err.msg != nullptr)
			_FtErrors[err.code] = err.msg;
	}

	for(int i = 0; i <= maxErrCode; ++i) {
		if(_FtErrors[i].empty())
			_FtErrors[i] = to_string(i);
	}

	return std::move(_FtErrors);
}

using namespace std;
const vector<const string> FtErrors(initFtErrors());
