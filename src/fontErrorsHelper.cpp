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

#include "fontErrorsHelper.h"

#ifndef AI_REVIEWER_CHECK

#include "warnings.h"

#pragma warning ( push, 0 )

#	include <ft2build.h>
#	include FT_TYPES_H
#	include FT_ERRORS_H

#pragma warning ( pop )

#undef __FTERRORS_H__
#define FT_ERRORDEF( e, v, s )  { s, e },
#define FT_ERROR_START_LIST     {
#define FT_ERROR_END_LIST       { 0, NULL } };

#endif // AI_REVIEWER_CHECK not defined

namespace {

	/// Pair describing a FreeType error - the code and the string message
	struct FtError {
		const char* msg;	///< error message
		int code;			///< error code
	};

	/// Initializes the vector of FreeType error strings
	static const std::vector<const std::stringType>& initFtErrors() {
#pragma warning ( disable : WARN_THREAD_UNSAFE )
		static std::vector<const std::stringType> _FtErrors;
		static bool initilized = false;
#pragma warning ( default : WARN_THREAD_UNSAFE )

#ifndef AI_REVIEWER_CHECK
		if(!initilized) {
			const FtError ft_errors[] =
#pragma warning ( push, 0 )
#	include FT_ERRORS_H
#pragma warning ( pop )

			int maxErrCode = INT_MIN;
			for(const FtError &err : ft_errors) {
				if(err.code > maxErrCode)
					maxErrCode = err.code;
			}

			_FtErrors.resize(size_t(maxErrCode + 1));

			for(const FtError &err : ft_errors) {
				if(err.msg != nullptr)
					_FtErrors[(size_t)err.code] = err.msg;
			}

			for(int i = 0; i <= maxErrCode; ++i) {
				if(_FtErrors[(size_t)i].empty())
					_FtErrors[(size_t)i] = std::to_string(i);
			}

			initilized = true;
		}
#endif // AI_REVIEWER_CHECK

		return _FtErrors;
	}
} // anonymous namespace

const std::vector<const std::stringType> &FtErrors = initFtErrors();
