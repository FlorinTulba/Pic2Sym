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

#ifndef H_SYMS_LOADING_FAILURE
#define H_SYMS_LOADING_FAILURE

#pragma warning ( push, 0 )

#include "std_string.h"

#pragma warning ( pop )

/// Base exception class for easier catching and handling failures while loading normal / tiny symbols
class SymsLoadingFailure /*abstract*/ : public std::runtime_error {
protected:
	explicit SymsLoadingFailure(const std::stringType &_Message);
	explicit SymsLoadingFailure(const char *_Message);

public:
#ifdef AI_REVIEWER_CHECK // Without this virtual destructor, AI Reviewer signals Refused Bequest for the derived types
	virtual ~SymsLoadingFailure() = 0 {}
#endif // AI_REVIEWER_CHECK defined

	void informUser(const std::stringType &msg) const; ///< informs the user about the problem
};

/// Distinct exception class for easier catching and handling failures while loading tiny symbols
struct TinySymsLoadingFailure : SymsLoadingFailure {
	explicit TinySymsLoadingFailure(const std::stringType &_Message);
	explicit TinySymsLoadingFailure(const char *_Message);
};

/// Distinct exception class for easier catching and handling failures while loading normal symbols
struct NormalSymsLoadingFailure : SymsLoadingFailure {
	explicit NormalSymsLoadingFailure(const std::stringType &_Message);
	explicit NormalSymsLoadingFailure(const char *_Message);
};

#endif // H_SYMS_LOADING_FAILURE
