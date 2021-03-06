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

#ifndef H_MISC
#define H_MISC

#include "warnings.h"

#pragma warning ( push, 0 )

#include "std_string.h"
#include <iostream>
#include <iomanip>

#pragma warning ( pop )

// Error margin
const double EPS = 1e-6;

// Prevent warning about unreferenced parameters
#define UNREFERENCED_PARAMETER(Par) (Par)

// Display an expression and its value
#define PRINT(expr)			std::cout<<#expr " : "<<(expr)
#define PRINTLN(expr)		PRINT(expr)<<std::endl
#define PRINT_H(expr)		std::cout<<#expr " : 0x"<<std::hex<<(expr)<<std::dec
#define PRINTLN_H(expr)		PRINT_H(expr)<<std::endl

// Oftentimes functions operating on ranges need the full range.
// Example: copy(x.begin(), x.end(), ..) => copy(BOUNDS(x), ..)
#define BOUNDS(iterable)	std::begin(iterable), std::end(iterable)
#define CBOUNDS(iterable)	std::cbegin(iterable), std::cend(iterable)
#define BOUNDS_FOR_ITEM_TYPE(iterable, type)	iterable.begin<type>(), iterable.end<type>()

// string <-> wstring conversions
std::wstringType str2wstr(const std::stringType &str);
std::stringType wstr2str(const std::wstringType &wstr);

// Notifying the user
void infoMsg(const std::stringType &text, const std::stringType &title = "");
void warnMsg(const std::stringType &text, const std::stringType &title = "");
void errMsg(const std::stringType &text, const std::stringType &title = "");

// Throwing exceptions while displaying the exception message to the console
#ifndef AI_REVIEWER_CHECK
/*
First version should be used when the exception message is constant.
Declaring a method-static variable makes sense only when the exception is caught,
so particularly for Unit Testing for this application.
Otherwise, the program just leaves, letting no chance for reusing the method-static variable.
*/
#define THROW_WITH_CONST_MSG(excMsg, excType) \
	{ \
		__pragma( warning( disable : WARN_THREAD_UNSAFE ) ) \
		static const std::stringType constErrMsgForConsoleAndThrow(excMsg); \
		__pragma( warning( default : WARN_THREAD_UNSAFE ) ) \
		std::cerr<<constErrMsgForConsoleAndThrow<<std::endl; \
		throw excType(constErrMsgForConsoleAndThrow); \
	}
/*
Second version should be used when the exception message is variable (reports specific values).
*/
#define THROW_WITH_VAR_MSG(msg, excType) \
	{ \
		const std::stringType varErrMsgForConsoleAndThrow(msg); \
		std::cerr<<varErrMsgForConsoleAndThrow<<std::endl; \
		throw excType(varErrMsgForConsoleAndThrow); \
	}

#else // AI_REVIEWER_CHECK defined

#define THROW_WITH_CONST_MSG(excMsg, excType) throw excType("")
#define THROW_WITH_VAR_MSG(msg, excType) throw excType("")

#endif // AI_REVIEWER_CHECK

#endif // H_MISC
