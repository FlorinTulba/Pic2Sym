/****************************************************************************************
 The application Pic2Sym approximates images by a
 grid of colored symbols with colored backgrounds.

 This file belongs to the Pic2Sym project.

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

#ifndef H_MISC
#define H_MISC

#include <iostream>
#include <iomanip>
#include <string>

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
std::wstring str2wstr(const std::string &str);
std::string wstr2str(const std::wstring &wstr);

// Notifying the user
void infoMsg(const std::string &text, const std::string &title = "");
void warnMsg(const std::string &text, const std::string &title = "");
void errMsg(const std::string &text, const std::string &title = "");

// Throwing exceptions while displaying the exception message to the console
/*
First version should be used when the exception message is constant.
Declaring a method-static variable makes sense only when the exception is caught,
so particularly for Unit Testing for this application.
Otherwise, the program just leaves, letting no chance for reusing the method-static variable.
*/
#define THROW_WITH_CONST_MSG(excMsg, excType) \
	{ \
		static const std::string constErrMsgForConsoleAndThrow(excMsg); \
		std::cerr<<constErrMsgForConsoleAndThrow<<std::endl; \
		throw excType(constErrMsgForConsoleAndThrow); \
	}
/*
Second version should be used when the exception message is variable (reports specific values).
*/
#define THROW_WITH_VAR_MSG(msg, excType) \
	{ \
		const std::string varErrMsgForConsoleAndThrow(msg); \
		std::cerr<<varErrMsgForConsoleAndThrow<<std::endl; \
		throw excType(varErrMsgForConsoleAndThrow); \
	}

#endif // H_MISC