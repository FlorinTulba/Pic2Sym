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

#ifndef H_STUDY
#define H_STUDY

#pragma warning ( push, 0 )

#include "std_string.h"

#pragma warning ( pop )

/**
Free function prompting the developer if a certain action is desired at the specified point in the application.
Helpful for checking branches which are quite difficult to reach while performing some code coverage.

For instance:
	if(rareCondition) {x;}

should be changed to:
	if(rareCondition || prompt(question, context)) {x;}

In this way, the developer can choose which branches to inspect.

The Unit Testing project can use this function, as well:
- instead of prompting the user, the function would return false, unless
- the context parameter is found within a customizable set of contexts dynamically controlling which branches to take
*/
bool prompt(const std::stringType &question, const std::stringType &context);

#ifndef UNIT_TESTING // Next 2 functions are used only within main.cpp

/**
Free function allowing the application to perform some separate studies instead of the normal launch.
@return false for a classic launch and true when intending to start some separate investigations
*/
bool studying();

/**
Free function to provide the implementation of the case to be studied.
The parameters are passed further from the main function of the application.
*/
void study(int argc, char* argv[]);

#endif // UNIT_TESTING not defined

#endif // H_STUDY
