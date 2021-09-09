/******************************************************************************
 The application Pic2Sym approximates images by a
 grid of colored symbols with colored backgrounds.

 Copyrights from the libraries used by the program:
 - (c) 2003-2021 Boost (www.boost.org)
     License: doc/licenses/Boost.lic
     http://www.boost.org/LICENSE_1_0.txt
 - (c) 2015-2021 OpenCV (www.opencv.org)
     License: doc/licenses/OpenCV.lic
     http://opencv.org/license/
 - (c) 1996-2021 The FreeType Project (www.freetype.org)
     License: doc/licenses/FTL.txt
     http://git.savannah.gnu.org/cgit/freetype/freetype2.git/plain/docs/FTL.TXT
 - (c) 1997-2021 OpenMP Architecture Review Board (www.openmp.org)
   (c) Microsoft Corporation (implementation for OpenMP C/C++ v2.0 March 2002)
     See: https://msdn.microsoft.com/en-us/library/8y6825x5.aspx
 - (c) 1995-2021 zlib software (Jean-loup Gailly and Mark Adler - www.zlib.net)
     License: doc/licenses/zlib.lic
     http://www.zlib.net/zlib_license.html
 - (c) 2015-2021 Microsoft Guidelines Support Library - github.com/microsoft/GSL
     License: doc/licenses/MicrosoftGSL.lic
     https://raw.githubusercontent.com/microsoft/GSL/main/LICENSE


 (c) 2016-2021 Florin Tulba <florintulba@yahoo.com>

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
 If not, see: http://www.gnu.org/licenses/agpl-3.0.txt .
 *****************************************************************************/

#ifndef H_STUDY
#define H_STUDY

#pragma warning(push, 0)

#include <span>
#include <string>
#include <string_view>

#include <gsl/gsl>

#pragma warning(pop)

namespace pic2sym {

/**
Free function prompting the developer if a certain action is desired at the
specified point in the application. Helpful for checking branches which are
quite difficult to reach while performing some code coverage.

For instance:
  if(rareCondition) {x;}

should be changed to:
  if(rareCondition || prompt(question, context)) {x;}

In this way, the developer can choose which branches to inspect.

The Unit Testing project can use this function, as well:
- instead of prompting the user, the function would return false, unless
- the context parameter is found within a customizable set of contexts
dynamically controlling which branches to take
*/
bool prompt(std::string_view question, const std::string& context) noexcept;

#ifndef UNIT_TESTING  // Next 2 functions are used only within main.cpp

/**
Free function allowing the application to perform some separate studies instead
of the normal launch.
@return false for a classic launch and true when intending to start some
separate investigations
*/
bool studying() noexcept;

/**
Free function to provide the implementation of the case to be studied.
The parameters are passed further from the main function of the application.
*/
int study(std::span<gsl::zstring<>> args) noexcept;

#endif  // UNIT_TESTING not defined

}  // namespace pic2sym

#endif  // H_STUDY
