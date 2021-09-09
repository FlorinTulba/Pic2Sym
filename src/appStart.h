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

#ifndef H_APP_START
#define H_APP_START

#ifdef UNIT_TESTING
#error Should not include this file when UNIT_TESTING is defined

#else  // UNIT_TESTING not defined

#pragma warning(push, 0)

#include <filesystem>
#include <string_view>

#include <gsl/gsl>

#pragma warning(pop)

// Forward declaration of main; For precompiled headers in VS, this declaration
// needs to be present within the precompiled header, as well.
extern int main(int, gsl::zstring<>*);

namespace pic2sym {

/**
Utility for:
- providing the folder where Pic2Sym.exe and the following items are found:
  * 'res/defaultMatchSettings.txt' and 'initMatchSettings.cfg'
  * 'Output' folder
  * 'ClusteredSets' folder
  * 'TinySymsDataSets' folder
  * 'SymsSelections' folder
- configuring the environment and the paths pointing to the dll-s required by
the application
*/
class AppStart {
 public:
  /**
  Provides the folder containing most of the files required by the application.

  Must be called only after a call to prepareEnv().

  @return the location of the executed application
  @throw logic_error when unappropriately used (without the call to prepareEnv)

  Despite it throws, the method should be noexcept because:
  - the program cannot continue when this method cannot provide a result
  - this class is not accessible within UnitTesting, which means there's no
  point in ensuring that it throws when it should
  */
  static const std::filesystem::path& dir() noexcept;

 private:
  // The only place where this global can be set
  friend int ::main(int, gsl::zstring<>*);

  /**
  1. Sets the directory of the executed application provided as parameter.
  2. Configures the environment and the paths pointing to the dll-s required by
  the application

  Must be called only once from within main().
  */
  static void prepareEnv(std::string_view appFile) noexcept;
};

}  // namespace pic2sym

#endif  // H_APP_START

#endif  // UNIT_TESTING
