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

#include "precompiled.h"
// This keeps precompiled.h first; Otherwise header sorting might move it

#ifndef UNIT_TESTING

#include "appStart.h"

#include "misc.h"

#pragma warning(push, 0)

#include <filesystem>

#include <gsl/gsl>

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <Windows.h>

#pragma warning(pop)

using namespace std;
using namespace std::filesystem;

namespace pic2sym {

namespace {
/// Folder containing application relevant files/subdirectories
std::filesystem::path baseFolder;

/**
The system might contain more versions of the dll-s required by this
application. Some of such versions might not be appropriate for Pic2Sym.
Therefore it is mandatory to provide the correct dll-s, especially when
deploying the program on other machines.

Some of the dll-s need to be selected while still loading the application
(before it starts running). Placing these dll-s in a folder "Pic2Sym.exe.local"
(near Pic2Sym.exe) is enough (This is a basic DLL-s redirection technique).

The dll-s loaded after the start of the application were conveniently copied
into the same directory. However, "Pic2Sym.exe.local" is ignored for these dll-s
when the application is installed in the default Program Files location, unless
forcefully pointed with 'SetDllDirectory'.

The plugins from Qt are a special dll category and can be located by:
- either calling QCoreApplication::addLibraryPath("Pic2Sym.exe.local");

- or creating 'qt.conf' file near Pic2Sym.exe containing:
  [Paths]
  Plugins=Pic2Sym.exe.local

- or by setting QT_QPA_PLATFORM_PLUGIN_PATH in the local environment to:
  Pic2Sym.exe.local/platforms

Last solution was the one adopted.
*/
void providePrivateDLLsPaths(string_view appPath) noexcept {
  const path dllsPath{absolute(appPath).concat(".local")};
  SetDllDirectory(dllsPath.wstring().c_str());
  _putenv_s("QT_QPA_PLATFORM_PLUGIN_PATH",
            path{dllsPath}.append("platforms").string().c_str());
}
}  // anonymous namespace

void AppStart::prepareEnv(string_view appFile) noexcept {
  if (!baseFolder.empty()) {
    cerr << HERE.function_name() << " shouldn't be called multiple times!";
    return;
  }

  baseFolder = absolute(appFile).remove_filename();
  providePrivateDLLsPaths(appFile);
}

const path& AppStart::dir() noexcept {
  // call it only after a call to AppStart::prepareEnv(appFile) in main()
  Expects(!baseFolder.empty());

  return baseFolder;
}

}  // namespace pic2sym

#endif  // UNIT_TESTING not defined
