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

#include "misc.h"

#include "warnings.h"

#pragma warning(push, 0)

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <Windows.h>

#pragma warning(pop)

using namespace std;

namespace pic2sym {

namespace {

/// Multiton supporting info/warn/errorMsg functions
class MsgCateg final {
 public:
  // Multiton, so delete copy / move ops
  MsgCateg(const MsgCateg&) = delete;
  MsgCateg(MsgCateg&&) = delete;
  void operator=(const MsgCateg&) = delete;
  void operator=(MsgCateg&&) = delete;

  constexpr const string& name() const noexcept { return categName; }
  constexpr UINT val() const noexcept { return categVal; }

 private:
  explicit constexpr MsgCateg(string_view categName_, UINT categVal_) noexcept
      : categName(categName_), categVal(categVal_) {}

  string categName;
  UINT categVal;

 public:
  /// Declaration of a wrapping struct for the only 3 instances of MsgCateg
  struct Instances;  // static constexpr fields need fully defined classes
};

struct MsgCateg::Instances {
  /// The only 3 instances of MsgCateg - which is fully defined here
  static constexpr MsgCateg InfoCateg{"Information", MB_ICONINFORMATION};
  static constexpr MsgCateg WarnCateg{"Warning", MB_ICONWARNING};
  static constexpr MsgCateg ErrCateg{"Error", MB_ICONERROR};
};

#ifndef UNIT_TESTING
/// When interacting with the user, the messages are nicer as popup windows
void msg(const MsgCateg& msgCateg,
         string_view title_,
         string_view text) noexcept {
  string title{title_};
  if (title.empty())
    title = msgCateg.name();

  MessageBox(nullptr, str2wstr(text).c_str(), str2wstr(title).c_str(),
             MB_OK | MB_TASKMODAL | MB_SETFOREGROUND | msgCateg.val());
}

#else  // UNIT_TESTING defined
/// When performing Unit Testing, the messages will appear on the console
void msg(const MsgCateg& msgCateg,
         string_view title,
         string_view text) noexcept {
  cout.flush();
  cerr.flush();
  ostream& os = (&msgCateg == &MsgCateg::Instances::ErrCateg) ? cerr : cout;
  os << msgCateg.name();
  if (title.empty())
    os << " ->\n";
  else
    os << " -> <<" << title << ">>\n";
  os << text << endl;
}

#endif  // UNIT_TESTING

}  // anonymous namespace

void infoMsg(string_view text, string_view title /* = ""*/) noexcept {
  msg(MsgCateg::Instances::InfoCateg, title, text);
}

void warnMsg(string_view text, string_view title /* = ""*/) noexcept {
  msg(MsgCateg::Instances::WarnCateg, title, text);
}

void errMsg(string_view text, string_view title /* = ""*/) noexcept {
  msg(MsgCateg::Instances::ErrCateg, title, text);
}

wstring str2wstr(string_view str) noexcept {
  return wstring{CBOUNDS(str)};
}

string wstr2str(wstring_view wstr) noexcept {
#pragma warning(disable : WARN_LOSSY_CONVERSION_CHANCE)
  return string{CBOUNDS(wstr)};
#pragma warning(default : WARN_LOSSY_CONVERSION_CHANCE)
}

}  // namespace pic2sym
