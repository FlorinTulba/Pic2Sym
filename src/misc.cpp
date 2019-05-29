/******************************************************************************
 The application Pic2Sym approximates images by a
 grid of colored symbols with colored backgrounds.

 Copyrights from the libraries used by the program:
 - (c) 2003 Boost (www.boost.org)
     License: doc/licenses/Boost.lic
     http://www.boost.org/LICENSE_1_0.txt
 - (c) 2015-2016 OpenCV (www.opencv.org)
     License: doc/licenses/OpenCV.lic
     http://opencv.org/license/
 - (c) 1996-2002, 2006 The FreeType Project (www.freetype.org)
     License: doc/licenses/FTL.txt
     http://git.savannah.gnu.org/cgit/freetype/freetype2.git/plain/docs/FTL.TXT
 - (c) 1997-2002 OpenMP Architecture Review Board (www.openmp.org)
   (c) Microsoft Corporation (implementation for OpenMP C/C++ v2.0 March 2002)
     See: https://msdn.microsoft.com/en-us/library/8y6825x5.aspx
 - (c) 1995-2017 zlib software (Jean-loup Gailly and Mark Adler - www.zlib.net)
     License: doc/licenses/zlib.lic
     http://www.zlib.net/zlib_license.html


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
 If not, see: http://www.gnu.org/licenses/agpl-3.0.txt .
 *****************************************************************************/

#include "precompiled.h"

#include "misc.h"

#pragma warning(push, 0)

#define WIN32_LEAN_AND_MEAN
#include <Windows.h>

#pragma warning(pop)

using namespace std;

namespace {

/// Multiton supporting info/warn/errorMsg functions
class MsgCateg final {
 public:
  // Multiton, so delete copy / move ops
  MsgCateg(const MsgCateg&) = delete;
  MsgCateg(MsgCateg&&) = delete;
  void operator=(const MsgCateg&) = delete;
  void operator=(MsgCateg&&) = delete;

  constexpr const char* name() const noexcept { return categName; }
  constexpr UINT val() const noexcept { return categVal; }

 private:
  constexpr MsgCateg(const char* categName_, UINT categVal_) noexcept
      : categName(categName_), categVal(categVal_) {}

  const char* categName;
  const UINT categVal;

 public:
  static const MsgCateg INFO_CATEG, WARN_CATEG, ERR_CATEG;
};

const MsgCateg MsgCateg::INFO_CATEG("Information", MB_ICONINFORMATION);
const MsgCateg MsgCateg::WARN_CATEG("Warning", MB_ICONWARNING);
const MsgCateg MsgCateg::ERR_CATEG("Error", MB_ICONERROR);

#ifndef UNIT_TESTING
/// When interacting with the user, the messages are nicer as popup windows
void msg(const MsgCateg& msgCateg,
         const string& title_,
         const string& text) noexcept {
  string title = title_;
  if (title.empty())
    title = msgCateg.name();

  MessageBox(nullptr, str2wstr(text).c_str(), str2wstr(title).c_str(),
             MB_OK | MB_TASKMODAL | MB_SETFOREGROUND | msgCateg.val());
}

#else  // UNIT_TESTING defined
/// When performing Unit Testing, the messages will appear on the console
void msg(const MsgCateg& msgCateg,
         const string& title,
         const string& text) noexcept {
  cout.flush();
  cerr.flush();
  ostream& os = (&msgCateg == &MsgCateg::ERR_CATEG) ? cerr : cout;
  os << msgCateg.name();
  if (title.empty())
    os << " ->\n";
  else
    os << " -> <<" << title << ">>\n";
  os << text << endl;
}

#endif  // UNIT_TESTING

}  // anonymous namespace

void infoMsg(const string& text, const string& title /* = ""*/) noexcept {
  msg(MsgCateg::INFO_CATEG, title, text);
}

void warnMsg(const string& text, const string& title /* = ""*/) noexcept {
  msg(MsgCateg::WARN_CATEG, title, text);
}

void errMsg(const string& text, const string& title /* = ""*/) noexcept {
  msg(MsgCateg::ERR_CATEG, title, text);
}

wstring str2wstr(const string& str) noexcept {
  return wstring(CBOUNDS(str));
}

string wstr2str(const wstring& wstr) noexcept {
  return string(CBOUNDS(wstr));
}

WrappingActions::WrappingActions(const std::function<void()>& fnCtor,
                                 const function<void()>& fnDtor_) noexcept
    : fnDtor(fnDtor_) {
  fnCtor();
}

WrappingActions::~WrappingActions() noexcept {
  fnDtor();
}

ScopeExitAction::ScopeExitAction(const std::function<void()>& fn) noexcept
    : WrappingActions(WrappingActions::NoAction, fn) {}
