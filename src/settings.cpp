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

#include "imgSettings.h"
#include "matchSettings.h"
#include "settings.h"
#include "symSettings.h"
#include "warnings.h"

using namespace std;

unsigned Settings::VERSION_FROM_LAST_IO_OP = UINT_MAX;

extern const unsigned Settings_MAX_THRESHOLD_FOR_BLANKS;
extern const unsigned Settings_MIN_H_SYMS;
extern const unsigned Settings_MAX_H_SYMS;
extern const unsigned Settings_MIN_V_SYMS;
extern const unsigned Settings_MAX_V_SYMS;
extern const unsigned Settings_MIN_FONT_SIZE;
extern const unsigned Settings_MAX_FONT_SIZE;
extern const unsigned Settings_DEF_FONT_SIZE;

bool ISettings::isBlanksThresholdOk(
    unsigned t,
    const string& reportedItem /*=""*/) noexcept {
  const bool result = (t < Settings_MAX_THRESHOLD_FOR_BLANKS);
  if (!result && !reportedItem.empty())
    cerr << "Configuration item '" << reportedItem << "' (" << t
         << ") needs to be < " << Settings_MAX_THRESHOLD_FOR_BLANKS << "!"
         << endl;
  return result;
}

bool ISettings::isHmaxSymsOk(unsigned syms) noexcept {
  return syms >= Settings_MIN_H_SYMS && syms <= Settings_MAX_H_SYMS;
}

bool ISettings::isVmaxSymsOk(unsigned syms) noexcept {
  return syms >= Settings_MIN_V_SYMS && syms <= Settings_MAX_V_SYMS;
}

bool ISettings::isFontSizeOk(unsigned fs) noexcept {
  return fs >= Settings_MIN_FONT_SIZE && fs <= Settings_MAX_FONT_SIZE;
}

Settings::Settings(const IMatchSettings& ms_) noexcept
    : ss(make_unique<SymSettings>(Settings_DEF_FONT_SIZE)),
      is(make_unique<ImgSettings>(Settings_MAX_H_SYMS, Settings_MAX_V_SYMS)),
      ms(ms_.clone()) {
  assert(ss);
  assert(is);
  assert(ms);
}

Settings::Settings() noexcept
    : ss(make_unique<SymSettings>(Settings_DEF_FONT_SIZE)),
      is(make_unique<ImgSettings>(Settings_MAX_H_SYMS, Settings_MAX_V_SYMS)),
      ms(make_unique<MatchSettings>()) {
  assert(ss);
  assert(is);
  assert(ms);
}

const ISymSettings& Settings::getSS() const noexcept {
  return *ss;
}

const IfImgSettings& Settings::getIS() const noexcept {
  return *is;
}

const IMatchSettings& Settings::getMS() const noexcept {
  return *ms;
}

ISymSettings& Settings::refSS() noexcept {
  return *ss;
}

IfImgSettings& Settings::refIS() noexcept {
  return *is;
}

IMatchSettings& Settings::refMS() noexcept {
  return *ms;
}

#pragma warning(disable : WARN_EXPR_ALWAYS_FALSE)
bool Settings::olderVersionDuringLastIO() noexcept {
  return MatchSettings::olderVersionDuringLastIO() ||
         SymSettings::olderVersionDuringLastIO() ||
         ImgSettings::olderVersionDuringLastIO() ||
         VERSION_FROM_LAST_IO_OP < VERSION;
}
#pragma warning(default : WARN_EXPR_ALWAYS_FALSE)

ostream& operator<<(ostream& os, const ISettings& s) noexcept {
  os << s.getSS() << s.getIS() << s.getMS() << endl;
  return os;
}

ostream& operator<<(ostream& os, const IfImgSettings& is) noexcept {
  os << "hMaxSyms"
     << " : " << is.getMaxHSyms() << '\n';
  os << "vMaxSyms"
     << " : " << is.getMaxVSyms() << endl;
  return os;
}

ostream& operator<<(ostream& os, const ISymSettings& ss) noexcept {
  os << "fontFile"
     << " : " << ss.getFontFile() << '\n';
  os << "encoding"
     << " : " << ss.getEncoding() << '\n';
  os << "fontSz"
     << " : " << ss.getFontSz() << endl;
  return os;
}

ostream& operator<<(ostream& os, const IMatchSettings& ms) noexcept {
  os << ms.toString(true);
  return os;
}
