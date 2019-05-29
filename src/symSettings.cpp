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

#include "symSettings.h"
#include "warnings.h"

#pragma warning(push, 0)

#include <iostream>

#pragma warning(pop)

using namespace std;

unsigned SymSettings::VERSION_FROM_LAST_IO_OP = UINT_MAX;

void SymSettings::reset() noexcept {
  fontFile = encoding = "";
  // the font size should remain on its value from the Control Panel
}

bool SymSettings::initialized() const noexcept {
  return !fontFile.empty() && !encoding.empty();
}

void SymSettings::setFontFile(const std::string& fontFile_) noexcept {
  if (fontFile == fontFile_)
    return;

  cout << "fontFile"
       << " : '" << fontFile << "' -> '" << fontFile_ << '\'' << endl;
  fontFile = fontFile_;
}

void SymSettings::setEncoding(const std::string& encoding_) noexcept {
  if (encoding == encoding_)
    return;

  cout << "encoding"
       << " : '" << encoding << "' -> '" << encoding_ << '\'' << endl;
  encoding = encoding_;
}

void SymSettings::setFontSz(unsigned fontSz_) noexcept {
  if (fontSz == fontSz_)
    return;

  cout << "fontSz"
       << " : " << fontSz << " -> " << fontSz_ << endl;
  fontSz = fontSz_;
}

unique_ptr<ISymSettings> SymSettings::clone() const noexcept {
  return make_unique<SymSettings>(*this);
}

#ifdef __cpp_lib_three_way_comparison
strong_equality SymSettings::operator<=>(const SymSettings& other) const
    noexcept {
  // if (this == &other) // Costly to always perform. Harmless & cheap if cut
  //  return strong_equality::equivalent;

  if (const auto cmp = (fontSz <=> other.fontSz); cmp != 0)
    return cmp;

  if (const auto cmp = fontFile.compare(other.fontFile) <=> 0; cmp != 0)
    return cmp;

  return encoding.compare(other.encoding) <=> 0;
}

#else   // __cpp_lib_three_way_comparison not defined
bool SymSettings::operator==(const SymSettings& other) const noexcept {
  // if (this == &other)  // Costly to always perform. Harmless & cheap if cut
  //  return true;

  if (fontSz != other.fontSz)
    return false;

  if (fontFile != other.fontFile)
    return false;

  return encoding == other.encoding;
}
#endif  // __cpp_lib_three_way_comparison

#pragma warning(disable : WARN_EXPR_ALWAYS_FALSE)
bool SymSettings::olderVersionDuringLastIO() noexcept {
  return VERSION_FROM_LAST_IO_OP < VERSION;
}
#pragma warning(default : WARN_EXPR_ALWAYS_FALSE)
