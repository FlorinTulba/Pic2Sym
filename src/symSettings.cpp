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

#include "symSettings.h"

#include "warnings.h"

#pragma warning(push, 0)

#include <iomanip>

#pragma warning(pop)

using namespace std;

namespace pic2sym::cfg {

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
       << " : " << quoted(encoding, '\'') << " -> " << quoted(encoding_, '\'')
       << endl;
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

bool SymSettings::operator==(const SymSettings& other) const noexcept {
  // if (this == &other)  // Costly to always perform. Harmless & cheap if cut
  //  return true;

  if (fontSz != other.fontSz)
    return false;

  if (fontFile != other.fontFile)
    return false;

  return encoding == other.encoding;
}

#pragma warning(disable : WARN_EXPR_ALWAYS_FALSE)
bool SymSettings::olderVersionDuringLastIO() noexcept {
  return VersionFromLast_IO_op < Version;
}
#pragma warning(default : WARN_EXPR_ALWAYS_FALSE)

}  // namespace pic2sym::cfg
