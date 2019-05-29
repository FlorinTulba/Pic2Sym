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

#ifndef H_SYM_SETTINGS_BASE
#define H_SYM_SETTINGS_BASE

#pragma warning(push, 0)

#include <iostream>
#include <memory>
#include <string>

#pragma warning(pop)

/// Base class for the parameters concerning the symbols set used for
/// approximating patches.
class ISymSettings /*abstract*/ {
 public:
  virtual const std::string& getFontFile() const noexcept = 0;
  virtual void setFontFile(const std::string& fontFile_) noexcept = 0;

  virtual const std::string& getEncoding() const noexcept = 0;
  virtual void setEncoding(const std::string& encoding_) noexcept = 0;

  virtual const unsigned& getFontSz() const noexcept = 0;
  virtual void setFontSz(unsigned fontSz_) noexcept = 0;

  /// Reset font settings apart from the font size
  /// which should remain on its value from the Control Panel
  virtual void reset() noexcept = 0;

  /// Report if these settings are initialized or not
  virtual bool initialized() const noexcept = 0;

  virtual ~ISymSettings() noexcept {}

  // If slicing is observed and becomes a severe problem, use `= delete` for all
  ISymSettings(const ISymSettings&) noexcept = default;
  ISymSettings(ISymSettings&&) noexcept = default;
  ISymSettings& operator=(const ISymSettings&) noexcept = default;
  ISymSettings& operator=(ISymSettings&&) noexcept = default;

  /// @return a copy of these settings
  virtual std::unique_ptr<ISymSettings> clone() const noexcept = 0;

 protected:
  constexpr ISymSettings() noexcept {}
};

std::ostream& operator<<(std::ostream& os, const ISymSettings& ss) noexcept;

#endif  // H_SYM_SETTINGS_BASE
