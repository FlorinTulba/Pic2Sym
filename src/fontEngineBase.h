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

#ifndef H_FONT_ENGINE_BASE
#define H_FONT_ENGINE_BASE

#include "jobMonitorBase.h"
#include "pixMapSymBase.h"
#include "tinySymsProvider.h"

#pragma warning(push, 0)

#include <ft2build.h>

#include <string>
#include FT_FREETYPE_H

#pragma warning(pop)

namespace pic2sym::syms {

/// FontEngine class wraps some necessary FreeType functionality.
class IFontEngine /*abstract*/ : public ITinySymsProvider {
 public:
  /// When unable to process a font type, invalidate it completely
  virtual void invalidateFont() noexcept = 0;

  /// Tries to use the font from 'fontFile_'
  virtual bool newFont(const std::string& fontFile_) noexcept = 0;

  /**
  Sets the desired font height in pixels
  @throw logic_error for an incomplete font configuration
  @throw invalid_argument for fontSz_ outside the range specified in the
  settings or the value cannot actually be used for the glyphs
  @throw runtime_error when there are issues loading the resized glyphs

  Exceptions from above to be only reported, not handled

  @throw AbortedJob when user aborts the operation.
  This must be handled by caller.
  */
  virtual void setFontSz(unsigned fontSz_) = 0;

  /**
  Sets an encoding by name
  @throw logic_error for an incomplete font configuration

  Exception to be only reported, not handled
  */
  virtual bool setEncoding(const std::string& encName,
                           bool forceUpdate = false) noexcept(!UT) = 0;

  /**
  Switches to nth unique encoding
  @throw logic_error for an incomplete font configuration

  Exception to be only reported, not handled
  */
  virtual bool setNthUniqueEncoding(unsigned idx) noexcept(!UT) = 0;

  /**
  Upper bound of symbols count in the cmap
  @throw logic_error for an incomplete font configuration

  Exception to be only reported, not handled
  */
  virtual unsigned upperSymsCount() const noexcept(!UT) = 0;

  /**
  Get the symbols set
  @throw logic_error for an incompletely-loaded font set

  Exception to be only reported, not handled
  */
  virtual const syms::VPixMapSym& symsSet() const noexcept(!UT) = 0;

  /**
  Get coverageOfSmallGlyphs
  @throw logic_error for an incompletely-loaded font set

  Exception to be only reported, not handled
  */
  virtual double smallGlyphsCoverage() const noexcept(!UT) = 0;

  /// Font name provided by Font Dialog
  virtual const std::string& fontFileName() const noexcept = 0;

  /**
  @return the count of unique encodings
  @throw logic_error for incomplete font configuration

  Exception to be only reported, not handled
  */
  virtual unsigned uniqueEncodings() const noexcept(!UT) = 0;

  /**
  Gets current encoding

  @param pEncodingIndex if not nullptr, address where to store the index of the
  current encoding

  @return current encoding

  @throw logic_error for incomplete font configuration

  Exception to be only reported, not handled
  */
  virtual const std::string& getEncoding(
      unsigned* pEncodingIndex = nullptr) const noexcept(!UT) = 0;

  virtual FT_String* getFamily() const noexcept = 0;  ///< get font family
  virtual FT_String* getStyle() const noexcept = 0;   ///< get font style

  /**
  @return the type of the symbols, independent of font size

  @throw logic_error only in UnitTesting for incomplete font configuration

  Exception to be only reported, not handled
  */
  virtual std::string getFontType() const noexcept(!UT) = 0;

  /// Setting the symbols monitor
  virtual IFontEngine& useSymsMonitor(
      ui::AbsJobMonitor& symsMonitor_) noexcept = 0;
};

}  // namespace pic2sym::syms

#endif  // H_FONT_ENGINE_BASE
