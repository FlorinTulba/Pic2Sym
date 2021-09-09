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

#ifndef H_SELECT_SYMBOLS_BASE
#define H_SELECT_SYMBOLS_BASE

#include "symbolsSupportBase.h"

namespace pic2sym {

/// Allows saving a selection of symbols pointed within the charmap viewer
class ISelectSymbols /*abstract*/ {
 public:
  /// Provides details about the symbol under the mouse
  virtual const syms::ISymData* pointedSymbol(int x, int y) const noexcept = 0;

  /// Appends the code of the symbol under the mouse to the status bar
  virtual void displaySymCode(unsigned long symCode) const noexcept = 0;

  /// Appends the matrix of the pointed symbol (by Ctrl + left click) to a list
  /// for separate investigation
  virtual void enlistSymbolForInvestigation(
      const syms::ISymData& sd) const noexcept = 0;

  /// Saves the list with the matrices of the symbols to investigate to a file
  /// and then clears this list
  virtual void symbolsReadyToInvestigate() const noexcept = 0;

  virtual ~ISelectSymbols() noexcept = 0 {}
};

}  // namespace pic2sym

#endif  // H_SELECT_SYMBOLS_BASE
