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

#ifndef H_UNREADABLE_SYMS_FILTER
#define H_UNREADABLE_SYMS_FILTER

#include "symFilter.h"

#include "misc.h"

namespace pic2sym::syms::inline filter {

/**
Determines if a symbol appears as unreadable.

Best approach would take size 50 of the glyph and compare its features with the
ones found in the current font size version of the symbol. However, this would
still produce some mislabeled cases and besides, humans do normally recognize
many symbols even when some of their features are missing.

Supervised machine learning would be the ideal solution here, since:
- humans can label corner-cases
- font sizes are 7..50
- there are lots of mono-spaced font families, each with several encodings,
each with several styles (Regular, Italic, Bold) and each with 100..30000
symbols

A basic criteria for selecting unreadable symbols is avoiding the ones with
compact rectangular / elliptical areas, larger than 20 - 25% of the side of the
enclosing square.

Apart from the ugly, various sizes rectangular monolithic glyphs, there are some
interesting solid symbols which could be preserved: filled triangles, circles,
playing cards suits, smilies, dices, arrows a.o.

The current implementation is a compromise surprising the fact that smaller
fonts are progressively less readable.
*/
class UnreadableSymsFilter : public TSymFilter<UnreadableSymsFilter> {
 public:
  SYM_FILTER_DECLARE_IS_ENABLED(UnreadableSymsFilter);

  /**
  Checks if pms can be filtered out based on this filter and sfc.
  @return false also if the filter is not enabled
  */
  static bool isDisposable(const IPixMapSym& pms,
                           const SymFilterCache& sfc) noexcept;

  explicit UnreadableSymsFilter(
      std::unique_ptr<ISymFilter> nextFilter_ = nullptr) noexcept(!UT);
};

}  // namespace pic2sym::syms::inline filter

#endif  // H_UNREADABLE_SYMS_FILTER
