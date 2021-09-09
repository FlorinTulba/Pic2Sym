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

#ifndef H_TINY_SYMS_PROVIDER
#define H_TINY_SYMS_PROVIDER

#include "tinySym.h"

namespace pic2sym::syms {

/// Allows providing tiny symbols both from FontEngine and from UnitTesting
class ITinySymsProvider /*abstract*/ {
 public:
  /**
  (Creates/Loads and) Returns(/Saves) the small versions of all the symbols from
  current cmap

  @return a list of tiny symbols from current cmap

  @throw logic_error if called before setAsReady()
  @throw TinySymsLoadingFailure for problems loading/processing tiny symbols,
  which might be difficult to handle (they belong to the FreeType project)

  Exceptions to be only reported, not handled
  */
  virtual const VTinySyms& getTinySyms() noexcept(!UT) = 0;

  /// Releases tiny symbols data
  virtual void disposeTinySyms() noexcept = 0;

  virtual ~ITinySymsProvider() noexcept = 0 {}
};

}  // namespace pic2sym::syms

#endif  // H_TINY_SYMS_PROVIDER
