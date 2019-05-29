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

#ifndef H_PMS_CONT_BASE
#define H_PMS_CONT_BASE

#include "pixMapSymBase.h"

#pragma warning(push, 0)

#include <unordered_map>

#include <ft2build.h>
#include FT_FREETYPE_H

#pragma warning(pop)

extern template class std::unordered_map<unsigned, unsigned>;

class SymFilterCache;  // forward declaration

/// Base for the container holding PixMapSym-s of same size
class IPmsCont /*abstract*/ {
 public:
  /// Is container ready to provide useful data?
  virtual bool isReady() const noexcept = 0;

  /**
  Bounding box size
  @throw logic_error if not called after setAsReady()

  Exception to be only reported, not handled
  */
  virtual unsigned getFontSz() const noexcept(!UT) = 0;

  /**
  How many Blank characters were within the charmap
  @throw logic_error if not called after setAsReady()

  Exception to be only reported, not handled
  */
  virtual unsigned getBlanksCount() const noexcept(!UT) = 0;

  /**
  How many duplicate symbols were within the charmap
  @throw logic_error if not called after setAsReady()

  Exception to be only reported, not handled
  */
  virtual unsigned getDuplicatesCount() const noexcept(!UT) = 0;

  /**
  Associations: filterId - count of detected syms
  @throw logic_error if not called after setAsReady()

  Exception to be only reported, not handled
  */
  virtual const std::unordered_map<unsigned, unsigned>&
  getRemovableSymsByCateg() const noexcept(!UT) = 0;

  /**
  Max ratio for small symbols of glyph area / containing area
  @throw logic_error if not called after setAsReady()

  Exception to be only reported, not handled
  */
  virtual double getCoverageOfSmallGlyphs() const noexcept(!UT) = 0;

  /**
  Data for each symbol within current charmap
  @throw logic_error if not called after setAsReady()

  Exception to be only reported, not handled
  */
  virtual const VPixMapSym& getSyms() const noexcept(!UT) = 0;

  /// Clears & prepares container for new entries
  virtual void reset(unsigned fontSz_ = 0U,
                     unsigned symsCount = 0U) noexcept = 0;

  /**
  appendSym puts valid glyphs into vector 'syms'.

  Space (empty / full) glyphs are invalid.
  Also updates the count of blanks & duplicates and of any filtered out symbols.

  @throw logic_error if called after setAsReady()

  Exception to be only reported, not handled
  */
  virtual void appendSym(FT_ULong c,
                         size_t symIdx,
                         FT_GlyphSlot g,
                         FT_BBox& bb,
                         SymFilterCache& sfc) noexcept(!UT) = 0;

  /// No other symbols to append. Statistics can be now computed
  virtual void setAsReady() noexcept = 0;

  virtual ~IPmsCont() noexcept {}

  // Slicing prevention
  IPmsCont(const IPmsCont&) = delete;
  IPmsCont(IPmsCont&&) = delete;
  IPmsCont& operator=(const IPmsCont&) = delete;
  IPmsCont& operator=(IPmsCont&&) = delete;

 protected:
  constexpr IPmsCont() noexcept {}
};

#endif  // H_PMS_CONT_BASE
