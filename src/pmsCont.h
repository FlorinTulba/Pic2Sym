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

#ifndef H_PMS_CONT
#define H_PMS_CONT

#include "pmsContBase.h"

#include "controllerBase.h"
#include "pixMapSymBase.h"
#include "symFilterBase.h"

namespace pic2sym::syms {

/// Convenience container to hold PixMapSym-s of same size
class PmsCont : public IPmsCont {
 public:
  explicit PmsCont(IController& ctrler_) noexcept;

  // Slicing prevention
  PmsCont(const PmsCont&) = delete;
  PmsCont(PmsCont&&) = delete;
  void operator=(const PmsCont&) = delete;
  void operator=(PmsCont&&) = delete;

  /// Is container ready to provide useful data?
  bool isReady() const noexcept override { return ready; }

  /**
  Bounding box size
  @throw logic_error if not called after setAsReady()

  Exception to be only reported, not handled
  */
  unsigned getFontSz() const noexcept(!UT) override;

  /**
  How many Blank characters were within the charmap
  @throw logic_error if not called after setAsReady()

  Exception to be only reported, not handled
  */
  unsigned getBlanksCount() const noexcept(!UT) override;

  /**
  How many duplicate symbols were within the charmap
  @throw logic_error if not called after setAsReady()

  Exception to be only reported, not handled
  */
  unsigned getDuplicatesCount() const noexcept(!UT) override;

  /**
  Associations: filterId - count of detected syms
  @throw logic_error if not called after setAsReady()

  Exception to be only reported, not handled
  */
  const std::unordered_map<unsigned, unsigned>& getRemovableSymsByCateg() const
      noexcept(!UT) override;

  /**
  Max ratio for small symbols of glyph area / containing area
  @throw logic_error if not called after setAsReady()

  Exception to be only reported, not handled
  */
  double getCoverageOfSmallGlyphs() const noexcept(!UT) override;

  /**
  Data for each symbol within current charmap
  @throw logic_error if not called after setAsReady()

  Exception to be only reported, not handled
  */
  const VPixMapSym& getSyms() const noexcept(!UT) override;

  /// Clears & prepares container for new entries
  void reset(unsigned fontSz_ = 0U, unsigned symsCount = 0U) noexcept override;

  /**
  appendSym puts valid glyphs into vector 'syms'.

  Space (empty / full) glyphs are invalid.
  Also updates the count of blanks & duplicates and of any filtered out symbols.

  @throw logic_error if called after setAsReady()

  Exception to be only reported, not handled
  */
  void appendSym(FT_ULong c,
                 size_t symIdx,
                 FT_GlyphSlot g,
                 FT_BBox& bb,
                 filter::SymFilterCache& sfc) noexcept(!UT) override;

  /// No other symbols to append. Statistics can be now computed
  void setAsReady() noexcept override;

 protected:
  /// If a symbol has a side equal to 0, it is a blank => increment blanks and
  /// return true
  bool exactBlank(unsigned height, unsigned width) noexcept;

  /// If a symbol contains almost only white pixels it is nearly blank =>
  /// increment blanks and return true
  bool nearBlank(const IPixMapSym& pms) noexcept;

  /// Is the symbol an exact duplicate of one which is already in the syms
  /// container? If yes, increment duplicates and return true
  bool isDuplicate(const IPixMapSym& pms) noexcept;

 private:
  VPixMapSym syms;  ///< data for each symbol within current charmap

  // Precomputed entities during reset
  cv::Mat consec;     ///< vector of consecutive values 0..fontSz-1
  cv::Mat revConsec;  ///< consec reversed

  /// Updates Cmap View as soon as there are enough symbols for 1 page
  gsl::not_null<IController*> ctrler;

  /// Member that allows setting filters to detect symbols with undesired
  /// features.
  std::unique_ptr<ISymFilter> symFilter{new DefSymFilter};

  /// Associations: filterId - count of detected syms
  std::unordered_map<unsigned, unsigned> removableSymsByCateg;

  /// Max sum of a glyph's pixels
  double maxGlyphSum{};

  /// Max ratio for small symbols of glyph area / containing area
  double coverageOfSmallGlyphs{};

  unsigned fontSz{};  ///< bounding box size

  unsigned blanks{};  ///< how many Blank characters were within the charmap

  /// How many duplicate symbols were within the charmap
  unsigned duplicates{};

  bool ready{false};  ///< is container ready to provide useful data?
};

}  // namespace pic2sym::syms

#endif  // H_PMS_CONT
