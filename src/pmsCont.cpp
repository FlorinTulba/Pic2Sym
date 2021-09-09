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

#include "pmsCont.h"

#include "bulkySymsFilter.h"
#include "controllerBase.h"
#include "filledRectanglesFilter.h"
#include "gridBarsFilter.h"
#include "pixMapSym.h"
#include "sievesSymsFilter.h"
#include "symFilterCache.h"
#include "unreadableSymsFilter.h"
#include "warnings.h"

#pragma warning(push, 0)

#include <numeric>

#pragma warning(pop)

using namespace std;
using namespace cv;
using namespace gsl;

extern template class optional<unsigned>;

namespace pic2sym {

extern const bool PreserveRemovableSymbolsForExamination;
extern const double PmsCont_SMALL_GLYPHS_PERCENT;

namespace syms {

PmsCont::PmsCont(IController& ctrler_) noexcept
    : ctrler(&ctrler_),

      // Add any additional filters as 'make_unique<NewFilter>()' in the last
      // set of unfilled '()'
      symFilter(make_unique<FilledRectanglesFilter>(make_unique<GridBarsFilter>(
          make_unique<BulkySymsFilter>(make_unique<UnreadableSymsFilter>(
              make_unique<SievesSymsFilter>()))))) {}

#pragma warning(disable : WARN_THROWS_ALTHOUGH_NOEXCEPT)
unsigned PmsCont::getFontSz() const noexcept(!UT) {
  EXPECTS_OR_REPORT_AND_THROW_CONST_MSG(
      ready, logic_error,
      HERE.function_name() + " cannot be called before setAsReady()"s);

  return fontSz;
}
#pragma warning(default : WARN_THROWS_ALTHOUGH_NOEXCEPT)

#pragma warning(disable : WARN_THROWS_ALTHOUGH_NOEXCEPT)
unsigned PmsCont::getBlanksCount() const noexcept(!UT) {
  EXPECTS_OR_REPORT_AND_THROW_CONST_MSG(
      ready, logic_error,
      HERE.function_name() + " cannot be called before setAsReady()"s);

  return blanks;
}
#pragma warning(default : WARN_THROWS_ALTHOUGH_NOEXCEPT)

#pragma warning(disable : WARN_THROWS_ALTHOUGH_NOEXCEPT)
unsigned PmsCont::getDuplicatesCount() const noexcept(!UT) {
  EXPECTS_OR_REPORT_AND_THROW_CONST_MSG(
      ready, logic_error,
      HERE.function_name() + " cannot be called before setAsReady()"s);

  return duplicates;
}
#pragma warning(default : WARN_THROWS_ALTHOUGH_NOEXCEPT)

#pragma warning(disable : WARN_THROWS_ALTHOUGH_NOEXCEPT)
const unordered_map<unsigned, unsigned>& PmsCont::getRemovableSymsByCateg()
    const noexcept(!UT) {
  EXPECTS_OR_REPORT_AND_THROW_CONST_MSG(
      ready, logic_error,
      HERE.function_name() + " cannot be called before setAsReady()"s);

  return removableSymsByCateg;
}
#pragma warning(default : WARN_THROWS_ALTHOUGH_NOEXCEPT)

#pragma warning(disable : WARN_THROWS_ALTHOUGH_NOEXCEPT)
double PmsCont::getCoverageOfSmallGlyphs() const noexcept(!UT) {
  EXPECTS_OR_REPORT_AND_THROW_CONST_MSG(
      ready, logic_error,
      HERE.function_name() + " cannot be called before setAsReady()"s);

  return coverageOfSmallGlyphs;
}
#pragma warning(default : WARN_THROWS_ALTHOUGH_NOEXCEPT)

#pragma warning(disable : WARN_THROWS_ALTHOUGH_NOEXCEPT)
const VPixMapSym& PmsCont::getSyms() const noexcept(!UT) {
  EXPECTS_OR_REPORT_AND_THROW_CONST_MSG(
      ready, logic_error,
      HERE.function_name() + " cannot be called before setAsReady()"s);

  return syms;
}
#pragma warning(default : WARN_THROWS_ALTHOUGH_NOEXCEPT)

void PmsCont::reset(unsigned fontSz_ /* = 0U*/,
                    unsigned symsCount /* = 0U*/) noexcept {
  ready = false;
  fontSz = fontSz_;
  maxGlyphSum = 255. * fontSz_ * fontSz_;
  blanks = duplicates = 0U;
  coverageOfSmallGlyphs = 0.;

  removableSymsByCateg.clear();
  syms.clear();
  if (symsCount)
    syms.reserve(symsCount);

  // max ensures consec and revConsec aren't empty when fontSz_ is 0
  // Transposing revConsec below throws for an empty matrix

  // 0. parameter prevents using initializer_list ctor of Mat
  consec = Mat{1, max((int)fontSz_, 1), CV_64FC1, 0.};
  revConsec.release();

  iota(BOUNDS_FOR_ITEM_TYPE(consec, double), (double)0.);
  flip(consec, revConsec, 1);
  revConsec = revConsec.t();
}

bool PmsCont::exactBlank(unsigned height, unsigned width) noexcept {
  if (height > 0U && width > 0U)
    return false;

  ++blanks;
  return true;
}

bool PmsCont::nearBlank(const IPixMapSym& pms) noexcept {
  if (pms.getAvgPixVal() > Eps && pms.getAvgPixVal() < OneMinusEps)
    return false;

  ++blanks;
  return true;
}

bool PmsCont::isDuplicate(const IPixMapSym& pms) noexcept {
  for (const unique_ptr<const IPixMapSym>& prevPms : syms)
    if (dynamic_cast<const PixMapSym&>(pms) ==
        dynamic_cast<const PixMapSym&>(*prevPms)) {
      ++duplicates;
      return true;
    }
  return false;
}

#pragma warning(disable : WARN_THROWS_ALTHOUGH_NOEXCEPT)
void PmsCont::appendSym(FT_ULong c,
                        size_t symIdx,
                        FT_GlyphSlot g,
                        FT_BBox& bb,
                        SymFilterCache& sfc) noexcept(!UT) {
  EXPECTS_OR_REPORT_AND_THROW_CONST_MSG(
      !ready, logic_error,
      HERE.function_name() +
          " cannot be called after setAsReady(), "
          "unless a reset() clears the container!"s);

  const FT_Bitmap b{g->bitmap};
  const unsigned height{b.rows};
  const unsigned width{b.width};

  // Skip Space characters
  if (exactBlank(height, width))
    return;

  PixMapSym pms{
      c,           symIdx,      g->bitmap, g->bitmap_left, g->bitmap_top,
      (int)fontSz, maxGlyphSum, consec,    revConsec,      bb};
  // discard disguised Space characters
  if (nearBlank(pms))
    return;

  // Exclude duplicates, as well
  if (isDuplicate(pms))
    return;

  sfc.setBoundingBox(height, width);

  const optional<unsigned> matchingFilterId =
      symFilter->matchingFilterId(pms, sfc);
  if (matchingFilterId) {
    if (auto it = removableSymsByCateg.find(*matchingFilterId);
        it == removableSymsByCateg.end())
      removableSymsByCateg.emplace(*matchingFilterId, 1U);
    else
      ++it->second;

    if (!PreserveRemovableSymbolsForExamination)
      return;

    pms.setRemovable();
  }

  syms.emplace_back(make_unique<const PixMapSym>(pms));

#ifndef UNIT_TESTING
  ctrler->display1stPageIfFull(syms);
#endif  // UNIT_TESTING not defined
}
#pragma warning(default : WARN_THROWS_ALTHOUGH_NOEXCEPT)

void PmsCont::setAsReady() noexcept {
  if (ready)
    return;

  // Determine below max box coverage for smallest glyphs from the kept symsSet.
  // This will be used to favor using larger glyphs when this option is
  // selected.
  const auto smallGlyphsQty =
      narrow_cast<long>(round(size(syms) * PmsCont_SMALL_GLYPHS_PERCENT));

  VPixMapSym::iterator itToNthGlyphSum{next(begin(syms), smallGlyphsQty)};
  nth_element(begin(syms), itToNthGlyphSum, end(syms),
              [](const unique_ptr<const IPixMapSym>& first,
                 const unique_ptr<const IPixMapSym>& second) noexcept {
                return first->getAvgPixVal() < second->getAvgPixVal();
              });

  coverageOfSmallGlyphs = (*itToNthGlyphSum)->getAvgPixVal();

  ready = true;
}

}  // namespace syms
}  // namespace pic2sym
