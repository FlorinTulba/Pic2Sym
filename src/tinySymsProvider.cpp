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

#include "tinySymsProvider.h"

#include "fontEngine.h"
#include "fontErrorsHelper.h"
#include "pixMapSym.h"
#include "pmsContBase.h"
#include "symsChangeIssues.h"
#include "tinySym.h"
#include "tinySymsDataSerialization.h"

#pragma warning(push, 0)

#include <filesystem>
#include <numeric>

#include FT_TRUETYPE_IDS_H

#pragma warning(pop)

using namespace std;
using namespace cv;
using namespace std::filesystem;

namespace pic2sym {

extern unsigned TinySymsSz();

}  // namespace pic2sym

#ifndef UNIT_TESTING
// isTinySymsDataSavedOnDisk and getTinySyms will have different implementations
// in UnitTesting

#include "appStart.h"

namespace pic2sym::syms {

bool FontEngine::isTinySymsDataSavedOnDisk(
    const string& fontType,
    std::filesystem::path& tinySymsDataFile) noexcept {
  if (fontType.empty())
    return false;

  tinySymsDataFile = AppStart::dir();
  if (!exists(tinySymsDataFile.append("TinySymsDataSets"))) {
    create_directory(tinySymsDataFile);
    return false;
  }

  tinySymsDataFile.append(fontType)
      .concat("_")
      .concat(to_string(TinySymsSz()))
      .concat(".tsd");  // Tiny Symbols Data => tsd

  return exists(tinySymsDataFile);
}

}  // namespace pic2sym::syms

#endif  // UNIT_TESTING not defined

namespace pic2sym::syms {

#pragma warning(disable : WARN_THROWS_ALTHOUGH_NOEXCEPT)
const VTinySyms& FontEngine::getTinySyms() noexcept(!UT) {
  /*
  Making sure the generation of small symbols doesn't overlap with filling
  symsCont with normal symbols Both involve requesting fonts of different sizes
  from the same font 'library' object and also operating on the same 'face'
  object while considering the requests.
  */
  EXPECTS_OR_REPORT_AND_THROW_CONST_MSG(
      symsCont->isReady(), logic_error,
      HERE.function_name() + " should be called only after setAsReady()"s);

  if (tinySyms.empty()) {
    VTinySymsIO tinySymsDataSerializer{tinySyms};
    path tinySymsDataFile;
    if (!FontEngine::isTinySymsDataSavedOnDisk(getFontType(),
                                               tinySymsDataFile) ||
        !tinySymsDataSerializer.loadFrom(tinySymsDataFile.string())) {
      static const unsigned TinySymsSize{TinySymsSz()};

      /*
      Instead of requesting directly fonts of size TinySymsSize, fonts of a
      larger size are loaded, then resized to TinySymsSize, so to get more
      precise approximations (non-zero fractional parts) and to avoid hinting
      that increases for small font sizes.
      */
      static const unsigned RefSymsSize{TinySymsSize * ITinySym::RatioRefTiny};
      static const unsigned RefSymsSizeX64{RefSymsSize << 6};
      static const double RefSymsSizeD{(double)RefSymsSize};
      static const double maxGlyphSum{255. * RefSymsSize * RefSymsSize};

      // 0. parameter prevents using initializer_list ctor of Mat
      static Mat consec{1, (int)RefSymsSize, CV_64FC1, 0.};
      static Mat revConsec;

      if (revConsec.empty()) {
        iota(BOUNDS_FOR_ITEM_TYPE(consec, double), (double)0.);
        flip(consec, revConsec, 1);
        revConsec = revConsec.t();
      }

      FT_BBox bbox;
      double factorH, factorV;
      adjustScaling(RefSymsSize, bbox, factorH, factorV);
      tinySyms.reserve(symsCount);
      const double szHd{factorH * RefSymsSizeX64};
      const double szVd{factorV * RefSymsSizeX64};
      const FT_Long szH{(FT_Long)floor(szHd)};
      const FT_Long szV{(FT_Long)floor(szVd)};

      size_t i{};
      size_t countOfSymsUnableToLoad{};
      FT_UInt idx{};
      FT_Size_RequestRec req;
      req.type = FT_SIZE_REQUEST_TYPE_REAL_DIM;
      req.horiResolution = req.vertResolution = 72U;
      for (FT_ULong c{FT_Get_First_Char(face, &idx)}; idx;
           c = FT_Get_Next_Char(face, c, &idx), ++i) {
        FT_Error error{FT_Err_Ok};
        if (error = FT_Load_Char(face, c, FT_LOAD_RENDER); error != FT_Err_Ok) {
          if (symsUnableToLoad.contains(c)) {  // known glyph
            ++countOfSymsUnableToLoad;

            // Insert a blank for this symbol
            tinySyms.emplace_back((unsigned long)c, i);
            continue;
          } else  // unexpected glyph
            reportAndThrow<TinySymsLoadingFailure>(
                "Couldn't load an unexpected glyph (" + to_string(c) +
                ") during initial resizing. Error: " + FtErrors[(size_t)error]);
        }
        FT_GlyphSlot g{face->glyph};
        FT_Bitmap b{g->bitmap};
        const unsigned height{b.rows};
        const unsigned width{b.width};
        if (!height || !width) {
          // Blank whose symbol code and index are provided
          tinySyms.emplace_back((unsigned long)c, i);
          continue;
        }

        if (width > RefSymsSize || height > RefSymsSize) {
          // Adjust font size to fit the RefSymsSize x RefSymsSize square
          req.height = (FT_Long)floor(szVd / max(1., height / RefSymsSizeD));
          req.width = (FT_Long)floor(szHd / max(1., width / RefSymsSizeD));
          if (error = FT_Request_Size(face, &req); error != FT_Err_Ok)
            reportAndThrow<TinySymsLoadingFailure>(
                "Couldn't set font size: " + to_string(req.height) + " x " +
                to_string(req.width) + "  Error: " + FtErrors[(size_t)error]);

          if (error = FT_Load_Char(face, c, FT_LOAD_RENDER); error != FT_Err_Ok)
            reportAndThrow<TinySymsLoadingFailure>(
                "Couldn't load glyph: " + to_string(c) +
                " for its second resizing. Error: " + FtErrors[(size_t)error]);
          g = face->glyph;
          b = g->bitmap;

          // Restore font size
          req.height = szV;
          req.width = szH;
          if (error = FT_Request_Size(face, &req); error != FT_Err_Ok)
            reportAndThrow<TinySymsLoadingFailure>(
                "Couldn't set font size: " + to_string(req.height) + " x " +
                to_string(req.width) + "  Error: " + FtErrors[(size_t)error]);
        }
        const FT_Int left{g->bitmap_left};
        const FT_Int top{g->bitmap_top};
        const PixMapSym refSym{
            (unsigned long)c, i,           b,      (int)left, (int)top,
            (int)RefSymsSize, maxGlyphSum, consec, revConsec, bbox};
        tinySyms.emplace_back(refSym);
      }

      if (countOfSymsUnableToLoad < size(symsUnableToLoad))
        reportAndThrow<TinySymsLoadingFailure>(
            "Initial resizing of the glyphs found only " +
            to_string(countOfSymsUnableToLoad) +
            " symbols that couldn't be loaded when expecting " +
            to_string(size(symsUnableToLoad)));

      tinySymsDataSerializer.saveTo(tinySymsDataFile.string());
    }
  }

  return tinySyms;
}
#pragma warning(default : WARN_THROWS_ALTHOUGH_NOEXCEPT)

void FontEngine::disposeTinySyms() noexcept {
  tinySyms.clear();
}

}  // namespace pic2sym::syms
