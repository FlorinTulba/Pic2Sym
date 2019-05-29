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

#include "precompiled.h"

#include "fontEngine.h"
#include "fontErrorsHelper.h"
#include "pixMapSym.h"
#include "pmsContBase.h"
#include "symsLoadingFailure.h"
#include "tinySym.h"
#include "tinySymsDataSerialization.h"

#pragma warning(push, 0)

#include <numeric>

#include <filesystem>
#include FT_TRUETYPE_IDS_H

#pragma warning(pop)

using namespace std;
using namespace cv;
using namespace std::filesystem;

extern unsigned TinySymsSz();

#ifndef UNIT_TESTING
// isTinySymsDataSavedOnDisk and getTinySyms will have different implementations
// in UnitTesting

#include "appStart.h"

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

#endif  // UNIT_TESTING not defined

#pragma warning(disable : WARN_THROWS_ALTHOUGH_NOEXCEPT)
const VTinySyms& FontEngine::getTinySyms() noexcept(!UT) {
  /*
  Making sure the generation of small symbols doesn't overlap with filling
  symsCont with normal symbols Both involve requesting fonts of different sizes
  from the same font 'library' object and also operating on the same 'face'
  object while considering the requests.
  */
  if (!symsCont->isReady())
    THROW_WITH_CONST_MSG(
        __FUNCTION__ " should be called only after setAsReady()", logic_error);

  if (tinySyms.empty()) {
    VTinySymsIO tinySymsDataSerializer(tinySyms);
    path tinySymsDataFile;
    if (!FontEngine::isTinySymsDataSavedOnDisk(getFontType(),
                                               tinySymsDataFile) ||
        !tinySymsDataSerializer.loadFrom(tinySymsDataFile.string())) {
      static const unsigned TinySymsSize = TinySymsSz();

      /*
      Instead of requesting directly fonts of size TinySymsSize, fonts of a
      larger size are loaded, then resized to TinySymsSize, so to get more
      precise approximations (non-zero fractional parts) and to avoid hinting
      that increases for small font sizes.
      */
      static const unsigned RefSymsSize = TinySymsSize * ITinySym::RatioRefTiny,
                            RefSymsSizeX64 = RefSymsSize << 6;
      static const double RefSymsSizeD = (double)RefSymsSize,
                          maxGlyphSum = 255. * RefSymsSize * RefSymsSize;

      static Mat consec(1, (int)RefSymsSize, CV_64FC1), revConsec;

      if (revConsec.empty()) {
        iota(BOUNDS_FOR_ITEM_TYPE(consec, double), (double)0.);
        flip(consec, revConsec, 1);
        revConsec = revConsec.t();
      }

      FT_BBox bbox;
      double factorH, factorV;
      adjustScaling(RefSymsSize, bbox, factorH, factorV);
      tinySyms.reserve(symsCount);
      const double szHd = factorH * RefSymsSizeX64,
                   szVd = factorV * RefSymsSizeX64;
      const FT_Long szH = (FT_Long)floor(szHd), szV = (FT_Long)floor(szVd);

      size_t i = 0ULL, countOfSymsUnableToLoad = 0ULL;
      FT_UInt idx = 0U;
      FT_Size_RequestRec req;
      req.type = FT_SIZE_REQUEST_TYPE_REAL_DIM;
      req.horiResolution = req.vertResolution = 72U;
      for (FT_ULong c = FT_Get_First_Char(face, &idx); idx != 0;
           c = FT_Get_Next_Char(face, c, &idx), ++i) {
        FT_Error error = FT_Err_Ok;
        if (error = FT_Load_Char(face, c, FT_LOAD_RENDER); error != FT_Err_Ok) {
          if (symsUnableToLoad.find(c) !=
              symsUnableToLoad.end()) {  // known glyph
            ++countOfSymsUnableToLoad;

            // Insert a blank for this symbol
            tinySyms.emplace_back((unsigned long)c, i);
            continue;
          } else  // unexpected glyph
            THROW_WITH_VAR_MSG("Couldn't load an unexpected glyph (" +
                                   to_string(c) +
                                   ") during initial resizing. Error: " +
                                   FtErrors[(size_t)error],
                               TinySymsLoadingFailure);
        }
        FT_GlyphSlot g = face->glyph;
        FT_Bitmap b = g->bitmap;
        const unsigned height = b.rows, width = b.width;
        if (height == 0U || width == 0U) {
          // Blank whose symbol code and index are provided
          tinySyms.emplace_back((unsigned long)c, i);
          continue;
        }

        if (width > RefSymsSize || height > RefSymsSize) {
          // Adjust font size to fit the RefSymsSize x RefSymsSize square
          req.height = (FT_Long)floor(szVd / max(1., height / RefSymsSizeD));
          req.width = (FT_Long)floor(szHd / max(1., width / RefSymsSizeD));
          if (error = FT_Request_Size(face, &req); error != FT_Err_Ok)
            THROW_WITH_VAR_MSG(
                "Couldn't set font size: " + to_string(req.height) + " x " +
                    to_string(req.width) +
                    "  Error: " + FtErrors[(size_t)error],
                TinySymsLoadingFailure);

          if (error = FT_Load_Char(face, c, FT_LOAD_RENDER); error != FT_Err_Ok)
            THROW_WITH_VAR_MSG("Couldn't load glyph: " + to_string(c) +
                                   " for its second resizing. Error: " +
                                   FtErrors[(size_t)error],
                               TinySymsLoadingFailure);
          g = face->glyph;
          b = g->bitmap;

          // Restore font size
          req.height = szV;
          req.width = szH;
          if (error = FT_Request_Size(face, &req); error != FT_Err_Ok)
            THROW_WITH_VAR_MSG(
                "Couldn't set font size: " + to_string(req.height) + " x " +
                    to_string(req.width) +
                    "  Error: " + FtErrors[(size_t)error],
                TinySymsLoadingFailure);
        }
        const FT_Int left = g->bitmap_left, top = g->bitmap_top;
        const PixMapSym refSym((unsigned long)c, i, b, (int)left, (int)top,
                               (int)RefSymsSize, maxGlyphSum, consec, revConsec,
                               bbox);
        tinySyms.emplace_back(refSym);
      }

      if (countOfSymsUnableToLoad < symsUnableToLoad.size())
        THROW_WITH_VAR_MSG(
            "Initial resizing of the glyphs found only " +
                to_string(countOfSymsUnableToLoad) +
                " symbols that couldn't be loaded when expecting " +
                to_string(symsUnableToLoad.size()),
            TinySymsLoadingFailure);

      tinySymsDataSerializer.saveTo(tinySymsDataFile.string());
    }
  }

  return tinySyms;
}
#pragma warning(default : WARN_THROWS_ALTHOUGH_NOEXCEPT)

void FontEngine::disposeTinySyms() noexcept {
  tinySyms.clear();
}
