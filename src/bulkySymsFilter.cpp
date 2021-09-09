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

#include "bulkySymsFilter.h"

#include "pixMapSymBase.h"
#include "symFilterCache.h"

#pragma warning(push, 0)

#include <opencv2/imgproc/imgproc.hpp>

#pragma warning(pop)

using namespace std;
using namespace cv;

namespace pic2sym {

extern const unsigned Settings_MAX_FONT_SIZE;

SYM_FILTER_DEFINE_IS_ENABLED(BulkySymsFilter)

namespace syms::inline filter {

namespace {

/// Erosion mask size: max(3, (fontSz/2 or the next odd value))
constexpr int compErMaskSide(unsigned fontSz) noexcept {
  return max(3, (((int)fontSz / 2) | 1));
}

/*
/// Closing mask size: max(3, (fontSz/6 or next odd value))
constexpr int compCloseMaskSide(unsigned fontSz) noexcept {
  return max(3, (((int)fontSz / 6) | 1));
};
*/

}  // anonymous namespace

BulkySymsFilter::BulkySymsFilter(
    unique_ptr<ISymFilter> nextFilter_ /* = nullptr*/) noexcept
    : TSymFilter{2U, "bulky symbols", move(nextFilter_)} {}

bool BulkySymsFilter::isDisposable(const IPixMapSym& pms,
                                   const SymFilterCache& sfc) noexcept {
  if (!isEnabled())
    return false;

  static unordered_map<int, Mat> circleMasks;

  if (min(pms.getRows(), pms.getCols()) <
      (unsigned)compErMaskSide(sfc.getSzU()))
    return false;

  if (circleMasks.empty()) {
    for (int maskSide{3}, maxMaskSide{compErMaskSide(Settings_MAX_FONT_SIZE)};
         maskSide <= maxMaskSide; maskSide += 2)
      circleMasks[maskSide] =
          getStructuringElement(MORPH_ELLIPSE, Size(maskSide, maskSide));
  }

  const Mat narrowGlyph{pms.asNarrowMat()};
  Mat processed;

  /*
  // Close with a small disk to fill any minor gaps.
  static const Point defAnchor{-1, -1};
  morphologyEx(narrowGlyph, processed, MORPH_CLOSE,
               circleMasks[compCloseMaskSide(sfc.getSzU())], defAnchor, 1,
               BORDER_CONSTANT, Scalar{});
  */

  // Erode with a large disk to detect large filled areas.
  erode(narrowGlyph, processed, circleMasks[compErMaskSide(sfc.getSzU())]);

  const bool result{countNonZero(processed > 45) > 0};
  return result;
}

}  // namespace syms::inline filter
}  // namespace pic2sym
