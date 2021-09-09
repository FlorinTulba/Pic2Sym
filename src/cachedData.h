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

#ifndef H_CACHED_DATA
#define H_CACHED_DATA

#include "fontEngineBase.h"
#include "misc.h"

#pragma warning(push, 0)

#include <numbers>

#include <opencv2/core/core.hpp>

#pragma warning(pop)

namespace pic2sym::transform {

/// Cached data for computing match parameters and evaluating match aspects
class CachedData {
 public:
  explicit CachedData(bool forTinySyms_ = false) noexcept;
  virtual ~CachedData() noexcept = default;

  CachedData(const CachedData&) noexcept = default;
  CachedData(CachedData&&) noexcept = default;

  // 'forTinySyms' is supposed to remain the same for original / copy
  void operator=(const CachedData&) = delete;
  void operator=(CachedData&&) = delete;

  // Getters which need to be fast, so inline,
  // instead of virtual realizations of a read-only interface of this cached
  // information
  const cv::Mat& getConsec() const noexcept { return consec; }
  double getSz_1() const noexcept { return sz_1; }
  double getSzSq() const noexcept { return szSq; }
  double getSmallGlyphsCoverage() const noexcept { return smallGlyphsCoverage; }

  /// Constants about maximum standard deviations for foreground/background or
  /// edges
  struct MaxSdev {
    /**
    Max possible std dev = 127.5  for foreground / background.
    Happens for an error matrix with a histogram with 2 equally large bins on 0
    and 255. In that case, the mean is 127.5 and the std dev is: sqrt(
    ((-127.5)^2 * sz^2/2 + 127.5^2 * sz^2/2) /sz^2) = 127.5
    */
    static constexpr double forFgOrBg{127.5};

    /**
    Max possible std dev for edge is 255.
    This happens in the following situation:
    a) Foreground and background masks cover an empty area of the patch =>
    approximated patch will be completely black
    b) Edge mask covers a full brightness (255) area of the patch =>
    every pixel from the patch covered by the edge mask has a deviation of 255
    from the corresponding zone within the approximated patch.
    */
    static constexpr double forEdges{255.};
  };

  /// Constants for computations concerning mass centers
  struct MassCenters {
    /// acceptable distance between mass centers (1/8)
    static constexpr double preferredMaxMcDist{.125};

    /// The center of a square with unit-length sides
    static const cv::Point2d& unitSquareCenter() noexcept;

    /// 1 / max possible distance between mass centers: sqrt(2) -
    /// preferredMaxMcDist
    static constexpr double invComplPrefMaxMcDist{
        1. / (std::numbers::sqrt2 - preferredMaxMcDist)};

    // See comment from above the definitions of these static methods in
    // cachedData.cpp, but also from DirectionalSmoothness::score

    /// mcsOffsetFactor = a * mcsOffset + b
    static const double a_mcsOffsetFactor() noexcept;

    /// mcsOffsetFactor = a * mcsOffset + b
    static const double b_mcsOffsetFactor() noexcept;
  };

  PROTECTED :

      cv::Mat consec;  ///< row matrix with consecutive elements: 0..sz-1
  double sz_1{};       ///< double version of sz - 1
  double szSq{};       ///< double version of sz^2

  /// Max density for symbols considered small
  double smallGlyphsCoverage{};

 public:
  /// Are all these values used for tiny symbols or for normal symbols?
  bool forTinySyms;
};

/// CachedData with modifiers
class CachedDataRW : public CachedData {
 public:
  explicit CachedDataRW(bool forTinySyms_ = false) noexcept;

  CachedDataRW(const CachedDataRW&) noexcept = default;
  CachedDataRW(CachedDataRW&&) noexcept = default;

  // 'forTinySyms' is supposed to remain the same for original / copy
  void operator=(const CachedDataRW&) = delete;
  void operator=(CachedDataRW&&) = delete;

  void update(unsigned sz_, const syms::IFontEngine& fe_) noexcept;
  void update(const syms::IFontEngine& fe_) noexcept;
  void useNewSymSize(unsigned sz_) noexcept;
};

}  // namespace pic2sym::transform

#endif  // H_CACHED_DATA
