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

#ifndef H_MATCH_ASPECTS
#define H_MATCH_ASPECTS

#include "match.h"

#include "misc.h"

namespace pic2sym::match {

/// Selecting a symbol with the scene underneath it as uniform as possible
class FgMatch : public MatchAspect {
 public:
  ~FgMatch() noexcept override = default;

  FgMatch(const FgMatch&) noexcept = default;
  FgMatch(FgMatch&&) noexcept = default;

  // 'k' reference is supposed not to change for the original / copy
  void operator=(const FgMatch&) = delete;
  void operator=(FgMatch&&) = delete;

  /// Providing a clue about how complex is this MatchAspect compared to the
  /// others
  double relativeComplexity() const noexcept override;

  PROTECTED :

      explicit FgMatch(const cfg::IMatchSettings& ms) noexcept;

  /// Defines the scoring rule, based on all required fields computed already in
  /// MatchParams mp
  double score(const IMatchParams& mp,
               const transform::CachedData& cachedData) const noexcept override;

  /// Prepares required fields from MatchParams mp to be able to assess the
  /// match
  void fillRequiredMatchParams(const cv::Mat& patch,
                               const syms::ISymData& symData,
                               const transform::CachedData& cachedData,
                               IMatchParamsRW& mp) const noexcept override;

  REGISTER_MATCH_ASPECT(FgMatch);
};

/// Aspect ensuring more uniform background scene around the selected symbol
class BgMatch : public MatchAspect {
 public:
  ~BgMatch() noexcept override = default;

  BgMatch(const BgMatch&) noexcept = default;
  BgMatch(BgMatch&&) noexcept = default;

  // 'k' reference is supposed not to change for the original / copy
  void operator=(const BgMatch&) = delete;
  void operator=(BgMatch&&) = delete;

  /// Providing a clue about how complex is this MatchAspect compared to the
  /// others
  double relativeComplexity() const noexcept override;

  PROTECTED :

      explicit BgMatch(const cfg::IMatchSettings& ms) noexcept;

  /// Defines the scoring rule, based on all required fields computed
  /// already in
  /// MatchParams mp
  double score(const IMatchParams& mp,
               const transform::CachedData& cachedData) const noexcept override;

  /// Prepares required fields from MatchParams mp to be able to assess the
  /// match
  void fillRequiredMatchParams(const cv::Mat& patch,
                               const syms::ISymData& symData,
                               const transform::CachedData& cachedData,
                               IMatchParamsRW& mp) const noexcept override;

  REGISTER_MATCH_ASPECT(BgMatch);
};

/// Aspect ensuring the edges of the selected symbol seem to appear also on the
/// patch
class EdgeMatch : public MatchAspect {
 public:
  ~EdgeMatch() noexcept override = default;

  EdgeMatch(const EdgeMatch&) noexcept = default;
  EdgeMatch(EdgeMatch&&) noexcept = default;

  // 'k' reference is supposed not to change for the original / copy
  void operator=(const EdgeMatch&) = delete;
  void operator=(EdgeMatch&&) = delete;

  /// Providing a clue about how complex is this MatchAspect compared to the
  /// others
  double relativeComplexity() const noexcept override;

  PROTECTED :

      explicit EdgeMatch(const cfg::IMatchSettings& ms) noexcept;

  /// Defines the scoring rule, based on all required fields computed already in
  /// MatchParams mp
  double score(const IMatchParams& mp,
               const transform::CachedData& cachedData) const noexcept override;

  /// Prepares required fields from MatchParams mp to be able to assess the
  /// match
  void fillRequiredMatchParams(const cv::Mat& patch,
                               const syms::ISymData& symData,
                               const transform::CachedData& cachedData,
                               IMatchParamsRW& mp) const noexcept override;

  REGISTER_MATCH_ASPECT(EdgeMatch);
};

/// Discouraging barely visible symbols
class BetterContrast : public MatchAspect {
 public:
  ~BetterContrast() noexcept override = default;

  BetterContrast(const BetterContrast&) noexcept = default;
  BetterContrast(BetterContrast&&) noexcept = default;

  // 'k' reference is supposed not to change for the original / copy
  void operator=(const BetterContrast&) = delete;
  void operator=(BetterContrast&&) = delete;

  /// Providing a clue about how complex is this MatchAspect compared to the
  /// others
  double relativeComplexity() const noexcept override;

  PROTECTED :

      explicit BetterContrast(const cfg::IMatchSettings& ms) noexcept;

  /// Defines the scoring rule, based on all required fields computed already in
  /// MatchParams mp
  double score(const IMatchParams& mp,
               const transform::CachedData& cachedData) const noexcept override;

  /// Prepares required fields from MatchParams mp to be able to assess the
  /// match
  void fillRequiredMatchParams(const cv::Mat& patch,
                               const syms::ISymData& symData,
                               const transform::CachedData& cachedData,
                               IMatchParamsRW& mp) const noexcept override;

  REGISTER_MATCH_ASPECT(BetterContrast);
};

/// Aspect concentrating on where's the center of gravity of the patch & its
/// approximation
class GravitationalSmoothness : public MatchAspect {
 public:
  ~GravitationalSmoothness() noexcept override = default;

  GravitationalSmoothness(const GravitationalSmoothness&) noexcept = default;
  GravitationalSmoothness(GravitationalSmoothness&&) noexcept = default;

  // 'k' reference is supposed not to change for the original / copy
  void operator=(const GravitationalSmoothness&) = delete;
  void operator=(GravitationalSmoothness&&) = delete;

  /// Providing a clue about how complex is this MatchAspect compared to the
  /// others
  double relativeComplexity() const noexcept override;

  PROTECTED :

      explicit GravitationalSmoothness(const cfg::IMatchSettings& ms) noexcept;

  /// Defines the scoring rule, based on all required fields computed already in
  /// MatchParams mp
  double score(const IMatchParams& mp,
               const transform::CachedData& cachedData) const noexcept override;

  /// Prepares required fields from MatchParams mp to be able to assess the
  /// match
  void fillRequiredMatchParams(const cv::Mat& patch,
                               const syms::ISymData& symData,
                               const transform::CachedData& cachedData,
                               IMatchParamsRW& mp) const noexcept override;

  REGISTER_MATCH_ASPECT(GravitationalSmoothness);
};

/// Aspect encouraging more accuracy while approximating the direction of the
/// patch
class DirectionalSmoothness : public MatchAspect {
 public:
  ~DirectionalSmoothness() noexcept override = default;

  DirectionalSmoothness(const DirectionalSmoothness&) noexcept = default;
  DirectionalSmoothness(DirectionalSmoothness&&) noexcept = default;

  // 'k' reference is supposed not to change for the original / copy
  void operator=(const DirectionalSmoothness&) = delete;
  void operator=(DirectionalSmoothness&&) = delete;

  /// Providing a clue about how complex is this MatchAspect compared to the
  /// others
  double relativeComplexity() const noexcept override;

  PROTECTED :

      explicit DirectionalSmoothness(const cfg::IMatchSettings& ms) noexcept;

  /// Defines the scoring rule, based on all required fields computed already in
  /// MatchParams mp
  double score(const IMatchParams& mp,
               const transform::CachedData& cachedData) const noexcept override;

  /// Prepares required fields from MatchParams mp to be able to assess the
  /// match
  void fillRequiredMatchParams(const cv::Mat& patch,
                               const syms::ISymData& symData,
                               const transform::CachedData& cachedData,
                               IMatchParamsRW& mp) const noexcept override;

  REGISTER_MATCH_ASPECT(DirectionalSmoothness);
};

/// Match aspect concerning user's preference for larger symbols as
/// approximations
class LargerSym : public MatchAspect {
 public:
  ~LargerSym() noexcept override = default;

  LargerSym(const LargerSym&) noexcept = default;
  LargerSym(LargerSym&&) noexcept = default;

  // 'k' reference is supposed not to change for the original / copy
  void operator=(const LargerSym&) = delete;
  void operator=(LargerSym&&) = delete;

  /// Providing a clue about how complex is this MatchAspect compared to the
  /// others
  double relativeComplexity() const noexcept override;

  PROTECTED :

      explicit LargerSym(const cfg::IMatchSettings& ms) noexcept;

  /// Defines the scoring rule, based on all required fields computed already in
  /// MatchParams mp
  double score(const IMatchParams& mp,
               const transform::CachedData& cachedData) const noexcept override;

  /// Prepares required fields from MatchParams mp to be able to assess the
  /// match
  void fillRequiredMatchParams(const cv::Mat& patch,
                               const syms::ISymData& symData,
                               const transform::CachedData& cachedData,
                               IMatchParamsRW& mp) const noexcept override;

  REGISTER_MATCH_ASPECT(LargerSym);
};

}  // namespace pic2sym::match

#endif  // H_MATCH_ASPECTS
