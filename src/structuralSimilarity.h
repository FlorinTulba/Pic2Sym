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

#ifndef H_STRUCTURAL_SIMILARITY
#define H_STRUCTURAL_SIMILARITY

#include "match.h"

class IBlurEngine;  // forward declaration

/**
Selecting a symbol with best structural similarity.

See https://ece.uwaterloo.ca/~z70wang/research/ssim for details.
*/
class StructuralSimilarity : public MatchAspect {
 public:
  ~StructuralSimilarity() noexcept = default;

  StructuralSimilarity(const StructuralSimilarity&) noexcept = default;
  StructuralSimilarity(StructuralSimilarity&&) noexcept = default;

  // 'k' reference is supposed not to change for the original / copy
  void operator=(const StructuralSimilarity&) = delete;
  void operator=(StructuralSimilarity&&) = delete;

  /// Providing a clue about how complex is this MatchAspect compared to the
  /// others
  double relativeComplexity() const noexcept override;

  /// Blurring algorithm used to support this match aspect. The Controller sets
  /// it at start.
  static const IBlurEngine& supportBlur;

  PROTECTED :

      explicit StructuralSimilarity(const IMatchSettings& ms) noexcept;

  /// Defines the scoring rule, based on all required fields computed already in
  /// MatchParams mp
  double score(const IMatchParams& mp, const CachedData& cachedData) const
      noexcept override;

  /// Prepares required fields from MatchParams mp to be able to assess the
  /// match
  void fillRequiredMatchParams(const cv::Mat& patch,
                               const ISymData& symData,
                               const CachedData& cachedData,
                               IMatchParamsRW& mp) const noexcept override;

  REGISTER_MATCH_ASPECT(StructuralSimilarity);
};

#endif  // H_STRUCTURAL_SIMILARITY
