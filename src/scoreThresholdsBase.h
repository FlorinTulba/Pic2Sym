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

#ifndef H_SCORE_THRESHOLDS_BASE
#define H_SCORE_THRESHOLDS_BASE

#include "misc.h"

#pragma warning(push, 0)

#include <vector>

#pragma warning(pop)

extern template class std::vector<double>;

namespace pic2sym::match {

/**
Interface of ScoreThresholds.

Stores and updates the threshold values for intermediary scores. These values
might help sparing the computation of some matching aspects.

When UseSkipMatchAspectsHeuristic is false, this class behaves almost like a
simple `double` value.
*/
class IScoreThresholds /*abstract*/ {
 public:
  /// Provides final threshold score
  virtual double overall() const noexcept = 0;

  /// @return the number of intermediary scores
  virtual size_t thresholdsCount() const noexcept = 0;

  /// Provides the idx-th intermediary score
  virtual double operator[](size_t idx) const noexcept(!UT) = 0;

  /// Sets final score to totalScore
  virtual void update(double totalScore) noexcept = 0;

  /// Updates the thresholds for clusters (thresholds for the symbols
  /// (references) multiplied by multiplier.)
  virtual void update(double multiplier,
                      const IScoreThresholds& references) noexcept = 0;

  // Methods used only when UseSkipMatchAspectsHeuristic is true

  /// Updates final and intermediary scores as totalScore * multipliers
  virtual void update(double totalScore,
                      const std::vector<double>& multipliers) noexcept = 0;

  /// Makes sure that intermediary results won't be used as long as finding only
  /// bad matches
  virtual void inferiorMatch() noexcept = 0;

  /// true for empty intermediaries [triggered by inferiorMatch()]
  virtual bool representsInferiorMatch() const noexcept = 0;

  virtual ~IScoreThresholds() noexcept = 0 {}
};

}  // namespace pic2sym::match

#endif  // H_SCORE_THRESHOLDS_BASE
