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

#ifndef H_SCORE_THRESHOLDS
#define H_SCORE_THRESHOLDS

#include "misc.h"
#include "scoreThresholdsBase.h"

/**
Stores and updates the threshold values for intermediary scores. These values
might help sparing the computation of some matching aspects.

Substitute class of valarray<double>, customized for optimal performance of the
use cases from Pic2Sym. When UseSkipMatchAspectsHeuristic is false, this class
behaves almost like a simple `double` value.
*/
class ScoreThresholds : public IScoreThresholds {
 public:
  constexpr ScoreThresholds() noexcept {}

  /**
  Used to set thresholds for clusters, which are the thresholds for the symbols
  (references) multiplied by multiplier.
  */
  ScoreThresholds(double multiplier,
                  const ScoreThresholds& references) noexcept;

  /// Provides final threshold score (field total)
  double overall() const noexcept final;

  /// @return the number of intermediary scores
  size_t thresholdsCount() const noexcept final;

  /**
  Provides intermediaries[idx]
  @throw out_of_range for an invalid idx

  Exception to be only reported, not handled
  */
  double operator[](size_t idx) const noexcept(!UT) override;

  /// Sets total to totalScore
  void update(double totalScore) noexcept final;

  /// Updates the thresholds for clusters (thresholds for the symbols
  /// (references) multiplied by multiplier.)
  void update(double multiplier,
              const IScoreThresholds& references) noexcept override;

  // Methods used only when UseSkipMatchAspectsHeuristic is true

  /// Updates total and intermediaries = totalScore*multipliers
  void update(double totalScore,
              const std::vector<double>& multipliers) noexcept override;

  /// Makes sure that intermediary results won't be used as long as finding only
  /// bad matches
  void inferiorMatch() noexcept override;

  /// true for empty intermediaries [triggered by inferiorMatch()]
  bool representsInferiorMatch() const noexcept override;

 private:
  std::vector<double> intermediaries;  ///< the intermediary threshold scores
  double total = 0.;                   ///< the final threshold score
};

#endif  // H_SCORE_THRESHOLDS
