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

#include "scoreThresholds.h"
#include "warnings.h"

#pragma warning(push, 0)

#include <cassert>

#pragma warning(pop)

using namespace std;

ScoreThresholds::ScoreThresholds(double multiplier,
                                 const ScoreThresholds& references) noexcept
    : intermediaries(references.intermediaries.size()),
      total(multiplier * references.total) {
  const size_t factorsCount = references.intermediaries.size();
  for (size_t i = 0ULL; i < factorsCount; ++i)
    intermediaries[i] = multiplier * references.intermediaries[i];
}

double ScoreThresholds::overall() const noexcept {
  return total;
}
size_t ScoreThresholds::thresholdsCount() const noexcept {
  return intermediaries.size();
}

#pragma warning(disable : WARN_THROWS_ALTHOUGH_NOEXCEPT)
double ScoreThresholds::operator[](size_t idx) const noexcept(!UT) {
  if (idx >= intermediaries.size())
    THROW_WITH_VAR_MSG(__FUNCTION__ " - idx=" + to_string(idx) +
                           " should be less than intermediaries.size()=" +
                           to_string(intermediaries.size()),
                       out_of_range);
  // intermediaries.at(idx) provides less details than the check above
  return intermediaries[idx];
}
#pragma warning(default : WARN_THROWS_ALTHOUGH_NOEXCEPT)

bool ScoreThresholds::representsInferiorMatch() const noexcept {
  return intermediaries.empty();
}

void ScoreThresholds::inferiorMatch() noexcept {
  if (!intermediaries.empty())
    intermediaries.clear();
}

void ScoreThresholds::update(double totalScore) noexcept {
  total = totalScore;
}

void ScoreThresholds::update(double totalScore,
                             const std::vector<double>& multipliers) noexcept {
  total = totalScore;
  const size_t factorsCount = multipliers.size();
  if (intermediaries.size() != factorsCount)
    intermediaries.resize(factorsCount);
  for (size_t i = 0ULL; i < factorsCount; ++i)
    intermediaries[i] = totalScore * multipliers[i];
}

void ScoreThresholds::update(double multiplier,
                             const IScoreThresholds& references) noexcept {
  total = multiplier * references.overall();
  const size_t factorsCount = references.thresholdsCount();
  if (intermediaries.size() != factorsCount)
    intermediaries.resize(factorsCount);
  for (size_t i = 0ULL; i < factorsCount; ++i)
    intermediaries[i] = multiplier * references[i];
}
