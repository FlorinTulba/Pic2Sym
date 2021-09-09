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

#include "scoreThresholds.h"

#include "warnings.h"

#pragma warning(push, 0)

#include <gsl/gsl>

#pragma warning(pop)

using namespace std;

namespace pic2sym::match {

ScoreThresholds::ScoreThresholds(double multiplier,
                                 const ScoreThresholds& references) noexcept
    : intermediaries(size(references.intermediaries)),
      total(multiplier * references.total) {
  const size_t factorsCount{size(references.intermediaries)};
  for (size_t i{}; i < factorsCount; ++i)
    intermediaries[i] = multiplier * references.intermediaries[i];
}

double ScoreThresholds::overall() const noexcept {
  return total;
}
size_t ScoreThresholds::thresholdsCount() const noexcept {
  return size(intermediaries);
}

#pragma warning(disable : WARN_THROWS_ALTHOUGH_NOEXCEPT)
double ScoreThresholds::operator[](size_t idx) const noexcept(!UT) {
  EXPECTS_OR_REPORT_AND_THROW(
      idx < size(intermediaries), out_of_range,
      HERE.function_name() + " - idx="s + to_string(idx) +
          " should be less than size(intermediaries)="s +
          to_string(size(intermediaries)));

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
  const size_t factorsCount{size(multipliers)};
  if (size(intermediaries) != factorsCount)
    intermediaries.resize(factorsCount);
  for (size_t i{}; i < factorsCount; ++i)
    intermediaries[i] = totalScore * multipliers[i];
}

void ScoreThresholds::update(double multiplier,
                             const IScoreThresholds& references) noexcept {
  total = multiplier * references.overall();
  const size_t factorsCount{references.thresholdsCount()};
  if (size(intermediaries) != factorsCount)
    intermediaries.resize(factorsCount);
  for (size_t i{}; i < factorsCount; ++i)
    intermediaries[i] = multiplier * references[i];
}

}  // namespace pic2sym::match
