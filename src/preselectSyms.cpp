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

#include "preselectSyms.h"

#include "warnings.h"

#pragma warning(push, 0)

#include <gsl/gsl>

#pragma warning(pop)

using namespace std;
using namespace gsl;

namespace pic2sym::transform {

/// Clearing containers without clear() method, but which do have empty() and
/// pop() methods
template <class Cont>
static void clearCont(Cont& cont) noexcept {
  while (!cont.empty())
    cont.pop();
}

TopCandidateMatches::Candidate::Candidate(CandidateId idx_,
                                          double score_) noexcept
    : score(score_), idx(idx_) {}

bool TopCandidateMatches::Candidate::Greater::operator()(
    const ICandidate& c1,
    const ICandidate& c2) const noexcept {
  return c1.getScore() > c2.getScore();
}

#pragma warning(disable : WARN_THROWS_ALTHOUGH_NOEXCEPT)
TopCandidateMatches::TopCandidateMatches(
    unsigned shortListLength /* = 1U*/,
    double origThreshScore /* = 0.*/) noexcept(!UT)
    : thresholdScore(origThreshScore), n(shortListLength) {
  EXPECTS_OR_REPORT_AND_THROW_CONST_MSG(
      shortListLength > 0U, invalid_argument,
      HERE.function_name() + " needs shortListLength>=1"s);
}
#pragma warning(default : WARN_THROWS_ALTHOUGH_NOEXCEPT)

void TopCandidateMatches::reset(double origThreshScore) noexcept {
  shortListReady = false;
  clearCont(scrapbook);
  clearCont(shortList);

  thresholdScore = origThreshScore;
}

#pragma warning(disable : WARN_THROWS_ALTHOUGH_NOEXCEPT)
bool TopCandidateMatches::checkCandidate(unsigned candidateIdx,
                                         double score) noexcept(!UT) {
  EXPECTS_OR_REPORT_AND_THROW_CONST_MSG(
      !shortListReady, logic_error,
      HERE.function_name() + " should be called only before prepareReport()!"s);

  if (score <= thresholdScore)
    return false;

  if (narrow_cast<unsigned>(size(scrapbook)) == n)
    // If the short list is full at function start
    scrapbook.pop();  // the worst candidate must be removed

  // The new better candidate enters the short list
  scrapbook.emplace(candidateIdx, score);

  if (narrow_cast<unsigned>(size(scrapbook)) == n)
    // If the short list is full now
    // New threshold is the score of the new worst candidate
    thresholdScore = scrapbook.top().getScore();

  return true;
}
#pragma warning(default : WARN_THROWS_ALTHOUGH_NOEXCEPT)

void TopCandidateMatches::prepareReport() noexcept {
  if (shortListReady)
    return;  // this method should be called only once

  while (!scrapbook.empty()) {
    // Place worse candidates to the bottom of the stack
    shortList.push(scrapbook.top().getIdx());
    scrapbook.pop();
  }

  shortListReady = true;
}

bool TopCandidateMatches::foundAny() const noexcept {
  return !shortList.empty() || !scrapbook.empty();
}

#pragma warning(disable : WARN_THROWS_ALTHOUGH_NOEXCEPT)
const CandidatesShortList& TopCandidateMatches::getShortList() const
    noexcept(!UT) {
  EXPECTS_OR_REPORT_AND_THROW_CONST_MSG(
      shortListReady, logic_error,
      HERE.function_name() +
          " should be called only after prepareReport() "
          "and not after moveShortList()!"s);

  return shortList;
}
#pragma warning(default : WARN_THROWS_ALTHOUGH_NOEXCEPT)

#pragma warning(disable : WARN_THROWS_ALTHOUGH_NOEXCEPT)
void TopCandidateMatches::moveShortList(CandidatesShortList& dest) noexcept(
    !UT) {
  EXPECTS_OR_REPORT_AND_THROW_CONST_MSG(
      shortListReady, logic_error,
      HERE.function_name() + " should be called only after prepareReport()!"s);

  dest = move(shortList);

  shortListReady = false;
}
#pragma warning(default : WARN_THROWS_ALTHOUGH_NOEXCEPT)

}  // namespace pic2sym::transform
