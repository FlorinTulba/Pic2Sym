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

#include "matchSupportWithPreselection.h"

#include "bestMatchBase.h"
#include "matchAssessment.h"
#include "matchParamsBase.h"
#include "matchSettingsBase.h"
#include "patchBase.h"
#include "scoreThresholds.h"

using namespace std;

namespace pic2sym {

extern unsigned TinySymsSz();

using namespace transform;
using namespace syms;

namespace match {

MatchSupportWithPreselection::MatchSupportWithPreselection(
    CachedDataRW& cd_,
    VSymData& symsSet_,
    MatchAssessor& matchAssessor_,
    const p2s::cfg::IMatchSettings& matchSettings_) noexcept
    : MatchSupport{cd_},
      cdPresel(true),
      symsSet(&symsSet_),
      matchAssessor(&matchAssessor_),
      matchSettings(&matchSettings_) {
  cdPresel.useNewSymSize(TinySymsSz());
}

const CachedData& MatchSupportWithPreselection::cachedData() const noexcept {
  return cdPresel;
}

void MatchSupportWithPreselection::updateCachedData(
    unsigned fontSz,
    const IFontEngine& fe) noexcept {
  MatchSupport::updateCachedData(fontSz, fe);
  cdPresel.update(fe);
}

bool MatchSupportWithPreselection::improvesBasedOnBatchShortList(
    CandidatesShortList&& shortList,
    IBestMatch& draftMatch) const noexcept(!UT) {
  IMatchParamsRW& mp = draftMatch.refParams();

  bool betterMatchFound{false};

  double score{};

  ScoreThresholds scoresToBeat;
  matchAssessor->scoresToBeat(draftMatch.getScore(), scoresToBeat);

  while (!shortList.empty()) {
    const CandidateId candidateIdx{shortList.top()};

    mp.reset();  // preserves patch-invariant fields

    if (matchAssessor->isBetterMatch(draftMatch.getPatch().matrixToApprox(),
                                     *(*symsSet)[(size_t)candidateIdx], *cd,
                                     scoresToBeat, mp, score)) {
      const ISymData& symData = *(*symsSet)[(size_t)candidateIdx];
      draftMatch.update(score, symData.getCode(), candidateIdx, symData);
      matchAssessor->scoresToBeat(score, scoresToBeat);

      betterMatchFound = true;
    }

    shortList.pop();
  }

  if (betterMatchFound)
    draftMatch.updatePatchApprox(*matchSettings);

  return betterMatchFound;
}

}  // namespace match
}  // namespace pic2sym
