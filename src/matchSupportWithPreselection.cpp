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

#include "bestMatchBase.h"
#include "matchAssessment.h"
#include "matchParamsBase.h"
#include "matchSettingsBase.h"
#include "matchSupportWithPreselection.h"
#include "patchBase.h"
#include "scoreThresholds.h"

using namespace std;

extern unsigned TinySymsSz();

MatchSupportWithPreselection::MatchSupportWithPreselection(
    CachedDataRW& cd_,
    VSymData& symsSet_,
    MatchAssessor& matchAssessor_,
    const IMatchSettings& matchSettings_) noexcept
    : MatchSupport(cd_),
      cdPresel(true),
      symsSet(symsSet_),
      matchAssessor(matchAssessor_),
      matchSettings(matchSettings_) {
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

#pragma warning(disable : WARN_THROWS_ALTHOUGH_NOEXCEPT)
bool MatchSupportWithPreselection::improvesBasedOnBatchShortList(
    CandidatesShortList&& shortList,
    IBestMatch& draftMatch) const noexcept(!UT) {
  const unique_ptr<IMatchParamsRW>& mp = draftMatch.refParams();
  if (!mp)
    THROW_WITH_CONST_MSG(__FUNCTION__ " called for uniformous patch "
                         "with nullptr match parameters!", invalid_argument);

  bool betterMatchFound = false;

  double score = 0.;

  ScoreThresholds scoresToBeat;
  matchAssessor.scoresToBeat(draftMatch.getScore(), scoresToBeat);

  while (!shortList.empty()) {
    const CandidateId candidateIdx = shortList.top();

    mp->reset();  // preserves patch-invariant fields

    if (matchAssessor.isBetterMatch(draftMatch.getPatch().matrixToApprox(),
                                    *symsSet[(size_t)candidateIdx], cd,
                                    scoresToBeat, *mp, score)) {
      const ISymData& symData = *symsSet[(size_t)candidateIdx];
      draftMatch.update(score, symData.getCode(), candidateIdx, symData);
      matchAssessor.scoresToBeat(score, scoresToBeat);

      betterMatchFound = true;
    }

    shortList.pop();
  }

  if (betterMatchFound)
    draftMatch.updatePatchApprox(matchSettings);

  return betterMatchFound;
}
#pragma warning(default : WARN_THROWS_ALTHOUGH_NOEXCEPT)
