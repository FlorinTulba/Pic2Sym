/************************************************************************************************
 The application Pic2Sym approximates images by a
 grid of colored symbols with colored backgrounds.

 Copyrights from the libraries used by the program:
 - (c) 2016 Boost (www.boost.org)
		License: <http://www.boost.org/LICENSE_1_0.txt>
			or doc/licenses/Boost.lic
 - (c) 2015 OpenCV (www.opencv.org)
		License: <http://opencv.org/license.html>
            or doc/licenses/OpenCV.lic
 - (c) 2015 The FreeType Project (www.freetype.org)
		License: <http://git.savannah.gnu.org/cgit/freetype/freetype2.git/plain/docs/FTL.TXT>
	        or doc/licenses/FTL.txt
 - (c) 1997-2002 OpenMP Architecture Review Board (www.openmp.org)
   (c) Microsoft Corporation (Visual C++ implementation for OpenMP C/C++ Version 2.0 March 2002)
		See: <https://msdn.microsoft.com/en-us/library/8y6825x5(v=vs.140).aspx>
 - (c) 1995-2013 zlib software (Jean-loup Gailly and Mark Adler - see: www.zlib.net)
		License: <http://www.zlib.net/zlib_license.html>
            or doc/licenses/zlib.lic
 
 (c) 2016, 2017 Florin Tulba <florintulba@yahoo.com>

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
 If not, see <http://www.gnu.org/licenses/agpl-3.0.txt>.
 ***********************************************************************************************/

#include "matchSupportWithPreselection.h"
#include "matchAssessment.h"
#include "scoreThresholds.h"
#include "matchParamsBase.h"
#include "patchBase.h"
#include "bestMatchBase.h"
#include "matchSettingsBase.h"

using namespace std;

extern unsigned TinySymsSz();

MatchSupportWithPreselection::MatchSupportWithPreselection(CachedData &cd_, VSymData &symsSet_,
														   MatchAssessor &matchAssessor_,
														   const IMatchSettings &matchSettings_) :
		MatchSupport(cd_), cdPresel(true), symsSet(symsSet_),
		matchAssessor(matchAssessor_), matchSettings(matchSettings_) {
	cdPresel.useNewSymSize(TinySymsSz());
}

const CachedData& MatchSupportWithPreselection::cachedData() const {
	return cdPresel;
}

void MatchSupportWithPreselection::updateCachedData(unsigned fontSz, const IFontEngine &fe) {
	MatchSupport::updateCachedData(fontSz, fe);
	cdPresel.update(fe);
}

bool MatchSupportWithPreselection::improvesBasedOnBatchShortList(CandidatesShortList &&shortList,
																 IBestMatch &draftMatch) const {
	bool betterMatchFound = false;

	double score;

	const uniquePtr<IMatchParamsRW> &mp = draftMatch.refParams();
	assert(mp);
	ScoreThresholds scoresToBeat;
	matchAssessor.scoresToBeat(draftMatch.getScore(), scoresToBeat);

	while(!shortList.empty()) {
		const CandidateId candidateIdx = shortList.top();

		mp->reset(); // preserves patch-invariant fields

		if(matchAssessor.isBetterMatch(draftMatch.getPatch().matrixToApprox(),
									*symsSet[(size_t)candidateIdx], cd,
									scoresToBeat, *mp, score)) {
			const ISymData &symData = *symsSet[(size_t)candidateIdx];
			draftMatch.update(score, symData.getCode(), candidateIdx, symData);
			matchAssessor.scoresToBeat(score, scoresToBeat);

			betterMatchFound = true;
		}

		shortList.pop();
	}

	if(betterMatchFound)
		draftMatch.updatePatchApprox(matchSettings);

	return betterMatchFound;
}
