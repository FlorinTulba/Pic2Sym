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

#include "transformSupportWithPreselection.h"
#include "preselectSyms.h"
#include "matchEngine.h"
#include "matchParamsBase.h"
#include "bestMatchBase.h"
#include "matchSupportWithPreselection.h"

#pragma warning ( push, 0 )

#include <opencv2/imgproc/imgproc.hpp>

#pragma warning ( pop )

using namespace std;
using namespace cv;

extern const bool UsingOMP;
extern const double AdmitOnShortListEvenForInferiorScoreFactor;
extern const unsigned ShortListLength;
extern unsigned TinySymsSz();
static const unsigned TinySymsSize = TinySymsSz();

TransformSupportWithPreselection::TransformSupportWithPreselection(MatchEngine &me_, 
																   const MatchSettings &matchSettings_,
																   Mat &resized_,
																   Mat &resizedBlurred_,
																   vector<vector<unique_ptr<IBestMatch>>> &draftMatches_,
																   MatchSupport &matchSupport_) :
	TransformSupport(me_, matchSettings_, resized_, resizedBlurred_, draftMatches_),
	matchSupport(dynamic_cast<MatchSupportWithPreselection&>(matchSupport_)) {}

void TransformSupportWithPreselection::initDrafts(bool isColor, unsigned patchSz,
												  unsigned patchesPerCol, unsigned patchesPerRow) {
	draftMatches.clear(); draftMatches.resize((size_t)patchesPerCol);

	const Size imgSzForTinySyms(int(TinySymsSize * patchesPerRow), int(TinySymsSize * patchesPerCol));
	resize(resized, resizedForTinySyms, imgSzForTinySyms, 0., 0., INTER_AREA);
	resize(resizedBlurred, resBlForTinySyms, imgSzForTinySyms, 0., 0., INTER_AREA);
	draftMatchesForTinySyms.clear(); draftMatchesForTinySyms.resize(patchesPerCol);

#pragma omp parallel if(UsingOMP)
#pragma omp for schedule(static, 1) nowait
	for(int r = 0; r < (int)patchesPerCol; ++r) {
		initDraftRow(draftMatches, r, patchesPerRow, resized, resizedBlurred, (int)patchSz, isColor);
		initDraftRow(draftMatchesForTinySyms, r, patchesPerRow, resizedForTinySyms,
					 resBlForTinySyms, (int)TinySymsSize, isColor);
	}
}

void TransformSupportWithPreselection::resetDrafts(unsigned patchesPerCol) {
#pragma omp parallel if(UsingOMP)
#pragma omp for schedule(static, 1) nowait
	for(int r = 0; r < (int)patchesPerCol; ++r) {
		resetDraftRow(draftMatches, r);
		resetDraftRow(draftMatchesForTinySyms, r);
	}
}

void TransformSupportWithPreselection::approxRow(int r, int width, unsigned patchSz,
												 unsigned fromSymIdx, unsigned upperSymIdx,
												 Mat &result) {
	const int row = r * (int)patchSz;
	const Range rowRange(row, row + (int)patchSz);
	auto &draftMatchesRow = draftMatches[(size_t)r];
	auto &draftMatchesRowTiny = draftMatchesForTinySyms[(size_t)r];

	TopCandidateMatches tcm(ShortListLength);
	MatchProgressWithPreselection mpwp(tcm);

	for(int c = 0, patchColumn = 0; c < width; c += (int)patchSz, ++patchColumn) {
		// Skip patches who appear rather uniform either in tiny or normal format
		auto &draftMatch = *draftMatchesRow[(size_t)patchColumn];
		auto &draftMatchTiny = *draftMatchesRowTiny[(size_t)patchColumn];
		if(checkUnifPatch(draftMatchTiny) || checkUnifPatch(draftMatch)) {
			manageUnifPatch(matchSettings, result, patchSz, draftMatch, rowRange, c);
			continue;
		}

		// Using the actual score as reference for the original threshold
		const double scoreToBeatByTinyDraft =
			draftMatch.getScore() * AdmitOnShortListEvenForInferiorScoreFactor;
		draftMatchTiny.setScore(scoreToBeatByTinyDraft);
		tcm.reset(scoreToBeatByTinyDraft);

		if(me.improvesBasedOnBatch(fromSymIdx, upperSymIdx, draftMatchTiny, mpwp)) {
			tcm.prepareReport();
			auto &&shortList = tcm.getShortList();

			// Examine shortList on actual patches and symbols, not tiny ones
			if(matchSupport.improvesBasedOnBatchShortList(std::move(shortList), draftMatch))
				patchImproved(result, patchSz, draftMatch, rowRange, c);
		}
	} // columns loop
}