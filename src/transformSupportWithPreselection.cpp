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
#include "matchEngineBase.h"
#include "matchParamsBase.h"
#include "matchProgressWithPreselection.h"
#include "matchSupportWithPreselection.h"
#include "preselectSyms.h"
#include "transformSupportWithPreselection.h"

#pragma warning(push, 0)

#include <opencv2/imgproc/imgproc.hpp>

#pragma warning(pop)

using namespace std;
using namespace cv;

extern const bool UsingOMP;
extern const double AdmitOnShortListEvenForInferiorScoreFactor;
extern const unsigned ShortListLength;
extern unsigned TinySymsSz();
static const unsigned TinySymsSize = TinySymsSz();

TransformSupportWithPreselection::TransformSupportWithPreselection(
    IMatchEngine& me_,
    const IMatchSettings& matchSettings_,
    Mat& resized_,
    Mat& resizedBlurred_,
    vector<vector<unique_ptr<IBestMatch>>>& draftMatches_,
    IMatchSupport& matchSupport_) noexcept
    : TransformSupport(me_,
                       matchSettings_,
                       resized_,
                       resizedBlurred_,
                       draftMatches_),
      matchSupport(dynamic_cast<MatchSupportWithPreselection&>(matchSupport_)) {
}

void TransformSupportWithPreselection::initDrafts(
    bool isColor,
    unsigned patchSz,
    unsigned patchesPerCol,
    unsigned patchesPerRow) noexcept {
  draftMatches.clear();
  draftMatches.resize((size_t)patchesPerCol);

  const Size imgSzForTinySyms(int(TinySymsSize * patchesPerRow),
                              int(TinySymsSize * patchesPerCol));
  resize(resized, resizedForTinySyms, imgSzForTinySyms, 0., 0., INTER_AREA);
  resize(resizedBlurred, resBlForTinySyms, imgSzForTinySyms, 0., 0.,
         INTER_AREA);
  draftMatchesForTinySyms.clear();
  draftMatchesForTinySyms.resize(patchesPerCol);

#pragma omp parallel if (UsingOMP)
#pragma omp for schedule(static, 1) nowait
  for (int r = 0; r < (int)patchesPerCol; ++r) {
    initDraftRow(draftMatches, r, patchesPerRow, resized, resizedBlurred,
                 (int)patchSz, isColor);
    initDraftRow(draftMatchesForTinySyms, r, patchesPerRow, resizedForTinySyms,
                 resBlForTinySyms, (int)TinySymsSize, isColor);
  }
}

void TransformSupportWithPreselection::resetDrafts(
    unsigned patchesPerCol) noexcept {
#pragma omp parallel if (UsingOMP)
#pragma omp for schedule(static, 1) nowait
  for (int r = 0; r < (int)patchesPerCol; ++r) {
    resetDraftRow(draftMatches, r);
    resetDraftRow(draftMatchesForTinySyms, r);
  }
}

void TransformSupportWithPreselection::approxRow(int r,
                                                 int width,
                                                 unsigned patchSz,
                                                 unsigned fromSymIdx,
                                                 unsigned upperSymIdx,
                                                 Mat& result) noexcept {
  const int row = r * (int)patchSz;
  const Range rowRange(row, row + (int)patchSz);
  const vector<unique_ptr<IBestMatch>>& draftMatchesRow =
      draftMatches[(size_t)r];
  const vector<unique_ptr<IBestMatch>>& draftMatchesRowTiny =
      draftMatchesForTinySyms[(size_t)r];

  TopCandidateMatches tcm(ShortListLength);
  MatchProgressWithPreselection mpwp(tcm);

  for (int c = 0, patchColumn = 0; c < width;
       c += (int)patchSz, ++patchColumn) {
    // Skip patches who appear rather uniform either in tiny or normal format
    IBestMatch& draftMatch = *draftMatchesRow[(size_t)patchColumn];
    IBestMatch& draftMatchTiny = *draftMatchesRowTiny[(size_t)patchColumn];
    if (checkUnifPatch(draftMatchTiny) || checkUnifPatch(draftMatch)) {
      manageUnifPatch(matchSettings, result, patchSz, draftMatch, rowRange, c);
      continue;
    }

    // Using the actual score as reference for the original threshold
    const double scoreToBeatByTinyDraft =
        draftMatch.getScore() * AdmitOnShortListEvenForInferiorScoreFactor;
    draftMatchTiny.setScore(scoreToBeatByTinyDraft);
    tcm.reset(scoreToBeatByTinyDraft);

    if (me.improvesBasedOnBatch(fromSymIdx, upperSymIdx, draftMatchTiny,
                                mpwp)) {
      tcm.prepareReport();
      CandidatesShortList shortList;
      tcm.moveShortList(shortList);

      // Examine shortList on actual patches and symbols, not tiny ones
      if (matchSupport.improvesBasedOnBatchShortList(move(shortList),
                                                     draftMatch))
        patchImproved(result, patchSz, draftMatch, rowRange, c);
    }
  }  // columns loop
}
