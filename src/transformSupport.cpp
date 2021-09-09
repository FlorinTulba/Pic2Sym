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

#include "transformSupport.h"

#include "bestMatch.h"
#include "matchEngineBase.h"
#include "matchParamsBase.h"
#include "matchProgress.h"
#include "matchSupport.h"
#include "patch.h"

using namespace std;
using namespace cv;

namespace pic2sym {

extern const bool UsingOMP;

using namespace match;

static MatchProgress dummy;

namespace transform {

void TransformSupport::initDraftRow(
    vector<vector<unique_ptr<IBestMatch>>>& draft,
    int r,
    unsigned patchesPerRow,
    const Mat& res,
    const Mat& resBlurred,
    int patchSz,
    bool isColor) noexcept {
  const int row{r * patchSz};
  const Range rowRange{row, row + patchSz};

  vector<unique_ptr<IBestMatch>>& draftMatchesRow = draft[(size_t)r];
  draftMatchesRow.reserve(patchesPerRow);
  for (int c{}, cLim{(int)patchesPerRow * patchSz}; c < cLim; c += patchSz) {
    const Range colRange{c, c + patchSz};
    const Mat patch{res, rowRange, colRange};
    const Mat blurredPatch{resBlurred, rowRange, colRange};

    // Building a Patch with the blurred patch computed for its actual borders
    draftMatchesRow.push_back(make_unique<BestMatch>(
        p2s::input::Patch(patch, blurredPatch, isColor)));
  }
}

void TransformSupport::resetDraftRow(
    vector<vector<unique_ptr<IBestMatch>>>& draft,
    int r) noexcept {
  vector<unique_ptr<IBestMatch>>& draftMatchesRow = draft[(size_t)r];
  for (const unique_ptr<IBestMatch>& draftMatch : draftMatchesRow)
    draftMatch->reset();  // leave nothing but the Patch field
}

void TransformSupport::patchImproved(Mat& result,
                                     unsigned sz,
                                     const IBestMatch& draftMatch,
                                     const Range& rowRange,
                                     int startCol) noexcept {
  Mat destRegion{result, rowRange, Range{startCol, startCol + (int)sz}};
  draftMatch.getApprox().copyTo(destRegion);
}

void TransformSupport::manageUnifPatch(const p2s::cfg::IMatchSettings& ms,
                                       Mat& result,
                                       unsigned sz,
                                       IBestMatch& draftMatch,
                                       const Range& rowRange,
                                       int startCol) noexcept {
  if (draftMatch.getApprox().empty()) {
    draftMatch.updatePatchApprox(ms);
    patchImproved(result, sz, draftMatch, rowRange, startCol);
  }
}

bool TransformSupport::checkUnifPatch(const IBestMatch& draftMatch) noexcept {
  return !draftMatch.getPatch().nonUniform();
}

TransformSupport::TransformSupport(
    IMatchEngine& me_,
    const p2s::cfg::IMatchSettings& matchSettings_,
    Mat& resized_,
    Mat& resizedBlurred_,
    vector<vector<unique_ptr<IBestMatch>>>& draftMatches_) noexcept
    : me(&me_),
      matchSettings(&matchSettings_),
      resized(&resized_),
      resizedBlurred(&resizedBlurred_),
      draftMatches(&draftMatches_) {}

void TransformSupport::initDrafts(bool isColor,
                                  unsigned patchSz,
                                  unsigned patchesPerCol,
                                  unsigned patchesPerRow) noexcept {
  draftMatches->clear();
  draftMatches->resize((size_t)patchesPerCol);

#pragma warning(disable : WARN_CODE_ANALYSIS_IGNORES_OPENMP)
#pragma omp parallel if (UsingOMP)
#pragma omp for schedule(static, 1) nowait
  for (int r{}; r < (int)patchesPerCol; ++r)
    initDraftRow(*draftMatches, r, patchesPerRow, *resized, *resizedBlurred,
                 (int)patchSz, isColor);
#pragma warning(default : WARN_CODE_ANALYSIS_IGNORES_OPENMP)
}

void TransformSupport::resetDrafts(unsigned patchesPerCol) noexcept {
#pragma omp parallel if (UsingOMP)
#pragma omp for schedule(static, 1) nowait
  for (int r{}; r < (int)patchesPerCol; ++r)
    resetDraftRow(*draftMatches, r);
}

void TransformSupport::approxRow(int r,
                                 int width,
                                 unsigned patchSz,
                                 unsigned fromSymIdx,
                                 unsigned upperSymIdx,
                                 Mat& result) noexcept {
  const int row{r * (int)patchSz};
  const Range rowRange{row, row + (int)patchSz};
  const vector<unique_ptr<IBestMatch>>& draftMatchesRow =
      (*draftMatches)[(size_t)r];

  for (int c{}, patchColumn{}; c < width; c += (int)patchSz, ++patchColumn) {
    // Skip patches who appear rather uniform
    IBestMatch& draftMatch = *draftMatchesRow[(size_t)patchColumn];
    if (checkUnifPatch(draftMatch)) {
      manageUnifPatch(*matchSettings, result, patchSz, draftMatch, rowRange, c);
      continue;
    }

    if (me->improvesBasedOnBatch(fromSymIdx, upperSymIdx, draftMatch, dummy))
      patchImproved(result, patchSz, draftMatch, rowRange, c);
  }  // columns loop
}

}  // namespace transform
}  // namespace pic2sym
