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

#include "transformSupport.h"
#include "matchEngine.h"
#include "matchParamsBase.h"
#include "patch.h"
#include "bestMatch.h"
#include "matchSupport.h"

using namespace std;
using namespace cv;

extern const bool UsingOMP;
static MatchProgress dummy;

void TransformSupport::initDraftRow(vector<vector<unique_ptr<IBestMatch>>> &draft,
									int r, unsigned patchesPerRow,
									const Mat &res, const Mat &resBlurred,
									int patchSz, bool isColor) {
	const int row = r * patchSz;
	const Range rowRange(row, row + patchSz);

	auto &draftMatchesRow = draft[(size_t)r]; draftMatchesRow.reserve(patchesPerRow);
	for(int c = 0, cLim = (int)patchesPerRow * patchSz; c < cLim; c += patchSz) {
		const Range colRange(c, c+patchSz);
		const Mat patch(res, rowRange, colRange),
			blurredPatch(resBlurred, rowRange, colRange);

		// Building a Patch with the blurred patch computed for its actual borders
		draftMatchesRow.emplace_back(new BestMatch(Patch(patch, blurredPatch, isColor)));
	}
}

void TransformSupport::resetDraftRow(vector<vector<unique_ptr<IBestMatch>>> &draft, int r) {
	auto &draftMatchesRow = draft[(size_t)r];
	for(auto &draftMatch : draftMatchesRow)
		draftMatch->reset(); // leave nothing but the Patch field
}

void TransformSupport::patchImproved(Mat &result, unsigned sz, const IBestMatch &draftMatch,
									 const Range &rowRange, int startCol) {
	Mat destRegion(result, rowRange, Range(startCol, startCol + (int)sz));
	draftMatch.getApprox().copyTo(destRegion);
}

void TransformSupport::manageUnifPatch(const MatchSettings &ms, Mat &result, unsigned sz, 
									   IBestMatch &draftMatch, const Range &rowRange, int startCol) {
	if(draftMatch.getApprox().empty()) {
		draftMatch.updatePatchApprox(ms);
		patchImproved(result, sz, draftMatch, rowRange, startCol);
	}
}

bool TransformSupport::checkUnifPatch(IBestMatch &draftMatch) {
	return !draftMatch.getPatch().nonUniform();
}

TransformSupport::TransformSupport(MatchEngine &me_, const MatchSettings &matchSettings_,
								   Mat &resized_, Mat &resizedBlurred_,
								   vector<vector<unique_ptr<IBestMatch>>> &draftMatches_) :
	me(me_), matchSettings(matchSettings_),
	resized(resized_), resizedBlurred(resizedBlurred_),
	draftMatches(draftMatches_) {}

void TransformSupport::initDrafts(bool isColor, unsigned patchSz,
								  unsigned patchesPerCol, unsigned patchesPerRow) {
	draftMatches.clear(); draftMatches.resize((size_t)patchesPerCol);

#pragma omp parallel if(UsingOMP)
#pragma omp for schedule(static, 1) nowait
	for(int r = 0; r < (int)patchesPerCol; ++r)
		initDraftRow(draftMatches, r, patchesPerRow, resized, resizedBlurred, (int)patchSz, isColor);
}

void TransformSupport::resetDrafts(unsigned patchesPerCol) {
#pragma omp parallel if(UsingOMP)
#pragma omp for schedule(static, 1) nowait
	for(int r = 0; r < (int)patchesPerCol; ++r)
		resetDraftRow(draftMatches, r);
}

void TransformSupport::approxRow(int r, int width, unsigned patchSz,
								 unsigned fromSymIdx, unsigned upperSymIdx, Mat &result) {
	const int row = r * (int)patchSz;
	const Range rowRange(row, row + (int)patchSz);
	auto &draftMatchesRow = draftMatches[(size_t)r];

	for(int c = 0, patchColumn = 0; c < width; c += (int)patchSz, ++patchColumn) {
		// Skip patches who appear rather uniform
		auto &draftMatch = *draftMatchesRow[(size_t)patchColumn];
		if(checkUnifPatch(draftMatch)) {
			manageUnifPatch(matchSettings, result, patchSz, draftMatch, rowRange, c);
			continue;
		}

		if(me.improvesBasedOnBatch(fromSymIdx, upperSymIdx, draftMatch, dummy))
			patchImproved(result, patchSz, draftMatch, rowRange, c);
	} // columns loop
}