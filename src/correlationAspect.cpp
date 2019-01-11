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
 If not, see <http://www.gnu.org/licenses/agpl-3.0.txt>.
 ***********************************************************************************************/

#include "correlationAspect.h"
#include "matchParams.h"
#include "symDataBase.h"
#include "cachedData.h"
#include "misc.h"

#pragma warning ( push, 0 )

#include <opencv2/imgproc/imgproc.hpp>

#pragma warning ( pop )

using namespace std;
using namespace cv;

extern const double EPSp1();

REGISTERED_MATCH_ASPECT(CorrelationAspect);

CorrelationAspect::CorrelationAspect(const IMatchSettings &ms) : MatchAspect(ms.get_kCorrel()) {}

double CorrelationAspect::relativeComplexity() const {
	return 90.21;
}

double CorrelationAspect::score(const IMatchParams &mp, const CachedData&) const {
	return k * mp.getAbsCorr().value();
}

void CorrelationAspect::fillRequiredMatchParams(const cv::Mat &patch,
												const ISymData &symData,
												const CachedData &cachedData,
												IMatchParamsRW &mp) const {
	mp.computeAbsCorr(patch, symData, cachedData);
}

/**
Norm L2 of (patch - miuPatch)

The formula is:
  sqrt(sum of (pixel - miu)^2), where miu = (the sum of pixels) / pixelsCount

After expansion, the norm can also be expressed as:
  sqrt((sum of squared pixels) - (square of the sum of pixels)/pixelsCount)

The sum of squared pixels is used also by Structural Similarity,
so this is why using this formula.
*/
void MatchParams::computeNormPatchMinMiu(const cv::Mat &patch, const CachedData &cachedData) {
	if(normPatchMinMiu)
		return;

	computePatchSum(patch);
	computePatchSq(patch);

	const double patch_sum = patchSum.value();

	normPatchMinMiu = sqrt(*sum(patchSq.value()).val - patch_sum * patch_sum / cachedData.getSzSq());

#ifdef _DEBUG // Checking normPatchMinMiu against the official formula
	const double officialNormPatchMinMiu = norm(patch - patch_sum / cachedData.getSzSq(), NORM_L2);
	assert(abs(officialNormPatchMinMiu - normPatchMinMiu.value()) < EPS);
#endif // _DEBUG
}

/**
Computes absCorr (the absolute value of the correlation between patchApprox (PA) and param patch (P) )
or ensures that it has already been computed.

PA is the approximation of P through S - the symbol provided by symData param.
S needs to update its foreground and background to match the corresponding average shades from P.
So, PA is S after undergoing a scaling and 2 translations:
- First the notations:
	- let n be the width/height of P,S and PA => n^2 is their pixels count
	- let bgS = min(S) - the background shade of S (typically 0)
	- let fgS = max(S) - the foreground shade of S (typically 255)
	- let contrS = fgS-minS ( > 0 ) - the contrast of S (range: minContrast .. 255)
	- let bgPA, fgPA and contrPA - the background, foreground and contrast of patchApprox, where
	  bgPA is the average of the shades from P under symData.masks[BG_MASK_IDX]
	  fgPA is the average of the shades from P under symData.masks[FG_MASK_IDX]
	  contrPA = fgPA-bgPA (range -255 .. 255) Note that contrPA might be negative!
	- let miuS, miuPA, miuP ( > 0 ) be the average shade of S, PA and P (sum of shades / n^2)
	- let normS0, normPA0, normP0 ( > 0 ) be the norm (L2) of the shades of S-miuS, PA-miuPA and P-miuP,
	  computed as the square root of (sum of ((pixel shade - average shade)^2))
	- let sigmaS, sigmaPA, sigmaP ( > 0 ) be the population standard deviation
	  (https://en.wikipedia.org/wiki/Standard_deviation#Population_standard_deviation_of_grades_of_eight_students )
	  of the shades of S, PA and P, computed as:
	  (0)  sigma = square root of ((sum of ((pixel shade - average shade)^2)) / n^2)
				= norm(matrix-miu) / n
	  Note that the denominator is n^2 (for population standard deviation),
	  which means subtracting 1 is not necessary!
- Now the resulting formulae:
  (1)  PA = bgPA + (S-bgS) * contrPA/contrS
  (2)  miuPA = bgPA + (miuS-bgS) * contrPA/contrS
  (3)  PA - miuPA = (S - miuS) * contrPA/contrS 
  (4)  sigmaPA = abs(contrPA/contrS) * sigmaS

The formula for the zero normalized cross correlation (zncc) between P and PA is:
 (5)  zncc(P, PA) = (sum of ((P(i,j) - miuP) * (PA(i,j) - miuPA))) / (n^2 * sigmaP * sigmaPA)

Using (3) and (4) in (5) we obtain that the absolute value of the mentioned correlation can avoid
PA completely and is able to use only the original S:
 (6)  absCorr = abs(zncc(P, PA))
		= abs((sum of ((P(i,j) - miuP) * (S(i,j) - miuS))) / (n^2 * sigmaP * sigmaS))
		= abs(zncc(P, S))

Relation (6) is significant, since it means that absCorr is further invariant at
scaling / translating and negating the patch / symbol, which can therefore:
- have either 0..255 or 0..1 range
- be used as negated matrices (255..0 or 1..0 ranges, that is swapping the foreground and background)
  Negated matrices preserve their initial sigma and norm, but shift their miu:
    negatedMiu = maxRange-originalMiu

After expansion of (6) and after making use of (0) (norm = sigma * n) we get:
 (7)  absCorr = abs((sum of (P(i,j) * S(i,j))) - n^2 * miuP * miuS) / (normP0 * normS0)

Using S0 = S - miuS instead of S would keep the normS0, but miuS0 would become 0,

so we get:
 (8) absCorr = abs(sum of (P(i,j) * S0(i,j))) / (normP0 * normS0)
*/
void MatchParams::computeAbsCorr(const Mat &patch, const ISymData &symData,
								 const CachedData &cachedData) {
	if(absCorr)
		return;

	computeNormPatchMinMiu(patch, cachedData);

	const double numerator = abs(*sum(patch.mul(symData.getSymMiu0())).val),
		denominator = normPatchMinMiu.value() * symData.getNormSymMiu0();

	if(denominator != 0.) {
		absCorr = numerator / denominator;

#ifdef _DEBUG // checking that absCorr is in 0..1 range
		assert(*absCorr < EPSp1());
#endif // _DEBUG

		return;
	}

	if(normPatchMinMiu.value() == 0.) { // uniform patch
		absCorr = 1.; // perfect match, as the symbol fades perfectly into the patch
		return;
	}

	// uniform symbol and non-uniform patch
	absCorr = 0.; // cannot fit a blank on something non-blank
}
