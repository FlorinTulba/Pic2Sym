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

#include "matchParams.h"
#include "matchSettings.h"
#include "symDataBase.h"
#include "cachedData.h"
#include "patchBase.h"
#include "misc.h"

using namespace std;
using namespace boost;
using namespace cv;

extern const double EPSp1();

namespace {
	const double EPSp255 = 255. + EPS;
	const double EPSpSdevMaxFgBg = CachedData::sdevMaxFgBg() + EPS;
	const double EPSpSdevMaxEdge = CachedData::sdevMaxEdge() + EPS;
	const double EPSpSqrt2 = sqrt(2.) + EPS;

	/// Prepares the value before launching any image transformation (spares transformation time)
	const MatchParams& createPerfectMatch() {
#pragma warning ( disable : WARN_THREAD_UNSAFE )
		static MatchParams idealMatch;
#pragma warning ( default : WARN_THREAD_UNSAFE )

		idealMatch.
			setMcPatch(Point2d()).setMcPatchApprox(Point2d()).
			setMcsOffset(0.).	// Same mass centers
			setSdevFg(0.).setSdevBg(0.).setSdevEdge(0.). // All standard deviations 0
			setSsim(1.).		// Perfect structural similarity
			setSymDensity(1.).	// Largest density possible
			setContrast(255.);	// Largest contrast possible

		return idealMatch;
	}

	/// Early processing of the perfect match, to spare transformation time
	const MatchParams &thePerfectMatch = createPerfectMatch();

} // anonymous namespace

const MatchParams& MatchParams::perfectMatch() {
	return thePerfectMatch;
}

const optional<Point2d>& MatchParams::getMcPatch() const { return mcPatch; }
const optional<Mat>& MatchParams::getBlurredPatch() const { return blurredPatch; }
const optional<Mat>& MatchParams::getBlurredPatchSq() const { return blurredPatchSq; }
const optional<Mat>& MatchParams::getVariancePatch() const { return variancePatch; }
const optional<Mat>& MatchParams::getPatchApprox() const { return patchApprox; }
const optional<Point2d>& MatchParams::getMcPatchApprox() const { return mcPatchApprox; }
const optional<double>& MatchParams::getMcsOffset() const { return mcsOffset; }
const optional<double>& MatchParams::getSymDensity() const { return symDensity; }
const optional<double>& MatchParams::getFg() const { return fg; }
const optional<double>& MatchParams::getBg() const { return bg; }
const optional<double>& MatchParams::getContrast() const { return contrast; }
const optional<double>& MatchParams::getSsim() const { return ssim; }
const optional<double>& MatchParams::getSdevFg() const { return sdevFg; }
const optional<double>& MatchParams::getSdevBg() const { return sdevBg; }
const optional<double>& MatchParams::getSdevEdge() const { return sdevEdge; }
unique_ptr<IMatchParamsRW> MatchParams::clone() const { return make_unique<MatchParams>(*this); }

MatchParams& MatchParams::reset(bool skipPatchInvariantParts/* = true*/) {
	mcPatchApprox = none;
	patchApprox = none;
	ssim = fg = bg = contrast = sdevFg = sdevBg = sdevEdge = symDensity = mcsOffset = none;

	if(!skipPatchInvariantParts) {
		mcPatch = none;
		blurredPatch = blurredPatchSq = variancePatch = none;
	}
	return *this;
}

MatchParams& MatchParams::setMcPatch(const cv::Point2d &p) { mcPatch = p; return *this; }
MatchParams& MatchParams::setBlurredPatch(const cv::Mat &m) { blurredPatch = m; return *this; }
MatchParams& MatchParams::setBlurredPatchSq(const cv::Mat &m) { blurredPatchSq = m; return *this; }
MatchParams& MatchParams::setVariancePatch(const cv::Mat &m) { variancePatch = m; return *this; }
MatchParams& MatchParams::setPatchApprox(const cv::Mat &m) { patchApprox = m; return *this; }
MatchParams& MatchParams::setMcPatchApprox(const cv::Point2d &p) { mcPatchApprox = p; return *this; }
MatchParams& MatchParams::setMcsOffset(double v) { mcsOffset = v; return *this; }
MatchParams& MatchParams::setSymDensity(double v) { symDensity = v; return *this; }
MatchParams& MatchParams::setFg(double v) { fg = v; return *this; }
MatchParams& MatchParams::setBg(double v) { bg = v; return *this; }
MatchParams& MatchParams::setContrast(double v) { contrast = v; return *this; }
MatchParams& MatchParams::setSsim(double v) { ssim = v; return *this; }
MatchParams& MatchParams::setSdevFg(double v) { sdevFg = v; return *this; }
MatchParams& MatchParams::setSdevBg(double v) { sdevBg = v; return *this; }
MatchParams& MatchParams::setSdevEdge(double v) { sdevEdge = v; return *this; }

void MatchParams::computeMean(const Mat &patch, const Mat &mask, optional<double> &miu) {
	if(miu)
		return;

	miu = *mean(patch, mask).val;
	assert(*miu > -EPS && *miu < EPSp255);
}

void MatchParams::computeFg(const Mat &patch, const ISymData &symData) {
	computeMean(patch, symData.getMask(ISymData::FG_MASK_IDX), fg);
}

void MatchParams::computeBg(const Mat &patch, const ISymData &symData) {
	computeMean(patch, symData.getMask(ISymData::BG_MASK_IDX), bg);
}

void MatchParams::computeContrast(const Mat &patch, const ISymData &symData) {
	if(contrast)
		return;

	computeFg(patch, symData);
	computeBg(patch, symData);

	contrast = fg.value() - bg.value();
	assert(abs(contrast.value()) < 255.5);
}

void MatchParams::computeSdev(const Mat &patch, const Mat &mask,
							  optional<double> &miu, optional<double> &sdev) {
	if(sdev)
		return;

	if(miu) {
		sdev = norm(patch - miu.value(), NORM_L2, mask) / sqrt(countNonZero(mask));
	} else {
		Scalar miu_, sdev_;
		meanStdDev(patch, miu_, sdev_, mask);
		miu = *miu_.val;
		sdev = *sdev_.val;
	}
	assert(*sdev < EPSpSdevMaxFgBg);
}

void MatchParams::computeSdevFg(const Mat &patch, const ISymData &symData) {
	computeSdev(patch, symData.getMask(ISymData::FG_MASK_IDX), fg, sdevFg);
}

void MatchParams::computeSdevBg(const Mat &patch, const ISymData &symData) {
	computeSdev(patch, symData.getMask(ISymData::BG_MASK_IDX), bg, sdevBg);
}

void MatchParams::computePatchApprox(const Mat &patch, const ISymData &symData) {
	if(patchApprox)
		return;

	computeContrast(patch, symData);

	if(contrast.value() == 0.) {
		patchApprox = Mat(patch.rows, patch.cols, CV_64FC1, Scalar(bg.value()));
		return;
	}

	patchApprox = bg.value() +
		symData.getMask(ISymData::GROUNDED_SYM_IDX) * (contrast.value() / symData.getDiffMinMax());
}

void MatchParams::computeSdevEdge(const Mat &patch, const ISymData &symData) {
	if(sdevEdge)
		return;

	const auto &edgeMask = symData.getMask(ISymData::EDGE_MASK_IDX);
	const int cnz = countNonZero(edgeMask);
	if(cnz == 0) {
		sdevEdge = 0.;
		return;
	}

	computePatchApprox(patch, symData);

	sdevEdge = norm(patch, patchApprox.value(), NORM_L2, edgeMask) / sqrt(cnz);
	assert(*sdevEdge < EPSpSdevMaxEdge);
}

void MatchParams::computeSymDensity(const ISymData &symData) {
	if(symDensity)
		return;

	// The method 'MatchAspect::score(const MatchParams &mp, const CachedData &cachedData)'
	// needs symData.avgPixVal stored within MatchParams mp. That's why the mere value copy from below:
	symDensity = symData.getAvgPixVal();
	assert(*symDensity < EPSp1());
}

void MatchParams::computeMcPatch(const Mat &patch, const CachedData &cachedData) {
	if(mcPatch)
		return;

	Mat temp;
	double patchSum, mcX, mcY;

	patchSum = *sum(patch).val;

	reduce(patch, temp, 0, CV_REDUCE_SUM);	// sum all rows
	mcX = temp.dot(cachedData.consec);

	reduce(patch, temp, 1, CV_REDUCE_SUM);	// sum all columns
	mcY = temp.t().dot(cachedData.consec);

	mcPatch = Point2d(mcX, mcY) / (patchSum * cachedData.sz_1);
	assert(mcPatch->x > -EPS && mcPatch->x < EPSp1());
	assert(mcPatch->y > -EPS && mcPatch->y < EPSp1());
}

void MatchParams::computeMcPatchApprox(const Mat &patch, const ISymData &symData,
									   const CachedData&) {
	if(mcPatchApprox)
		return;

	computeContrast(patch, symData);
	computeSymDensity(symData);

	// Obtaining glyph's mass center
	const double k = symDensity.value() * contrast.value(),
				delta = .5 * bg.value(),
				denominator = k + bg.value();
	if(denominator == 0.)
		mcPatchApprox = CachedData::unitSquareCenter();
	else
		mcPatchApprox = (k * symData.getMc() + Point2d(delta, delta)) / denominator;
	assert(mcPatchApprox->x > -EPS && mcPatchApprox->x < EPSp1());
	assert(mcPatchApprox->y > -EPS && mcPatchApprox->y < EPSp1());
}

void MatchParams::computeMcsOffset(const Mat &patch, const ISymData &symData,
								   const CachedData &cachedData) {
	if(mcsOffset)
		return;

	computeMcPatch(patch, cachedData);
	computeMcPatchApprox(patch, symData, cachedData);

	mcsOffset = norm(mcPatch.value() - mcPatchApprox.value());
	assert(mcsOffset < EPSpSqrt2);
}
