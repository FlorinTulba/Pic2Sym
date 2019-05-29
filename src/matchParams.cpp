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

#include "cachedData.h"
#include "matchParams.h"
#include "matchSettingsBase.h"
#include "misc.h"
#include "patchBase.h"
#include "symDataBase.h"
#include "warnings.h"

#if defined(_DEBUG) || defined(UNIT_TESTING)

extern const std::wstring& COMMA();

#endif  // defined(_DEBUG) || defined(UNIT_TESTING)

using namespace std;
using namespace cv;

namespace {
constexpr double EPSp255 = 255. + EPS;
constexpr double EPSpSdevMaxFgBg = CachedData::MaxSdev::forFgOrBg + EPS;
constexpr double EPSpSdevMaxEdge = CachedData::MaxSdev::forEdges + EPS;
constexpr double EPSpSqrt2 = M_SQRT2 + EPS;
}  // anonymous namespace

const MatchParams& MatchParams::perfectMatch() noexcept {
  static MatchParams idealMatch;
  static bool initialized = false;

  if (!initialized) {
    // Same mass centers
    idealMatch.mcPatch = idealMatch.mcPatchApprox = Point2d();
    idealMatch.mcsOffset = 0.;

    // All standard deviations 0
    idealMatch.sdevFg = idealMatch.sdevBg = idealMatch.sdevEdge = 0.;

    idealMatch.ssim = 1.;  // Perfect structural similarity

    idealMatch.absCorr = 1.;  // Perfect correlation

    idealMatch.symDensity = 1.;  // Largest density possible

    idealMatch.contrast = 255.;  // Largest contrast possible

    initialized = true;
  }

  return idealMatch;
}

const optional<Point2d>& MatchParams::getMcPatch() const noexcept {
  return mcPatch;
}
#ifdef UNIT_TESTING
const std::optional<double>& MatchParams::getPatchSum() const noexcept {
  return patchSum;
}
const std::optional<cv::Mat>& MatchParams::getPatchSq() const noexcept {
  return patchSq;
}
const std::optional<double>& MatchParams::getNormPatchMinMiu() const noexcept {
  return normPatchMinMiu;
}
const optional<Mat>& MatchParams::getBlurredPatch() const noexcept {
  return blurredPatch;
}
const optional<Mat>& MatchParams::getBlurredPatchSq() const noexcept {
  return blurredPatchSq;
}
const optional<Mat>& MatchParams::getVariancePatch() const noexcept {
  return variancePatch;
}
const optional<Mat>& MatchParams::getPatchApprox() const noexcept {
  return patchApprox;
}
#endif  // UNIT_TESTING defined
const optional<Point2d>& MatchParams::getMcPatchApprox() const noexcept {
  return mcPatchApprox;
}
const optional<double>& MatchParams::getMcsOffset() const noexcept {
  return mcsOffset;
}
const optional<double>& MatchParams::getSymDensity() const noexcept {
  return symDensity;
}

#if defined(_DEBUG) || defined(UNIT_TESTING)
const wstring MatchParams::toWstring() const noexcept {
  wostringstream os;
  os << ssim << COMMA() << absCorr << COMMA() << sdevFg << COMMA() << sdevEdge
     << COMMA() << sdevBg << COMMA() << fg << COMMA() << bg << COMMA();

  if (mcPatchApprox)
    os << mcPatchApprox->x << COMMA() << mcPatchApprox->y << COMMA();
  else
    os << L"--" << COMMA() << L"--" << COMMA();

  if (mcPatch)
    os << mcPatch->x << COMMA() << mcPatch->y << COMMA();
  else
    os << L"--" << COMMA() << L"--" << COMMA();

  os << symDensity;
  return os.str();
}

const optional<double>& MatchParams::getFg() const noexcept {
  return fg;
}
#endif  // defined(_DEBUG) || defined(UNIT_TESTING)

const optional<double>& MatchParams::getBg() const noexcept {
  return bg;
}
const optional<double>& MatchParams::getContrast() const noexcept {
  return contrast;
}
const optional<double>& MatchParams::getSsim() const noexcept {
  return ssim;
}
const optional<double>& MatchParams::getAbsCorr() const noexcept {
  return absCorr;
}
const optional<double>& MatchParams::getSdevFg() const noexcept {
  return sdevFg;
}
const optional<double>& MatchParams::getSdevBg() const noexcept {
  return sdevBg;
}
const optional<double>& MatchParams::getSdevEdge() const noexcept {
  return sdevEdge;
}
#ifdef UNIT_TESTING
unique_ptr<IMatchParamsRW> MatchParams::clone() const noexcept {
  return make_unique<MatchParams>(*this);
}
#endif  // UNIT_TESTING defined

MatchParams& MatchParams::reset(
    bool skipPatchInvariantParts /* = true*/) noexcept {
  mcPatchApprox = nullopt;
  patchApprox = nullopt;
  ssim = absCorr = fg = bg = contrast = sdevFg = sdevBg = sdevEdge =
      symDensity = mcsOffset = nullopt;

  if (!skipPatchInvariantParts) {
    mcPatch = nullopt;
    patchSq = blurredPatch = blurredPatchSq = variancePatch = nullopt;
    patchSum = normPatchMinMiu = 0.;
  }
  return *this;
}

void MatchParams::computeMean(const Mat& patch,
                              const Mat& mask,
                              optional<double>& miu) noexcept {
  if (miu)
    return;

  miu = *mean(patch, mask).val;
  assert(*miu > -EPS && *miu < EPSp255);
}

void MatchParams::computeFg(const Mat& patch,
                            const ISymData& symData) noexcept {
  computeMean(patch, symData.getMask(ISymData::MaskType::Fg), fg);
}

void MatchParams::computeBg(const Mat& patch,
                            const ISymData& symData) noexcept {
  computeMean(patch, symData.getMask(ISymData::MaskType::Bg), bg);
}

void MatchParams::computeContrast(const Mat& patch,
                                  const ISymData& symData) noexcept {
  if (contrast)
    return;

  computeFg(patch, symData);
  computeBg(patch, symData);

  contrast = fg.value() - bg.value();
  assert(abs(contrast.value()) < 255.5);
}

void MatchParams::computeSdev(const Mat& patch,
                              const Mat& mask,
                              optional<double>& miu,
                              optional<double>& sdev) noexcept {
  if (sdev)
    return;

  if (miu) {
    sdev = norm(patch - miu.value(), NORM_L2, mask) / sqrt(countNonZero(mask));
  } else {
    Scalar miu_, sdev_;
    meanStdDev(patch, miu_, sdev_, mask);
    miu = *miu_.val;
    sdev = *sdev_.val;
  }
  assert(*sdev < EPSpSdevMaxFgBg);
}

void MatchParams::computeSdevFg(const Mat& patch,
                                const ISymData& symData) noexcept {
  computeSdev(patch, symData.getMask(ISymData::MaskType::Fg), fg, sdevFg);
}

void MatchParams::computeSdevBg(const Mat& patch,
                                const ISymData& symData) noexcept {
  computeSdev(patch, symData.getMask(ISymData::MaskType::Bg), bg, sdevBg);
}

void MatchParams::computePatchApprox(const Mat& patch,
                                     const ISymData& symData) noexcept {
  if (patchApprox)
    return;

  computeContrast(patch, symData);

  if (contrast.value() == 0.) {
    patchApprox = Mat(patch.rows, patch.cols, CV_64FC1, Scalar(bg.value()));
    return;
  }

  patchApprox = bg.value() + symData.getMask(ISymData::MaskType::GroundedSym) *
                                 (contrast.value() / symData.getDiffMinMax());
}

void MatchParams::computeSdevEdge(const Mat& patch,
                                  const ISymData& symData) noexcept {
  if (sdevEdge)
    return;

  const Mat& edgeMask = symData.getMask(ISymData::MaskType::Edge);
  const int cnz = countNonZero(edgeMask);
  if (cnz == 0) {
    sdevEdge = 0.;
    return;
  }

  computePatchApprox(patch, symData);

  sdevEdge = norm(patch, patchApprox.value(), NORM_L2, edgeMask) / sqrt(cnz);
  assert(*sdevEdge < EPSpSdevMaxEdge);
}

void MatchParams::computeSymDensity(const ISymData& symData) noexcept {
  if (symDensity)
    return;

  // The method 'MatchAspect::score(const MatchParams &mp, const CachedData
  // &cachedData)' needs symData.avgPixVal stored within MatchParams mp. That's
  // why the mere value copy from below:
  symDensity = symData.getAvgPixVal();
  assert(*symDensity < EPSp1);
}

void MatchParams::computePatchSum(const Mat& patch) noexcept {
  if (patchSum)
    return;

  patchSum = *sum(patch).val;
}

void MatchParams::computePatchSq(const Mat& patch) noexcept {
  if (patchSq)
    return;

  patchSq = patch.mul(patch);
}

void MatchParams::computeMcPatch(const Mat& patch,
                                 const CachedData& cachedData) noexcept {
  if (mcPatch)
    return;

  computePatchSum(patch);

  Mat temp;
  double mcX = 0., mcY = 0.;

  cv::reduce(patch, temp, 0, cv::REDUCE_SUM);  // sum all rows
  mcX = temp.dot(cachedData.getConsec());

  cv::reduce(patch, temp, 1, cv::REDUCE_SUM);  // sum all columns
  mcY = temp.t().dot(cachedData.getConsec());

  mcPatch = Point2d(mcX, mcY) / (patchSum.value() * cachedData.getSz_1());
  assert(mcPatch->x > -EPS && mcPatch->x < EPSp1);
  assert(mcPatch->y > -EPS && mcPatch->y < EPSp1);
}

void MatchParams::computeMcPatchApprox(const Mat& patch,
                                       const ISymData& symData,
                                       const CachedData&) noexcept {
  if (mcPatchApprox)
    return;

  computeContrast(patch, symData);
  computeSymDensity(symData);

  // Obtaining glyph's mass center
  const double k = symDensity.value() * contrast.value(),
               delta = .5 * bg.value(), denominator = k + bg.value();
  if (denominator == 0.)
    mcPatchApprox = CachedData::MassCenters::unitSquareCenter();
  else
    mcPatchApprox = (k * symData.getMc() + Point2d(delta, delta)) / denominator;
  assert(mcPatchApprox->x > -EPS && mcPatchApprox->x < EPSp1);
  assert(mcPatchApprox->y > -EPS && mcPatchApprox->y < EPSp1);
}

void MatchParams::computeMcsOffset(const Mat& patch,
                                   const ISymData& symData,
                                   const CachedData& cachedData) noexcept {
  if (mcsOffset)
    return;

  computeMcPatch(patch, cachedData);
  computeMcPatchApprox(patch, symData, cachedData);

  mcsOffset = norm(mcPatch.value() - mcPatchApprox.value());
  assert(mcsOffset < EPSpSqrt2);
}
