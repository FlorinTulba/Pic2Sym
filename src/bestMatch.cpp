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

#include "bestMatch.h"

#include "matchParams.h"
#include "warnings.h"

using namespace std;
using namespace gsl;
using namespace cv;

namespace pic2sym {

#if defined(_DEBUG) || defined(UNIT_TESTING)

extern constinit not_null<cwzstring<> const> Comma;

#endif  // defined(_DEBUG) || defined(UNIT_TESTING)

namespace match {

BestMatch::BestMatch(const p2s::input::IPatch& patch_) noexcept
    : patch(patch_.clone()),
      params(patch_.nonUniform() ? make_unique<MatchParams>()
                                 : unique_ptr<MatchParams>{}) {
  Ensures(patch);  // clone of a reference
}

const p2s::input::IPatch& BestMatch::getPatch() const noexcept {
  return *patch;
}

const Mat& BestMatch::getApprox() const noexcept {
  return approx;
}

const optional<const IMatchParams*> BestMatch::getParams() const noexcept {
  if (params)
    return &(*params);
  return nullopt;
}

IMatchParamsRW& BestMatch::refParams() const noexcept {
  Expects(params);  // Don't call this for blur-only approximations
  return *params;
}

const optional<unsigned>& BestMatch::getSymIdx() const noexcept {
  return symIdx;
}

const optional<unsigned>& BestMatch::getLastPromisingNontrivialCluster()
    const noexcept {
  return lastPromisingNontrivialCluster;
}

BestMatch& BestMatch::setLastPromisingNontrivialCluster(
    unsigned clustIdx) noexcept {
  lastPromisingNontrivialCluster = clustIdx;
  return *this;
}

const optional<unsigned long>& BestMatch::getSymCode() const noexcept {
  return symCode;
}

double BestMatch::getScore() const noexcept {
  return score;
}

BestMatch& BestMatch::setScore(double score_) noexcept {
  score = score_;
  return *this;
}

BestMatch& BestMatch::reset() noexcept {
  score = 0.;
  symCode = nullopt;
  symIdx = lastPromisingNontrivialCluster = nullopt;
  pSymData = nullptr;
  approx.release();
  if (params)
    params->reset();  // keeps patch-invariant parameters
  return *this;
}

BestMatch& BestMatch::update(double score_,
                             unsigned long symCode_,
                             unsigned symIdx_,
                             const p2s::syms::ISymData& sd) noexcept {
  score = score_;
  symCode = symCode_;
  symIdx = symIdx_;
  pSymData = &sd;
  return *this;
}

#pragma warning(disable : WARN_THROWS_ALTHOUGH_NOEXCEPT)
BestMatch& BestMatch::updatePatchApprox(
    const p2s::cfg::IMatchSettings& ms) noexcept(!UT) {
  if (!pSymData) {
    approx = patch->getBlurred();
    return *this;
  }

  using namespace syms;

  const ISymData& dataOfBest = *pSymData;
  const ISymData::MatArray& matricesForBest = dataOfBest.getMasks();
  const Mat& groundedBest =
      matricesForBest[(size_t)ISymData::MaskType::GroundedSym];
  const auto patchSz = patch->getOrig().rows;

  Mat patchResult;
  if (patch->isColored()) {
    const Mat &fgMask = matricesForBest[(size_t)ISymData::MaskType::Fg],
              &bgMask = matricesForBest[(size_t)ISymData::MaskType::Bg];

    vector<Mat> channels;
    split(patch->getOrig(), channels);

    double diffFgBg{};
    const size_t channelsCount{size(channels)};
    for (Mat& ch : channels) {
      ch.convertTo(ch, CV_64FC1);  // processing double values

      const double miuFg{*mean(ch, fgMask).val};
      const double miuBg{*mean(ch, bgMask).val};
      const double newDiff{miuFg - miuBg};

      groundedBest.convertTo(ch, CV_8UC1, newDiff / dataOfBest.getDiffMinMax(),
                             miuBg);

      diffFgBg += abs(newDiff);
    }

    if (diffFgBg < channelsCount * ms.getBlankThreshold())
      patchResult = Mat{patchSz, patchSz, CV_8UC3, mean(patch->getOrig())};
    else
      merge(channels, patchResult);

  } else {
    // grayscale result
    EXPECTS_OR_REPORT_AND_THROW_CONST_MSG(
        params, logic_error,
        HERE.function_name() +
            " - params must be set; the patch shouldn't be uniform!"s);

    params->computeContrast(patch->getOrig(), *pSymData);

    if (abs(*params->getContrast()) < ms.getBlankThreshold())
      patchResult =
          Mat{patchSz, patchSz, CV_8UC1, Scalar{*mean(patch->getOrig()).val}};
    else
      groundedBest.convertTo(
          patchResult, CV_8UC1,
          *params->getContrast() / dataOfBest.getDiffMinMax(),
          *params->getBg());
  }

  approx = patchResult;

  // For non-hybrid result mode we're done
  if (!ms.isHybridResult())
    return *this;

  // Hybrid Result Mode - Combine the approximation with the blurred patch:
  // the less satisfactory the approximation is,
  // the more the weight of the blurred patch should be
  Scalar miu, sdevApproximation, sdevBlurredPatch;
  cv::meanStdDev(patch->getOrig() - approx, miu, sdevApproximation);
  cv::meanStdDev(patch->getOrig() - patch->getBlurred(), miu, sdevBlurredPatch);

  double totalSdevBlurredPatch{*sdevBlurredPatch.val};
  double totalSdevApproximation{*sdevApproximation.val};
  if (patch->isColored()) {
    totalSdevBlurredPatch +=
        sdevBlurredPatch.val[1ULL] + sdevBlurredPatch.val[2ULL];
    totalSdevApproximation +=
        sdevApproximation.val[1ULL] + sdevApproximation.val[2ULL];
  }
  const double sdevSum{totalSdevBlurredPatch + totalSdevApproximation};
  const double weight{(sdevSum > 0.) ? (totalSdevApproximation / sdevSum) : 0.};
  Mat combination;
  cv::addWeighted(patch->getBlurred(), weight, approx, 1. - weight, 0.,
                  combination);
  approx = combination;

  return *this;
}
#pragma warning(default : WARN_THROWS_ALTHOUGH_NOEXCEPT)

#if defined _DEBUG || defined UNIT_TESTING

bool BestMatch::isUnicode() const noexcept {
  return unicode;
}

BestMatch& BestMatch::setUnicode(bool unicode_) noexcept {
  unicode = unicode_;
  return *this;
}

wstring BestMatch::toWstring() const noexcept {
  wostringstream wos;
  if (!symCode)
    wos << L"--";
  else {
    const unsigned long theSymCode{*symCode};
    if (unicode) {
      switch (theSymCode) {
        case (unsigned long)',':
          wos << L"COMMA";
          break;
        case (unsigned long)'(':
          wos << L"OPEN_PAR";
          break;
        case (unsigned long)')':
          wos << L"CLOSE_PAR";
          break;
        default:
          // for other characters, check if they can be displayed on the current
          // console
          if (wos << (wchar_t)theSymCode) {
            // when they can be displayed, add in () their code
            wos << '(' << theSymCode << ')';
          } else {        // when they can't be displayed, show just their code
            wos.clear();  // clear the error first
            wos << theSymCode;
          }
      }
    } else
      wos << theSymCode;
  }

  wos << Comma << score;
  if (params)
    wos << Comma << *params;

  return wos.str();
}

wostream& operator<<(wostream& wos, const IBestMatch& bm) noexcept {
  wos << bm.toWstring();
  return wos;
}

#endif  // _DEBUG || UNIT_TESTING

}  // namespace match
}  // namespace pic2sym
