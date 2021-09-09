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

#include "tinySym.h"

#include "pixMapSymBase.h"

#pragma warning(push, 0)

#include <opencv2/imgproc/imgproc.hpp>

#pragma warning(pop)

using namespace std;
using namespace cv;

namespace pic2sym {

extern unsigned TinySymsSz();

namespace syms {

namespace {

const unsigned TinySymsSize{TinySymsSz()};
const unsigned RefSymSz{TinySymsSize * ITinySym::RatioRefTiny};
const unsigned DiagsCountTinySym{2U * TinySymsSize - 1U};

const double invTinySymSz{1. / TinySymsSize};
const double invTinySymArea{invTinySymSz * invTinySymSz};
const double invDiagsCountTinySym{1. / DiagsCountTinySym};

const Size SizeTinySyms{(int)TinySymsSize, (int)TinySymsSize};

}  // anonymous namespace

TinySym::TinySym(unsigned long code_ /* = ULONG_MAX*/,
                 size_t symIdx_ /* = 0ULL*/) noexcept
    : SymData{code_, symIdx_},
      // 0. parameter prevents using initializer_list ctor of Mat
      mat{(int)TinySymsSize, (int)TinySymsSize, CV_64FC1, 0.},
      hAvgProj{1, (int)TinySymsSize, CV_64FC1, 0.},
      vAvgProj{(int)TinySymsSize, 1, CV_64FC1, 0.},
      backslashDiagAvgProj{1, (int)DiagsCountTinySym, CV_64FC1, 0.},
      slashDiagAvgProj{1, (int)DiagsCountTinySym, CV_64FC1, 0.} {}

TinySym::TinySym(const IPixMapSym& refSym) noexcept
    : SymData{refSym.getSymCode(), refSym.getSymIdx(), refSym.getAvgPixVal(),
              refSym.getMc()},
      // 0. parameter prevents using initializer_list ctor of Mat
      backslashDiagAvgProj{1, (int)DiagsCountTinySym, CV_64FC1, 0.},
      slashDiagAvgProj{1, (int)DiagsCountTinySym, CV_64FC1, 0.} {
  const Mat refSymMat{refSym.toMatD01(RefSymSz)};

  Mat tinySymMat;
  resize(refSymMat, tinySymMat, SizeTinySyms, 0., 0., INTER_AREA);

  // keep the double type for negSym of tiny symbols
  setNegSym(255. - 255. * tinySymMat);

  SymData::computeFields(tinySymMat, *this, true);

  mat = getMask(MaskType::GroundedSym).clone();

  // computing average projections
  cv::reduce(mat, hAvgProj, 0, cv::REDUCE_AVG);
  cv::reduce(mat, vAvgProj, 1, cv::REDUCE_AVG);

  Mat flippedMat;
  flip(mat, flippedMat, 1);  // flip around vertical axis
  for (int diagIdx{-(int)TinySymsSize + 1}, i{}; diagIdx < (int)TinySymsSize;
       ++diagIdx, ++i) {
    const Mat backslashDiag{mat.diag(diagIdx)};
    backslashDiagAvgProj.at<double>(i) = *mean(backslashDiag).val;

    const Mat slashDiag{flippedMat.diag(-diagIdx)};
    slashDiagAvgProj.at<double>(i) = *mean(slashDiag).val;
  }

  // Ensuring the sum of all elements of the following matrices is in [0..1]
  // range
  mat *= invTinySymArea;
  hAvgProj *= invTinySymSz;
  vAvgProj *= invTinySymSz;
  backslashDiagAvgProj *= invDiagsCountTinySym;
  slashDiagAvgProj *= invDiagsCountTinySym;
}

const Mat& TinySym::getMat() const noexcept {
  return mat;
}

const Mat& TinySym::getHAvgProj() const noexcept {
  return hAvgProj;
}

const Mat& TinySym::getVAvgProj() const noexcept {
  return vAvgProj;
}

const Mat& TinySym::getBackslashDiagAvgProj() const noexcept {
  return backslashDiagAvgProj;
}

const Mat& TinySym::getSlashDiagAvgProj() const noexcept {
  return slashDiagAvgProj;
}

#pragma warning(disable : WARN_EXPR_ALWAYS_FALSE)
bool TinySym::olderVersionDuringLastIO() noexcept {
  return SymData::olderVersionDuringLastIO() || VersionFromLast_IO_op < Version;
}
#pragma warning(default : WARN_EXPR_ALWAYS_FALSE)

#pragma warning(disable : WARN_THROWS_ALTHOUGH_NOEXCEPT)
void TinySym::setMat(const Mat& mat_) noexcept {
  // Check parameter and throw invalid_argument if invalid
  mat = mat_;
}
#pragma warning(default : WARN_THROWS_ALTHOUGH_NOEXCEPT)

#pragma warning(disable : WARN_THROWS_ALTHOUGH_NOEXCEPT)
void TinySym::setHAvgProj(const Mat& hAvgProj_) noexcept {
  // Check parameter and throw invalid_argument if invalid
  hAvgProj = hAvgProj_;
}
#pragma warning(default : WARN_THROWS_ALTHOUGH_NOEXCEPT)

#pragma warning(disable : WARN_THROWS_ALTHOUGH_NOEXCEPT)
void TinySym::setVAvgProj(const Mat& vAvgProj_) noexcept {
  // Check parameter and throw invalid_argument if invalid
  vAvgProj = vAvgProj_;
}
#pragma warning(default : WARN_THROWS_ALTHOUGH_NOEXCEPT)

#pragma warning(disable : WARN_THROWS_ALTHOUGH_NOEXCEPT)
void TinySym::setSlashDiagAvgProj(const Mat& slashDiagAvgProj_) noexcept {
  // Check parameter and throw invalid_argument if invalid
  slashDiagAvgProj = slashDiagAvgProj_;
}
#pragma warning(default : WARN_THROWS_ALTHOUGH_NOEXCEPT)

#pragma warning(disable : WARN_THROWS_ALTHOUGH_NOEXCEPT)
void TinySym::setBackslashDiagAvgProj(
    const Mat& backslashDiagAvgProj_) noexcept {
  // Check parameter and throw invalid_argument if invalid
  backslashDiagAvgProj = backslashDiagAvgProj_;
}
#pragma warning(default : WARN_THROWS_ALTHOUGH_NOEXCEPT)

}  // namespace syms
}  // namespace pic2sym
