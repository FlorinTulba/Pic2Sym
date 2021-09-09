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

#include "clusterData.h"

#include "warnings.h"

#pragma warning(push, 0)

#include <gsl/gsl>

#pragma warning(pop)

using namespace std;
using namespace gsl;
using namespace cv;

namespace pic2sym::syms::inline cluster {

#pragma warning(disable : WARN_THROWS_ALTHOUGH_NOEXCEPT)
ClusterData::ClusterData(const VSymData& symsSet,
                         unsigned idxOfFirstSym_,
                         const vector<unsigned>& clusterSymIndices,
                         ISymsSupport& symsSupport) noexcept(!UT)
    : SymData(),
      idxOfFirstSym(idxOfFirstSym_),
      sz{narrow_cast<unsigned>(size(clusterSymIndices))} {
  EXPECTS_OR_REPORT_AND_THROW_CONST_MSG(
      !clusterSymIndices.empty(), invalid_argument,
      HERE.function_name() + " requires non-empty clusterSymIndices!"s);
  EXPECTS_OR_REPORT_AND_THROW_CONST_MSG(
      !symsSet.empty(), invalid_argument,
      HERE.function_name() + " requires non-empty symsSet!"s);

  const double invClusterSz{1. / sz};
  const ISymData& firstSym = *symsSet[0ULL];
  const int symSz{firstSym.getNegSym().rows};
  double avgPixVal_{};
  Point2d mc_;
  vector<not_null<const ISymData*>> clusterSyms;
  clusterSyms.reserve((size_t)sz);

  for (const auto clusterSymIdx : clusterSymIndices) {
    const ISymData& symData = *symsSet[(size_t)clusterSymIdx];
    clusterSyms.push_back(&symData);

    // avgPixVal and mc are taken from the normal-size symbol (guaranteed to be
    // non-blank)
    avgPixVal_ += symData.getAvgPixVal();
    mc_ += symData.getMc();
  }
  setAvgPixVal(avgPixVal_ * invClusterSz);
  setMc(mc_ * invClusterSz);

  Mat synthesizedSym, negSym_;
  symsSupport.computeClusterRepresentative(clusterSyms, symSz, invClusterSz,
                                           synthesizedSym, negSym_);
  setNegSym(negSym_);

  SymData::computeFields(synthesizedSym, *this, symsSupport.usingTinySymbols());
}
#pragma warning(default : WARN_THROWS_ALTHOUGH_NOEXCEPT)

unsigned ClusterData::getIdxOfFirstSym() const noexcept {
  return idxOfFirstSym;
}
unsigned ClusterData::getSz() const noexcept {
  return sz;
}

}  // namespace pic2sym::syms::inline cluster
