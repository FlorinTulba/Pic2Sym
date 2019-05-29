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
#include "fontEngineBase.h"
#include "misc.h"

#pragma warning(push, 0)

#include <numeric>

#define _USE_MATH_DEFINES
#include <math.h>

#pragma warning(pop)

using namespace std;
using namespace cv;

extern const double DirSmooth_DesiredBaseForCenterAndCornerMcs;

/*
DirectionalSmoothness should depend MOSTLY on the angleFactor, and LESS on
mcsOffsetFactor. See these variables defined and explained in
DirectionalSmoothness::score() method.

Here's the case that catches the undesired high impact of mcsOffsetFactor:

Mass center 1 is in the square's center, while the second mass center is in one
corner. Thus, angleFactor has its largest possible value (MaxAngleFactor = 2 *
(2-sqrt(2)) = ~1.17), since the angle between the mc-s is 0 (because the angle
between the center and any point is 0). But mcsOffsetFactor is computed based on
a large offset between mc-s: maxMcDist/2. To mention that maxMcDist is sqrt(2).

As MaxAngleFactor is slightly over 1, it's imperative that mcsOffsetFactor to
remain closely below 1, to convey the message that the angle is indeed a
convenient one. However, computing mcsOffsetFactor like (MaxMcDist - mcsOffset)
/ (MaxMcDist - PreferredMaxMcDist) delivers a value ~0.55 for mcsOffset =
maxMcDist/2.

The constant DirSmooth_DesiredBaseForCenterAndCornerMcs makes sure that
mcsOffsetFactor will have its value when mcsOffset = maxMcDist/2, instead of
0.55.

mcsOffsetFactor will further be 1 for mcsOffset = PreferredMaxMcDist

Considering the linear rule for mcsOffsetFactor:
mcsOffsetFactor = a * mcsOffset + b

we have following system of 2 equations:
a * PreferredMaxMcDist + b = 1
a * maxMcDist/2        + b = DirSmooth_DesiredBaseForCenterAndCornerMcs /
MaxAngleFactor

Solution:
a = (DirSmooth_DesiredBaseForCenterAndCornerMcs / MaxAngleFactor - 1) /
(maxMcDist/2 - PreferredMaxMcDist) b = 1 - a * PreferredMaxMcDist
*/
const double CachedData::MassCenters::a_mcsOffsetFactor() noexcept {
  static constexpr double maxMcDist = M_SQRT2,
                          MaxAngleFactor = 2. * (2. - M_SQRT2),
                          DenominatorA_mcsOffsetFactor =
                              .5 * maxMcDist - preferredMaxMcDist;
  static const double NumeratorA_mcsOffsetFactor =
                          DirSmooth_DesiredBaseForCenterAndCornerMcs /
                              MaxAngleFactor -
                          1.,
                      result = NumeratorA_mcsOffsetFactor /
                               DenominatorA_mcsOffsetFactor;
  return result;
}
const double CachedData::MassCenters::b_mcsOffsetFactor() noexcept {
  static const double result = 1. - a_mcsOffsetFactor() * preferredMaxMcDist;
  return result;
}

const Point2d& CachedData::MassCenters::unitSquareCenter() noexcept {
  static const Point2d center(.5, .5);
  return center;
}

CachedData::CachedData(bool forTinySyms_ /* = false*/) noexcept
    : forTinySyms(forTinySyms_) {}

CachedDataRW::CachedDataRW(bool forTinySyms_ /* = false*/) noexcept
    : CachedData(forTinySyms_) {}

void CachedDataRW::useNewSymSize(unsigned sz_) noexcept {
  const double szd = (double)sz_;
  sz_1 = szd - 1.;
  szSq = szd * szd;

  consec = Mat(1, (int)sz_, CV_64FC1);
  iota(BOUNDS_FOR_ITEM_TYPE(consec, double), (double)0.);
}

void CachedDataRW::update(const IFontEngine& fe_) noexcept {
  smallGlyphsCoverage = fe_.smallGlyphsCoverage();
}

void CachedDataRW::update(unsigned sz_, const IFontEngine& fe_) noexcept {
  useNewSymSize(sz_);
  update(fe_);
}
