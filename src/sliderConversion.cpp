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

#ifdef UNIT_TESTING
#error Should not include this file when UNIT_TESTING is defined

#else  // UNIT_TESTING not defined

#include "misc.h"
#include "sliderConversion.h"
#include "warnings.h"

#pragma warning(push, 0)

#include <cassert>

#pragma warning(pop)

using namespace std;

#pragma warning(disable : WARN_THROWS_ALTHOUGH_NOEXCEPT)
ProportionalSliderValue::Params::Params(int maxSlider_,
                                        double maxVal_) noexcept(!UT)
    : SliderConvParams(maxSlider_), maxVal(maxVal_) {
  if (maxVal_ <= 0.)
    THROW_WITH_CONST_MSG(__FUNCTION__ " should get maxVal_ > 0!",
                         invalid_argument);
}
#pragma warning(default : WARN_THROWS_ALTHOUGH_NOEXCEPT)

#pragma warning(disable : WARN_THROWS_ALTHOUGH_NOEXCEPT)
ProportionalSliderValue::ProportionalSliderValue(
    unique_ptr<const Params> sp_) noexcept(!UT)
    : SliderConverter(move(sp_)) {
  if (!sp)  // Testing on sp, since sp_ was moved to sp
    THROW_WITH_CONST_MSG(__FUNCTION__ " received a nullptr Params parameter",
                         invalid_argument);
}
#pragma warning(default : WARN_THROWS_ALTHOUGH_NOEXCEPT)

double ProportionalSliderValue::fromSlider(int sliderPos) const noexcept {
  const Params* lsp = dynamic_cast<const Params*>(sp.get());
  assert(lsp != nullptr);  // sp != nullptr and  Params derives SliderConvParams
  return (double)sliderPos * lsp->getMaxVal() / (double)lsp->getMaxSlider();
}

int ProportionalSliderValue::toSlider(double actualValue) const noexcept {
  const Params* lsp = dynamic_cast<const Params*>(sp.get());
  assert(lsp != nullptr);  // sp != nullptr and  Params derives SliderConvParams
  return int(.5 + actualValue * (double)lsp->getMaxSlider() /
                      lsp->getMaxVal());  // performs rounding
}

#endif  // UNIT_TESTING
