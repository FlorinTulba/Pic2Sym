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

#ifndef UNIT_TESTING

#include "sliderConversion.h"

#include "misc.h"
#include "warnings.h"

#pragma warning(push, 0)

#include <gsl/gsl>

#pragma warning(pop)

using namespace std;
using namespace gsl;

namespace pic2sym::ui {

#pragma warning(disable : WARN_THROWS_ALTHOUGH_NOEXCEPT)
ProportionalSliderValue::Params::Params(int maxSlider_,
                                        double maxVal_) noexcept(!UT)
    : SliderConvParams(maxSlider_), maxVal(maxVal_) {
  EXPECTS_OR_REPORT_AND_THROW_CONST_MSG(
      maxVal_ > 0., invalid_argument,
      HERE.function_name() + " should get maxVal_ > 0!"s);
}
#pragma warning(default : WARN_THROWS_ALTHOUGH_NOEXCEPT)

#pragma warning(disable : WARN_THROWS_ALTHOUGH_NOEXCEPT)
ProportionalSliderValue::ProportionalSliderValue(
    unique_ptr<const Params> sp_) noexcept(!UT)
    : SliderConverter(move(sp_)) {
  EXPECTS_OR_REPORT_AND_THROW_CONST_MSG(
      sp, invalid_argument,
      HERE.function_name() + " received a nullptr Params parameter"s);
}
#pragma warning(default : WARN_THROWS_ALTHOUGH_NOEXCEPT)

double ProportionalSliderValue::fromSlider(int sliderPos) const noexcept {
  not_null<const Params*> lsp = dynamic_cast<const Params*>(sp.get());

  // Params derives SliderConvParams
  return (double)sliderPos * lsp->getMaxVal() / (double)lsp->getMaxSlider();
}

int ProportionalSliderValue::toSlider(double actualValue) const noexcept {
  not_null<const Params*> lsp = dynamic_cast<const Params*>(sp.get());

  // Params derives SliderConvParams
  return int(.5 + actualValue * (double)lsp->getMaxSlider() /
                      lsp->getMaxVal());  // performs rounding
}

}  // namespace pic2sym::ui

#endif  // UNIT_TESTING
