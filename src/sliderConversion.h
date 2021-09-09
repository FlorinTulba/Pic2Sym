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

#ifdef UNIT_TESTING
#error Should not include this header when UNIT_TESTING is defined

#else  // UNIT_TESTING not defined

#ifndef H_SLIDER_CONVERSION
#define H_SLIDER_CONVERSION

#include "sliderConversionBase.h"

#include "misc.h"

namespace pic2sym::ui {

/// Performing conversions from and to slider range with a simple linear rule
class ProportionalSliderValue : public SliderConverter {
 public:
  /// Parameters used while interpreting/generating the value of a slider for a
  /// linear rule
  class Params : public SliderConvParams {
   public:
    /**
    Sets the upper limits for the domain range and the slider range
    @throw invalid_argument if maxVal_<=0

    Exception to be only reported, not handled
    */
    Params(int maxSlider_, double maxVal_) noexcept(!UT);

    /// Largest actual value
    double getMaxVal() const noexcept { return maxVal; }

   private:
    double maxVal;  ///< largest actual value
  };

  /**
  Applies a proportional rule for the slider
  @throw invalid_argument for nullptr sp_

  Exception to be only reported, not handled
  */
  explicit ProportionalSliderValue(std::unique_ptr<const Params> sp_) noexcept(
      !UT);
  virtual ~ProportionalSliderValue() noexcept = default;

  ProportionalSliderValue(const ProportionalSliderValue&) = delete;
  void operator=(const ProportionalSliderValue&) = delete;

  double fromSlider(int sliderPos) const noexcept override;
  int toSlider(double actualValue) const noexcept override;
};

}  // namespace pic2sym::ui

#endif  // H_SLIDER_CONVERSION

#endif  // UNIT_TESTING
