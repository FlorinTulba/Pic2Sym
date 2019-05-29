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

#ifdef UNIT_TESTING
#error Should not include this header when UNIT_TESTING is defined

#else  // UNIT_TESTING not defined

#ifndef H_SLIDER_CONVERSION_BASE
#define H_SLIDER_CONVERSION_BASE

#pragma warning(push, 0)

#include <algorithm>
#include <memory>

#pragma warning(pop)

/// Base class for defining the parameters used while interpreting/generating
/// the value of a slider
class SliderConvParams /*abstract*/ {
 public:
  virtual ~SliderConvParams() noexcept {}

  // Slicing prevention
  SliderConvParams(const SliderConvParams&) = delete;
  SliderConvParams(SliderConvParams&&) = delete;
  SliderConvParams& operator=(const SliderConvParams&) = delete;
  SliderConvParams& operator=(SliderConvParams&&) = delete;

  int getMaxSlider() const noexcept { return maxSlider; }

 protected:
  /// Ensure a positive maxSlider field
  explicit SliderConvParams(int maxSlider_) noexcept
      : maxSlider((std::max)(1, maxSlider_)) {}

 private:
  int maxSlider;  ///< largest slider value (at least 1)
};

/// Base class for performing conversions from and to slider range
class SliderConverter /*abstract*/ {
 public:
  // 'sp' is supposed to not change for the original / copy
  SliderConverter(const SliderConverter&) = delete;
  SliderConverter(SliderConverter&&) = delete;
  void operator=(const SliderConverter&) = delete;
  void operator=(SliderConverter&&) = delete;

  virtual ~SliderConverter() noexcept {}

  virtual double fromSlider(int sliderPos) const noexcept = 0;
  virtual int toSlider(double actualValue) const noexcept = 0;

 protected:
  /// Take ownership of the parameter
  explicit SliderConverter(std::unique_ptr<const SliderConvParams> sp_) noexcept
      : sp(std::move(sp_)) {}

  /// Parameters required for interpreting/generating slider values
  const std::unique_ptr<const SliderConvParams> sp;
};

#endif  // H_SLIDER_CONVERSION_BASE

#endif  // UNIT_TESTING
