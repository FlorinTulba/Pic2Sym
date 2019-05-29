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

#ifndef H_BLUR_BASE
#define H_BLUR_BASE

#include "misc.h"

#pragma warning(push, 0)

#include <opencv2/core/core.hpp>

#pragma warning(pop)

/// Interface for the BlurEngine
class IBlurEngine /*abstract*/ {
 public:
  /**
  Blurring toBlur into blurred depending on forTinySym.

  @param toBlur is a single channel matrix with values of type double
  @param blurred the result of the blur (not initialized when method gets
  called)
  @param forTinySym demands generating a Gaussian blur with smaller window and
  standard deviation for tiny symbols

  @throw invalid_argument for empty toBlur or if it contains not double values

  Exception should be only reported, not handled.
  */
  virtual void process(const cv::Mat& toBlur,
                       cv::Mat& blurred,
                       bool forTinySym) const noexcept(!UT) = 0;

  virtual ~IBlurEngine() noexcept {}

  // Slicing prevention
  IBlurEngine(const IBlurEngine&) = delete;
  IBlurEngine(IBlurEngine&&) = delete;
  IBlurEngine& operator=(const IBlurEngine&) = delete;
  IBlurEngine& operator=(IBlurEngine&&) = delete;

 protected:
  constexpr IBlurEngine() noexcept {}
};

#endif  // H_BLUR_BASE
