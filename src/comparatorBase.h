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

#ifndef UNIT_TESTING

#ifndef H_COMPARATOR_BASE
#define H_COMPARATOR_BASE

#include "viewsBase.h"

#pragma warning(push, 0)

#include <opencv2/core/core.hpp>

#pragma warning(pop)

extern const int Comparator_trackMax;
extern const double Comparator_defaultTransparency;

/**
Interface of a view which permits comparing the original image with the
transformed one.

A slider adjusts the transparency of the resulted image,
so that the original can be more or less visible.
*/
class IComparator /*abstract*/ : public virtual ICvWin {
 protected:
  IComparator() noexcept {}

 public:
  /**
  Setting the original image to be processed.
  @throw invalid_argument for an empty reference_

  Exception to be only reported, not handled
  */
  virtual void setReference(const cv::Mat& reference_) noexcept = 0;

  /**
  Setting the resulted image after processing.

  @throw invalid_argument for a result_ with other dimensions than reference_
  @throw logic_error if called before setReference()

  Exceptions to be only reported, not handled
  */
  virtual void setResult(const cv::Mat& result_,
                         int transparency = (int)
                             round(Comparator_defaultTransparency *
                                   Comparator_trackMax)) noexcept = 0;

  using ICvWin::resize;  // to remain visible after declaring the overload below
  virtual void resize() const noexcept = 0;
};

#endif  // H_COMPARATOR_BASE

#endif  // UNIT_TESTING not defined
