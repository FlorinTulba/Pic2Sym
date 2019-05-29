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

#ifndef H_RESIZED_IMG
#define H_RESIZED_IMG

#include "resizedImgBase.h"

#pragma warning(push, 0)

#include <compare>

#pragma warning(pop)

class IfImgSettings;  // forward declaration

/// ResizedImg is the version of the original image which is ready to be
/// transformed
class ResizedImg : public IResizedImg {
 public:
  /**
  If possible, it adapts the original image to the parameters of the
  transformation:
  - The image must fit within prescribed bounds
  - The image must preserve its original aspect ratio and cannot become larger

  @throw logic_error when no image was set yet (source is empty)

  Exception to be only reported, not handled
  */
  ResizedImg(const cv::Mat& source,
             const IfImgSettings& is,
             unsigned patchSz_) noexcept(!UT);

  const cv::Mat& get() const noexcept final { return res; }

#ifdef __cpp_lib_three_way_comparison
  std::strong_equality operator<=>(const ResizedImg& other) const noexcept;

#else   // __cpp_lib_three_way_comparison not defined
  bool operator==(const ResizedImg& other) const noexcept;
  bool operator!=(const ResizedImg& other) const noexcept {
    return !(*this == other);
  }
#endif  // __cpp_lib_three_way_comparison

 private:
  cv::Mat res;  ///< the resized image

  unsigned patchSz;  ///< patch / font size
};

#endif  // H_RESIZED_IMG
