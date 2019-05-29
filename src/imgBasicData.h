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

#ifndef H_IMG_BASIC_DATA
#define H_IMG_BASIC_DATA

#pragma warning(push, 0)

#include <string>

#include <opencv2/core/core.hpp>

#pragma warning(pop)

/// Interface for providing basic information about an image
class IBasicImgData /*abstract*/ {
 public:
  /// @return the original image
  virtual const cv::Mat& original() const noexcept = 0;

  /// @return the stem of the image file name
  virtual const std::string& name() const noexcept = 0;

  /// color / grayscale image
  virtual bool isColor() const noexcept = 0;

  virtual ~IBasicImgData() noexcept {}

  // Slicing prevention
  IBasicImgData(const IBasicImgData&) = delete;
  IBasicImgData(IBasicImgData&&) = delete;
  IBasicImgData& operator=(const IBasicImgData&) = delete;
  IBasicImgData& operator=(IBasicImgData&&) = delete;

 protected:
  constexpr IBasicImgData() noexcept {}
};

#endif  // H_IMG_BASIC_DATA
