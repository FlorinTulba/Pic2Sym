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

#ifndef H_PATCH_BASE
#define H_PATCH_BASE

#pragma warning(push, 0)

#include <memory>

#include <opencv2/core/core.hpp>

#pragma warning(pop)

/**
Base for the Patch class.

It decides whether this patch needs approximation or not - uniform patches
don't produce interesting approximations.
*/
class IPatch /*abstract*/ {
 public:
  /// The patch to approximate
  virtual const cv::Mat& getOrig() const noexcept = 0;

  /// The blurred version of the orig
  virtual const cv::Mat& getBlurred() const noexcept = 0;

  /// Is the patch color or grayscale?
  virtual bool isColored() const noexcept = 0;

  /// Patches that appear uniform use 'blurred' as approximation
  virtual bool nonUniform() const noexcept = 0;

  /**
  Specifies which matrix to use during the approximation process (gray, of type
  double)
  @throw logic_error if called for uniformous patches

  Exception to be only reported, not handled
  */
  virtual const cv::Mat& matrixToApprox() const noexcept(!UT) = 0;

  virtual ~IPatch() noexcept {}

  // If slicing is observed and becomes a severe problem, use `= delete` for all
  IPatch(const IPatch&) noexcept = default;
  IPatch(IPatch&&) noexcept = default;
  IPatch& operator=(const IPatch&) noexcept = default;
  IPatch& operator=(IPatch&&) noexcept = default;

  /// @return a clone of itself
  virtual std::unique_ptr<const IPatch> clone() const noexcept = 0;

 protected:
  constexpr IPatch() noexcept {}
};

#endif  // H_PATCH_BASE
