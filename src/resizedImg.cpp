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

#include "imgSettingsBase.h"
#include "misc.h"
#include "resizedImg.h"
#include "warnings.h"

#pragma warning(push, 0)

#include <opencv2/imgproc/imgproc.hpp>

#pragma warning(pop)

using namespace std;
using namespace cv;

#pragma warning(disable : WARN_THROWS_ALTHOUGH_NOEXCEPT)
ResizedImg::ResizedImg(const Mat& source,
                       const IfImgSettings& is,
                       unsigned patchSz_) noexcept(!UT)
    : patchSz(patchSz_) {
  if (source.empty())
    THROW_WITH_CONST_MSG("No image set yet", logic_error);

  const int initW = source.cols, initH = source.rows;
  const double initAr = initW / (double)initH;
  unsigned w = min(patchSz * is.getMaxHSyms(), (unsigned)initW),
           h = min(patchSz * is.getMaxVSyms(), (unsigned)initH);
  w -= w % patchSz;
  h -= h % patchSz;

  if (w / (double)h > initAr) {
    w = (unsigned)round(h * initAr);
    w -= w % patchSz;
  } else {
    h = (unsigned)round(w / initAr);
    h -= h % patchSz;
  }

  if (w == (unsigned)initW && h == (unsigned)initH)
    res = source;
  else {
    resize(source, res, Size((int)w, (int)h), 0, 0, cv::INTER_AREA);
    cout << "Resized to (" << w << 'x' << h << ")\n";
  }

  cout << "The result will be " << w / patchSz << " symbols wide and "
       << h / patchSz << " symbols high.\n"
       << endl;
}
#pragma warning(default : WARN_THROWS_ALTHOUGH_NOEXCEPT)

#ifdef __cpp_lib_three_way_comparison
strong_equality ResizedImg::operator<=>(const ResizedImg& other) const
    noexcept {
  // if (this == &other) // Costly to always perform. Harmless & cheap if cut
  //  return strong_equality::equivalent;

  if (const auto cmp = patchSz <=> other.patchSz; cmp != 0)
    return cmp;

  if (const auto cmp = res.size <=> other.res.size; cmp != 0)
    return cmp;

  if (const auto cmp = res.channels() <=> other.res.channels(); cmp != 0)
    return cmp;

  if (const auto cmp = res.type() <=> other.res.type(); cmp != 0)
    return cmp;

  const auto diffs = *sum(res != other.res).val / 255.;
  return (int)diffs <=> 0;
}

#else   // __cpp_lib_three_way_comparison not defined
bool ResizedImg::operator==(const ResizedImg& other) const noexcept {
  // if (this == &other) // Costly to always perform. Harmless & cheap if cut
  //  return true;

  if (patchSz != other.patchSz)
    return false;

  if (res.size != other.res.size)
    return false;

  if (res.channels() != other.res.channels())
    return false;

  if (res.type() != other.res.type())
    return false;

  const auto diffs = *sum(res != other.res).val / 255.;
  return (int)diffs == 0;
}
#endif  // __cpp_lib_three_way_comparison
