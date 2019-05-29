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

// Serialization support for cv::Mat

// Code adapted from the one provided by user1520427 in thread:
// http://stackoverflow.com/questions/4170745/serializing-opencv-mat-vec3f

#ifndef H_MAT_SERIALIZATION
#define H_MAT_SERIALIZATION

#include "misc.h"
#include "warnings.h"

#pragma warning(push, 0)

#include <opencv2/core/core.hpp>

#include <boost/serialization/array.hpp>

#pragma warning(pop)

namespace boost::serialization {

/**
cv::Mat serialization

Any throwing (for corrupt/deprecated ar) makes sense to be unit tested.
*/
template <class Archive>
void serialize(Archive& ar,
               cv::Mat& mat,
               const unsigned /*version*/) noexcept(!UT) {
  ar& mat.rows& mat.cols;  // These can be both 0 for empty matrices

  ar& mat.flags;  // provides the matrix type and continuity flag

  const bool continuous = mat.isContinuous();

  if constexpr (Archive::is_loading::value)
    mat.create(mat.rows, mat.cols, mat.type());

  /*
  In OpenCV 4.1.0, Mat::elemSize() has an assert accepting only Mat::dims>0.
  Therefore, in Debug mode, when reading empty matrices the assert fails.

  To correct this and to be still able to read previously saved matrices,
  the call to Mat::elemSize() is circumvented when the matrices are empty:
  */
  size_t mat_total = 0ULL, mat_elemSize = 0ULL;
  if (0 != mat.rows * mat.cols) {
    mat_elemSize = mat.elemSize();
    mat_total = mat.total();
  }

  if (continuous) {
    const auto data_size = mat_total * mat_elemSize;
    ar& boost::serialization::make_array(mat.ptr(), data_size);
  } else {
    const auto row_size = mat.cols * mat_elemSize;
    for (int i = 0; i < mat.rows; i++)
      ar& boost::serialization::make_array(mat.ptr(i), row_size);
  }
}

}  // namespace boost::serialization

#endif  // H_MAT_SERIALIZATION
