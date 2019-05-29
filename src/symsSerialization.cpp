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

#include "symsSerialization.h"

#pragma warning(push, 0)

#include <fstream>
#include <iomanip>

#pragma warning(pop)

using namespace std;
using namespace cv;

namespace ut {
void saveSymsSelection(const string& destFile,
                       const list<Mat>& symsSelection) noexcept {
  ofstream ofs(destFile);

  // First line specifies the number of symbols in the list
  ofs << symsSelection.size() << endl;
  for (const Mat& m : symsSelection) {
    const int rows = m.rows, cols = m.cols;

    // Every symbol is preceded by a header with the number of rows and columns
    ofs << rows << ' ' << cols << endl;
    for (int r = 0; r < rows; ++r) {
      for (int c = 0; c < cols; ++c)
        ofs << setw(3) << right  // align values to the right
            << (unsigned)m.at<unsigned char>(r, c)
            << ' ';  // pixel values are delimited by space
      ofs << endl;
    }
  }
}

void loadSymsSelection(const string& srcFile,
                       vector<Mat>& symsSelection) noexcept {
  ifstream ifs(srcFile);

  unsigned symsCount = 0U;

  // First line specifies the number of symbols in the list
  ifs >> symsCount;
  symsSelection.reserve(symsCount);

  for (unsigned symIdx = 0U; symIdx < symsCount; ++symIdx) {
    int rows, cols;

    // Every symbol is preceded by a header with the number of rows and columns
    ifs >> rows >> cols;
    assert(rows == cols);  // Hint

    Mat symMat(rows, cols, CV_8UC1);
    for (int r = 0; r < rows; ++r) {
      for (int c = 0; c < cols; ++c) {
        unsigned v;
        ifs >> v;
        symMat.at<unsigned char>(r, c) = (unsigned char)v;
      }
    }

    symsSelection.push_back(symMat);
  }
}
}  // namespace ut
