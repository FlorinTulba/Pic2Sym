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

#if defined _DEBUG || defined UNIT_TESTING

#include "bestMatchBase.h"
#include "matchParamsBase.h"
#include "warnings.h"

const std::wstring& COMMA() noexcept {
  static const std::wstring res(L",\t");
  return res;
}

std::wostream& operator<<(std::wostream& wos, const IMatchParams& mp) noexcept {
  wos << mp.toWstring();
  return wos;
}

std::wostream& operator<<(std::wostream& wos, const IBestMatch& bm) noexcept {
  wos << bm.toWstring();
  return wos;
}

#endif  // _DEBUG || UNIT_TESTING

#if defined _DEBUG && !defined UNIT_TESTING

#include "appStart.h"
#include "transformTrace.h"

using namespace std;
using namespace std::filesystem;

TransformTrace::TransformTrace(const string& studiedCase_,
                               unsigned sz_,
                               bool isUnicode_) noexcept
    : studiedCase(studiedCase_), sz(sz_), isUnicode(isUnicode_) {
  path traceFile(AppStart::dir());
  traceFile.append("data_")
      .concat(studiedCase)
      .concat(".csv");  // generating a CSV trace file

  extern const wstring BestMatch_HEADER;
  wofs = wofstream(traceFile.c_str());
  wofs << "#Row" << COMMA() << "#Col" << COMMA() << BestMatch_HEADER << endl;
}

TransformTrace::~TransformTrace() noexcept {
  wofs.close();
}

void TransformTrace::newEntry(unsigned r,
                              unsigned c,
                              const IBestMatch& best) noexcept {
  wofs << r / sz << COMMA() << c / sz << COMMA()
       << const_cast<IBestMatch&>(best).setUnicode(isUnicode) << endl;
  /*
  const_cast could be avoided by performing setUnicode(isUnicode) on a clone of
  'best'.
  However, the Unicode information doesn't affect anything apart from logging.
  */

  // flush after every row fully transformed
  if (r > transformingRow) {
    wofs.flush();
    transformingRow = r;
  }
}

#endif  // _DEBUG && !UNIT_TESTING
