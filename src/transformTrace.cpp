/******************************************************************************
 The application Pic2Sym approximates images by a
 grid of colored symbols with colored backgrounds.

 Copyrights from the libraries used by the program:
 - (c) 2003-2021 Boost (www.boost.org)
     License: doc/licenses/Boost.lic
     http://www.boost.org/LICENSE_1_0.txt
 - (c) 2015-2021 OpenCV (www.opencv.org)
     License: doc/licenses/OpenCV.lic
     http://opencv.org/license/
 - (c) 1996-2021 The FreeType Project (www.freetype.org)
     License: doc/licenses/FTL.txt
     http://git.savannah.gnu.org/cgit/freetype/freetype2.git/plain/docs/FTL.TXT
 - (c) 1997-2021 OpenMP Architecture Review Board (www.openmp.org)
   (c) Microsoft Corporation (implementation for OpenMP C/C++ v2.0 March 2002)
     See: https://msdn.microsoft.com/en-us/library/8y6825x5.aspx
 - (c) 1995-2021 zlib software (Jean-loup Gailly and Mark Adler - www.zlib.net)
     License: doc/licenses/zlib.lic
     http://www.zlib.net/zlib_license.html
 - (c) 2015-2021 Microsoft Guidelines Support Library - github.com/microsoft/GSL
     License: doc/licenses/MicrosoftGSL.lic
     https://raw.githubusercontent.com/microsoft/GSL/main/LICENSE


 (c) 2016-2021 Florin Tulba <florintulba@yahoo.com>

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
// This keeps precompiled.h first; Otherwise header sorting might move it

#if defined _DEBUG || defined UNIT_TESTING

#include "bestMatchBase.h"
#include "matchParamsBase.h"
#include "warnings.h"

#pragma warning(push, 0)

#include <gsl/gsl>

#pragma warning(pop)

using namespace std;
using namespace gsl;

namespace pic2sym {

// constinit avoids initialization-order fiasco for Comma, so no need for lazy
// initialization and the use of a function call instead of a variable.
// extern seems mandatory
extern constinit not_null<cwzstring<> const> Comma{L",\t"};

namespace match {
wostream& operator<<(wostream& wos, const IMatchParams& mp) noexcept {
  wos << mp.toWstring();
  return wos;
}
}  // namespace match

}  // namespace pic2sym

#endif  // _DEBUG || UNIT_TESTING

#if defined _DEBUG && !defined UNIT_TESTING

#include "transformTrace.h"

#include "appStart.h"

using namespace std;
using namespace std::filesystem;

namespace pic2sym {

extern const wstring BestMatch_HEADER;

namespace transform {

TransformTrace::TransformTrace(const string& studiedCase_,
                               unsigned sz_,
                               bool isUnicode_) noexcept
    : studiedCase(&studiedCase_), sz(sz_), isUnicode(isUnicode_) {
  path traceFile{AppStart::dir()};
  traceFile.append("data_")
      .concat(*studiedCase)
      .concat(".csv");  // generating a CSV trace file

  wofs = wofstream(traceFile.c_str());
  wofs << "#Row" << Comma << "#Col" << Comma << BestMatch_HEADER << endl;
}

TransformTrace::~TransformTrace() noexcept {
  wofs.close();
}

void TransformTrace::newEntry(unsigned r,
                              unsigned c,
                              const p2s::match::IBestMatch& best) noexcept {
  // the Unicode information doesn't affect anything apart from logging
  const_cast<p2s::match::IBestMatch&>(best).setUnicode(isUnicode);
  wofs << r / sz << Comma << c / sz << Comma << best << endl;

  // flush after every row fully transformed
  if (r > transformingRow) {
    wofs.flush();
    transformingRow = r;
  }
}

}  // namespace transform
}  // namespace pic2sym

#endif  // _DEBUG && !UNIT_TESTING
