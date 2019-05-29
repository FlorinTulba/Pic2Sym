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

#include "fontErrorsHelper.h"

#include "warnings.h"

#pragma warning(push, 0)

#include <ft2build.h>
#include FT_TYPES_H
#include FT_ERRORS_H

#pragma warning(pop)

#undef __FTERRORS_H__
#define FT_ERRORDEF(e, v, s) {s, e},
#define FT_ERROR_START_LIST {
#define FT_ERROR_END_LIST \
  { 0, NULL }             \
  }                       \
  ;

namespace {

/// Pair describing a FreeType error - the code and the string message
struct FtError {
  const char* msg;  ///< error message
  int code;         ///< error code
};

/// Initializes the vector of FreeType error strings
static const std::vector<std::string>& initFtErrors() noexcept {
  static std::vector<std::string> _FtErrors;
  static bool initilized = false;

  if (!initilized) {
    const FtError ft_errors[] =
#pragma warning(push, 0)
#include FT_ERRORS_H
#pragma warning(pop)

        int maxErrCode = INT_MIN;
    for (const FtError& err : ft_errors) {
      if (err.code > maxErrCode)
        maxErrCode = err.code;
    }

    _FtErrors.resize(size_t(maxErrCode) + 1ULL);

    for (const FtError& err : ft_errors) {
      if (err.msg != nullptr)
        _FtErrors[(size_t)err.code] = err.msg;
    }

    for (int i = 0; i <= maxErrCode; ++i) {
      if (_FtErrors[(size_t)i].empty())
        _FtErrors[(size_t)i] = std::to_string(i);
    }

    initilized = true;
  }

  return _FtErrors;
}
}  // anonymous namespace

const std::vector<std::string>& FtErrors = initFtErrors();
