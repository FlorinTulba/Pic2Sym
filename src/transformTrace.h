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

#if defined UNIT_TESTING || !defined _DEBUG

#include "warnings.h"

#pragma warning(disable : WARN_INCLUDE_UNSAFE_PATH)
#include "../test/mockTransformTrace.h"
#pragma warning(default : WARN_INCLUDE_UNSAFE_PATH)

#else  // Debug mode and UNIT_TESTING not defined

#ifndef H_TRANSFORM_TRACE
#define H_TRANSFORM_TRACE

#include "bestMatchBase.h"

#pragma warning(push, 0)

#include <fstream>
#include <string>

#pragma warning(pop)

namespace pic2sym::transform {

/// Facilitates the tracing process during the transformation of an image.
class TransformTrace {
 public:
  /// Opens a trace file stream and initializes required fields
  TransformTrace(const std::string& studiedCase_,
                 unsigned sz_,
                 bool isUnicode_) noexcept;
  virtual ~TransformTrace() noexcept;  ///< closes the trace stream

  // 'wofs' usage is safer if no copy / move ops
  TransformTrace(const TransformTrace&) = delete;
  TransformTrace(TransformTrace&&) = delete;
  void operator=(const TransformTrace&) = delete;
  void operator=(TransformTrace&&) = delete;

  /// adds a new line to the trace file containing row, column and details about
  /// the best match for a new patch
  void newEntry(unsigned r, unsigned c, const match::IBestMatch& best) noexcept;

 private:
  /// Used to establish the name of the generated trace file
  gsl::not_null<const std::string*> studiedCase;

  std::wofstream wofs;  ///< trace file stream

  unsigned sz;  ///< symbol size

  /// The index of the current row being transformed
  unsigned transformingRow{};

  /// Unicode symbols are logged in symbol format, while other encodings log
  /// just their code
  bool isUnicode;
};

}  // namespace pic2sym::transform

#endif  // !H_TRANSFORM_TRACE

#endif  // _DEBUG , UNIT_TESTING
