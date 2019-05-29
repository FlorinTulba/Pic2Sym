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

#ifndef H_SYMBOLS_SUPPORT_BASE
#define H_SYMBOLS_SUPPORT_BASE

#pragma warning(push, 0)

#include <vector>

#include <opencv2/core/core.hpp>

#pragma warning(pop)

class ISymData;  // Forward declaration

/// Interface for SymsSupport* types (blurring and computing cluster
/// representatives)
class ISymsSupport /*abstract*/ {
 public:
  /// @return the value of PreselectionByTinySyms
  virtual bool usingTinySymbols() const noexcept = 0;

  /// Generates clusters with normal / tiny format, depending on
  /// PreselectionByTinySyms
  virtual void computeClusterRepresentative(
      const std::vector<const ISymData*>& clusterSyms,
      int symSz,
      double invClusterSz,
      cv::Mat& synthesizedSym,
      cv::Mat& negSym) const noexcept = 0;

  virtual ~ISymsSupport() noexcept {}

  // If slicing is observed and becomes a severe problem, use `= delete` for all
  ISymsSupport(const ISymsSupport&) = delete;
  ISymsSupport(ISymsSupport&&) = delete;
  ISymsSupport& operator=(const ISymsSupport&) = delete;
  ISymsSupport& operator=(ISymsSupport&&) = delete;

 protected:
  constexpr ISymsSupport() noexcept {}
};

#endif  // H_SYMBOLS_SUPPORT_BASE
