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

#ifndef H_TINY_SYM_BASE
#define H_TINY_SYM_BASE

#include "symDataBase.h"

#pragma warning(push, 0)

#include <opencv2/core/core.hpp>

#pragma warning(pop)

namespace pic2sym::syms {

/// Base class for the data for tiny symbols
class ITinySym /*abstract*/ : public virtual ISymData {
 public:
  /// Ratio between reference symbols and the shrunken symbol
  static constexpr unsigned RatioRefTiny{8U};

  /*
  The matrices 'mat', 'hAvgProj', 'vAvgProj', 'backslashDiagAvgProj' and
  'slashDiagAvgProj' from below are for the grounded version, not the original.
  Each would normally contain elements in range 0..1, but all of them were
  divided by the number of elements of the corresponding matrix.
  The reason behind this division is that the matrices are norm() compared and
  when the matrices are normalized (by the above mentioned division), the norm()
  result ranges for all of them in 0..1. So, despite they have, respectively:
  n^2, n, n, 2*n-1 and 2*n-1 elements, comparing 2 of the same kind with norm()
  produces values within a UNIQUE range 0..1. Knowing that, we can define a
  SINGLE threshold for all 5 matrices that establishes when 2 matrices of the
  same kind are similar.

  The alternative was to define/derive a threshold for each individual category
  (n^2, n, 2*n-1), but this requires adapting these new thresholds to every n -
  configurable size of tiny symbols.

  So, the normalization allows setting a single threshold for comparing tiny
  symbols of any configured size:
  - MaxAvgProjErrForPartitionClustering for partition clustering
  - TTSAS_Threshold_Member for TTSAS clustering
  */

  /// Grounded version of the small symbol divided by TinySymArea
  virtual const cv::Mat& getMat() const noexcept = 0;

  /// Horizontal projection divided by TinySymSz
  virtual const cv::Mat& getHAvgProj() const noexcept = 0;

  /// Vertical projection divided by TinySymSz
  virtual const cv::Mat& getVAvgProj() const noexcept = 0;

  /// Normal diagonal projection divided by TinySymDiagsCount
  virtual const cv::Mat& getBackslashDiagAvgProj() const noexcept = 0;

  /// Inverse diagonal projection divided by TinySymDiagsCount
  virtual const cv::Mat& getSlashDiagAvgProj() const noexcept = 0;
};

}  // namespace pic2sym::syms

#endif  // H_TINY_SYM_BASE
