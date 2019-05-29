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

#ifndef H_CLUSTER_PROCESSING_BASE
#define H_CLUSTER_PROCESSING_BASE

#include "symDataBase.h"

#pragma warning(push, 0)

#include <string>

#pragma warning(pop)

/// The cluster processing interface of the ClusterEngine used within
/// ClusterSupport
class IClusterProcessing /*abstract*/ {
 public:
  /**
  Clusters symsSet & tinySymsSet into clusters, while clusterOffsets reports
  where each cluster starts. When using the tiny symbols preselection, the
  clusters will contain tiny symbols.
  @param symsSet original symbols to be clustered
  @param fontType allows checking for previously conducted clustering of current
  font type; empty for various unit tests
  @throw logic_error if called before useSymsMonitor()

  Exception to be only reported, not handled
  */
  virtual void process(VSymData& symsSet,
                       const std::string& fontType = "") noexcept(!UT) = 0;

  virtual unsigned getClustersCount() const noexcept = 0;

  virtual ~IClusterProcessing() noexcept {}

  // Slicing prevention
  IClusterProcessing(const IClusterProcessing&) = delete;
  IClusterProcessing(IClusterProcessing&&) = delete;
  IClusterProcessing& operator=(const IClusterProcessing&) = delete;
  IClusterProcessing& operator=(IClusterProcessing&&) = delete;

 protected:
  constexpr IClusterProcessing() noexcept {}
};

#endif  // H_CLUSTER_PROCESSING_BASE
