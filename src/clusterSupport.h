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

#ifndef H_CLUSTER_SUPPORT
#define H_CLUSTER_SUPPORT

#include "clusterSupportBase.h"

#include "clusterProcessingBase.h"
#include "symbolsSupportBase.h"

#pragma warning(push, 0)

#include <gsl/gsl>

#pragma warning(pop)

namespace pic2sym::syms::inline cluster {

/**
Polymorphic helper for clustering the symbols depending on the value of
PreselectionByTinySyms. For PreselectionByTinySyms == true, the clusters include
tiny symbols.
*/
class ClustersSupport : public IClustersSupport {
 public:
  /// Base constructor
  ClustersSupport(IClusterProcessing& ce_,
                  std::unique_ptr<ISymsSupport> ss_,
                  VSymData& symsSet_) noexcept;

  // Slicing prevention
  ClustersSupport(const ClustersSupport&) = delete;
  ClustersSupport(ClustersSupport&&) = delete;
  void operator=(const ClustersSupport&) = delete;
  void operator=(ClustersSupport&&) = delete;

  /**
  Clusters symsSet. For PreselectionByTinySyms == true it clusters also the tiny
  symbols.
  @param fontType allows checking for previously conducted clustering of current
  font type; empty for various unit tests
  @throw logic_error in UnitTesting due to IClusterEngine::process()

  Exceptions from above to be checked only within UnitTesting

  @throw AbortedJob when the user aborts the operation.
  This exception needs to be handled by the caller
  */
  void groupSyms(const std::string& fontType = "") override;

  /**
  Rearranges symsSet and its tiny correspondent version when
  PreselectionByTinySyms == true. Computes the cluster representatives and marks
  the limits between the symbols for different clusters.
  */
  void delimitGroups(std::vector<std::vector<unsigned>>& symsIndicesPerCluster,
                     VClusterData& clusters,
                     std::set<unsigned>& clusterOffsets) noexcept override;

  /// Returns the rearranged symsSet or its tiny correspondent version when
  /// PreselectionByTinySyms == true.
  const VSymData& clusteredSyms() const noexcept override;

 protected:
  gsl::not_null<IClusterProcessing*> ce;  ///< clusters manager
  std::unique_ptr<ISymsSupport> ss;       ///< helper for symbols
  gsl::not_null<VSymData*> symsSet;  ///< set of most information on each symbol
};

}  // namespace pic2sym::syms::inline cluster

#endif  // H_CLUSTER_SUPPORT
