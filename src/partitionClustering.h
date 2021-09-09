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

#ifndef H_PARTITION_CLUSTERING
#define H_PARTITION_CLUSTERING

#include "clusterAlg.h"

namespace pic2sym::syms::inline cluster {

/**
O(N^2) pseudo-clustering algorithm using 'partition' from OpenCV.
It isn't an actual clustering algorithm, as no distance measure is used.
Instead, the glyphs are grouped based on a boolean predicate assessing
if 2 given symbols could belong to the same category or not.
*/
class PartitionClustering : public ClusterAlg {
 public:
  /// Name of partition algorithm from varConfig.txt
  static inline const std::string Name{"Partition"};

  /**
  Performs clustering of a set of symbols.

  @param symsToGroup symbols to be grouped by similarity
  @param symsIndicesPerCluster returned vector of clusters, each cluster with
  the indices towards member symbols
  @param fontType font family, style and encoding (not the size); empty for
  various unit tests

  @return number of clusters obtained
  @throw logic_error when not called after setTinySymsProvider()

  Exceptions from above only to be reported, not handled

  @throw AbortedJob if the user aborts the operation.
  This exception needs to be handled by the caller.
  */
  unsigned formGroups(const VSymData& symsToGroup,
                      std::vector<std::vector<unsigned>>& symsIndicesPerCluster,
                      const std::string& fontType = "") override;
};

}  // namespace pic2sym::syms::inline cluster

#endif  // H_PARTITION_CLUSTERING
