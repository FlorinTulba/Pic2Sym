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

#ifndef H_PARTITION_CLUSTERING
#define H_PARTITION_CLUSTERING

#include "clusterAlg.h"

/**
O(N^2) pseudo-clustering algorithm using 'partition' from OpenCV.
It isn't an actual clustering algorithm, as no distance measure is used.
Instead, the glyphs are grouped based on a boolean predicate assessing
if 2 given symbols could belong to the same category or not.
*/
class PartitionClustering : public ClusterAlg {
 public:
  /**
  Performs clustering of a set of symbols.

  @param symsToGroup symbols to be grouped by similarity
  @param symsIndicesPerCluster returned vector of clusters, each cluster with
  the indices towards member symbols
  @param fontType font family, style and encoding (not the size); empty for
  various unit tests

  @return number of clusters obtained
  @throw logic_error when not called after setTinySymsProvider()

  Exception only to be reported, not handled
  */
  unsigned formGroups(const VSymData& symsToGroup,
                      std::vector<std::vector<unsigned>>& symsIndicesPerCluster,
                      const std::string& fontType = "") noexcept(!UT) override;

  /// Name of partition algorithm from varConfig.txt
  static inline const std::string Name{"Partition"};
};

#endif  // H_PARTITION_CLUSTERING
