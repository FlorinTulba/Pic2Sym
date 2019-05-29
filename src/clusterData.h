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

#ifndef H_CLUSTER_DATA
#define H_CLUSTER_DATA

#include "clusterDataBase.h"
#include "symData.h"

class ISymsSupport;  // forward declaration

#pragma warning(disable : WARN_INHERITED_VIA_DOMINANCE)

/**
Synthesized symbol as the representative of several symbols that were clustered
together.

The specific symbol indices that form the cluster aren't needed since
the clustering regroups the symbols by clusters, so only the index of 1st
cluster member and the cluster size appear as fields.
*/
class ClusterData : public SymData, public IClusterData {
 public:
  /**
  Constructs a cluster representative for the selected symbols before they get
  reordered.

  @param symsSet the set of all normal / tiny symbols (before clustering)
  @param idxOfFirstSym_ index of the first symbol from symsSet that belongs to
  this cluster
  @param clusterSymIndices the indices towards the symbols from symsSet which
  belong to this cluster
  @param symsSupport ensures that the generated clusters are formed from tiny
  symbols for preselection mode
  @throw invalid_argument for empty symsSet or clusterSymIndices

  Exception to be only reported, not handled
  */
  ClusterData(const VSymData& symsSet,
              unsigned idxOfFirstSym_,
              const std::vector<unsigned>& clusterSymIndices,
              ISymsSupport& symsSupport) noexcept(!UT);

  /// Index of the first symbol from symsSet that belongs to this cluster
  unsigned getIdxOfFirstSym() const noexcept final;

  /// Size of the cluster - how many symbols form the cluster
  unsigned getSz() const noexcept final;

  PRIVATE :

      /// Index of the first symbol from symsSet that belongs to this cluster
      unsigned idxOfFirstSym;

  unsigned sz;  ///< size of the cluster - how many symbols form the cluster
};

#pragma warning(default : WARN_INHERITED_VIA_DOMINANCE)

#endif  // H_CLUSTER_DATA
