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

#ifndef H_CLUSTER_SUPPORT_BASE
#define H_CLUSTER_SUPPORT_BASE

#include "clusterDataBase.h"

#pragma warning(push, 0)

#include <set>
#include <string>

#pragma warning(pop)

extern template class std::set<unsigned>;

/// Interface for ClustersSupport
class IClustersSupport /*abstract*/ {
 public:
  /**
  Clusters symsSet. For PreselectionByTinySyms == true it clusters also the tiny
  symbols.
  @param fontType allows checking for previously conducted clustering of current
  font type; empty for various unit tests
  @throw logic_error in UnitTesting due to IClusterEngine::process()

  Exception to be checked only within UnitTesting
  */
  virtual void groupSyms(const std::string& fontType = "") noexcept(!UT) = 0;

  /**
  Rearranges symsSet and its tiny correspondent version when
  PreselectionByTinySyms == true. Computes the cluster representatives and marks
  the limits between the symbols for different clusters.
  */
  virtual void delimitGroups(
      std::vector<std::vector<unsigned>>& symsIndicesPerCluster,
      VClusterData& clusters,
      std::set<unsigned>& clusterOffsets) noexcept = 0;

  /// Returns the rearranged symsSet or its tiny correspondent version when
  /// PreselectionByTinySyms == true.
  virtual const VSymData& clusteredSyms() const noexcept = 0;

  virtual ~IClustersSupport() noexcept {}

  // Slicing prevention
  IClustersSupport(const IClustersSupport&) = delete;
  IClustersSupport(IClustersSupport&&) = delete;
  IClustersSupport& operator=(const IClustersSupport&) = delete;
  IClustersSupport& operator=(IClustersSupport&&) = delete;

 protected:
  constexpr IClustersSupport() noexcept {}
};

#endif  // H_CLUSTER_SUPPORT_BASE
