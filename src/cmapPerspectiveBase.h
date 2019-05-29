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

#ifdef UNIT_TESTING
#include "../test/mockCmapPerspective.h"

#else  // UNIT_TESTING not defined

#ifndef H_CMAP_PERSPECTIVE_BASE
#define H_CMAP_PERSPECTIVE_BASE

#include "symDataBase.h"

#pragma warning(push, 0)

#include <set>

#pragma warning(pop)

extern template class std::set<unsigned>;

/**
Ensures the symbols from the Cmap Viewer appear sorted by cluster size and then
by average pixels sum. This arrangement of the symbols is true even when the
clusters will be ignored while transforming images.
*/
class ICmapPerspective /*abstract*/ {
 public:
  // Displaying the symbols requires dividing them into pages (ranges using
  // iterators)
  typedef std::vector<const ISymData*> VPSymData;
  typedef VPSymData::const_iterator VPSymDataCIt;
  typedef std::pair<VPSymDataCIt, VPSymDataCIt> VPSymDataCItPair;

  /**
  Rebuilds pSyms and clusterOffsets based on new values of parameters
  symsSet and symsIndicesPerCluster_.
  */
  virtual void reset(const VSymData& symsSet,
                     const std::vector<std::vector<unsigned> >&
                         symsIndicesPerCluster_) noexcept = 0;

  /// Needed to display the cmap - returns a pair of symsSet iterators
  virtual VPSymDataCItPair getSymsRange(unsigned from, unsigned count) const
      noexcept = 0;

  /// Offsets of the clusters, considering pSyms
  virtual const std::set<unsigned>& getClusterOffsets() const noexcept = 0;

  virtual ~ICmapPerspective() noexcept {}

  // Slicing prevention
  ICmapPerspective(const ICmapPerspective&) = delete;
  ICmapPerspective(ICmapPerspective&&) = delete;
  ICmapPerspective& operator=(const ICmapPerspective&) = delete;
  ICmapPerspective& operator=(ICmapPerspective&&) = delete;

 protected:
  constexpr ICmapPerspective() noexcept {}
};

#endif  // H_CMAP_PERSPECTIVE_BASE

#endif  // UNIT_TESTING not defined
