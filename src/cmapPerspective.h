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

#ifdef UNIT_TESTING
#include "../test/mockCmapPerspective.h"

#else  // UNIT_TESTING not defined

#ifndef H_CMAP_PERSPECTIVE
#define H_CMAP_PERSPECTIVE

#include "cmapPerspectiveBase.h"

namespace pic2sym::ui {

/**
Ensures the symbols from the Cmap Viewer appear sorted by cluster size and then
by average pixels sum. This arrangement of the symbols is true even when the
clusters will be ignored while transforming images.
*/
class CmapPerspective : public ICmapPerspective {
 public:
  CmapPerspective() noexcept : ICmapPerspective() {}

  // Slicing prevention
  CmapPerspective(const CmapPerspective&) = delete;
  CmapPerspective(CmapPerspective&&) = delete;
  void operator=(const CmapPerspective&) = delete;
  void operator=(CmapPerspective&&) = delete;

  /**
  Rebuilds pSyms and clusterOffsets based on new values of parameters
  symsSet and symsIndicesPerCluster_.
  */
  void reset(const syms::VSymData& symsSet,
             const std::vector<std::vector<unsigned> >&
                 symsIndicesPerCluster_) noexcept override;

  /// Needed to display the cmap - returns a pair of symsSet iterators
  VPSymDataRange getSymsRange(unsigned from,
                              unsigned count) const noexcept override;

  /// Offsets of the clusters, considering pSyms
  const std::set<unsigned>& getClusterOffsets() const noexcept override;

 private:
  VPSymData pSyms;  ///< vector of pointers towards the symbols from symsSet

  /// Offsets of the clusters, considering pSyms
  std::set<unsigned> clusterOffsets;
};

}  // namespace pic2sym::ui

#endif  // H_CMAP_PERSPECTIVE

#endif  // UNIT_TESTING not defined
