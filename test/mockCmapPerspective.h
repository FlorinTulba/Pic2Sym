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

#ifndef H_MOCK_CMAP_PERSPECTIVE
#define H_MOCK_CMAP_PERSPECTIVE

#ifndef UNIT_TESTING
#error Shouldn't include headers from UnitTesting project unless UNIT_TESTING is defined
#endif  // UNIT_TESTING not defined

#include "symDataBase.h"
#include "symbolsSupportBase.h"

#pragma warning(push, 0)

#include <ranges>
#include <set>

#pragma warning(pop)

extern template class std::set<unsigned>;

namespace pic2sym::ui {

class ICmapPerspective /*abstract*/ {
 public:
  // Displaying the symbols requires dividing them into pages (ranges using
  // iterators)
  using VPSymData = std::vector<syms::ISymData*>;
  using VPSymDataCIt = VPSymData::const_iterator;
  using VPSymDataRange = std::ranges::subrange<VPSymDataCIt>;

  virtual void reset(const syms::VSymData&,
                     const std::vector<std::vector<unsigned>>&) noexcept = 0;
  virtual VPSymDataRange getSymsRange(...) const noexcept = 0;
  virtual const std::set<unsigned>& getClusterOffsets() const noexcept = 0;

  virtual ~ICmapPerspective() noexcept = 0 {}
};

class CmapPerspective : public ICmapPerspective {
 public:
  CmapPerspective() noexcept : ICmapPerspective() {}

  // Slicing prevention
  CmapPerspective(const CmapPerspective&) = delete;
  CmapPerspective(CmapPerspective&&) = delete;
  void operator=(const CmapPerspective&) = delete;
  void operator=(CmapPerspective&&) = delete;

  void reset(const syms::VSymData&,
             const std::vector<std::vector<unsigned>>&) noexcept override;
  VPSymDataRange getSymsRange(...) const noexcept override;
  const std::set<unsigned>& getClusterOffsets() const noexcept override;
};

}  // namespace pic2sym::ui

#endif  // H_MOCK_CMAP_PERSPECTIVE
