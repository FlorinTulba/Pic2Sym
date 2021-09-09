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

#ifndef H_PRESENT_CMAP_BASE
#define H_PRESENT_CMAP_BASE

#include "cmapPerspectiveBase.h"

#pragma warning(push, 0)

#include <set>

#pragma warning(pop)

extern template class std::set<unsigned>;

namespace pic2sym {

/// Provides read-only access to Cmap data.
class IPresentCmap /*abstract*/ {
 public:
  /// Getting the fonts to fill currently displayed page
  virtual ui::ICmapPerspective::VPSymDataRange getFontFaces(
      unsigned from,
      unsigned maxCount) const noexcept = 0;

  /// Allows visualizing the symbol clusters within the Cmap View
  virtual const std::set<unsigned>& getClusterOffsets() const noexcept = 0;

  /**
  The viewer presents the identified clusters even when they're not used during
  the image transformation. In that case, the splits between the clusters use
  dashed line instead of a filled line.
  */
  virtual bool areClustersUsed() const noexcept = 0;

  /// Updates the Cmap View status bar with the details about the symbols
  virtual void showUnofficialSymDetails(unsigned symsCount) const noexcept = 0;

  virtual ~IPresentCmap() noexcept = 0 {}
};

}  // namespace pic2sym

#endif  // H_PRESENT_CMAP_BASE
