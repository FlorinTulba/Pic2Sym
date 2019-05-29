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

#ifndef UNIT_TESTING

#ifndef H_CMAP_INSPECT_BASE
#define H_CMAP_INSPECT_BASE

#include "updateSymsActions.h"
#include "viewsBase.h"

#pragma warning(push, 0)

#include <atomic>
#include <vector>

#include <opencv2/core/core.hpp>

#pragma warning(pop)

extern template class std::vector<cv::Mat>;

/**
Interface for displaying the symbols from the current charmap (cmap).

When there are lots of symbols, they are divided into pages which
can be browsed using the page slider.
*/
class ICmapInspect /*abstract*/ : public virtual ICvWin {
 protected:
  ICmapInspect() noexcept {}

 public:
  virtual unsigned getCellSide() const noexcept = 0;
  virtual unsigned getSymsPerRow() const noexcept = 0;
  virtual unsigned getSymsPerPage() const noexcept = 0;
  virtual unsigned getPageIdx() const noexcept = 0;
  virtual bool isBrowsable() const noexcept = 0;
  virtual void setBrowsable(bool readyToBrowse_ = true) noexcept = 0;

  /// Display an 'early' (unofficial) version of the 1st page from the Cmap
  /// view, if the official version isn't available yet
  virtual void showUnofficial1stPage(
      std::vector<cv::Mat>& symsOn1stPage,
      std::atomic_flag& updating1stCmapPage,
      LockFreeQueue& updateSymsActionsQueue) noexcept = 0;

  /// Clears the grid, the status bar and updates required fields
  virtual void clear() noexcept = 0;

  /// Puts also the slider on 0
  virtual void updatePagesCount(unsigned cmapSize) noexcept = 0;

  /// Changing font size must update also the grid
  virtual void updateGrid() noexcept = 0;

  /// Displays page 'pageIdx'
  virtual void showPage(unsigned pageIdx) noexcept = 0;
};

#endif  // H_CMAP_INSPECT_BASE

#endif  // UNIT_TESTING not defined
