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
#include "../test/mockUi.h"

#else  // UNIT_TESTING not defined

#ifndef H_VIEWS
#define H_VIEWS

#include "cmapInspectBase.h"
#include "cmapPerspectiveBase.h"
#include "comparatorBase.h"
#include "warnings.h"

/**
CvWin - base class for Comparator & CmapInspect from below.
Allows setting title, overlay, status, location, size and resizing properties.
*/
class CvWin /*abstract*/ : public virtual ICvWin {
 public:
  void setTitle(const std::string& title) const noexcept override;
  void setOverlay(const std::string& overlay, int timeoutMs = 0) const
      noexcept override;
  void setStatus(const std::string& status, int timeoutMs = 0) const
      noexcept override;

  void setPos(int x, int y) const noexcept override;
  void permitResize(bool allow = true) const noexcept override;
  void resize(int w, int h) const noexcept override;

 protected:
  explicit CvWin(const cv::String& winName_) noexcept;

  /// Window's handle
  const cv::String& winName() const noexcept { return _winName; }

  /// What to display - const version
  const cv::Mat& content() const noexcept { return _content; }

  /// What to display - reference version
  cv::Mat& content() noexcept { return _content; }

  /// What to display - setter version
  void content(const cv::Mat& content) noexcept { _content = content; }

 private:
  const cv::String _winName;  ///< window's handle
  cv::Mat _content;           ///< what to display
};

#pragma warning(disable : WARN_INHERITED_VIA_DOMINANCE)
/**
View which permits comparing the original image with the transformed one.

A slider adjusts the transparency of the resulted image,
so that the original can be more or less visible.
*/
class Comparator : public CvWin, public virtual IComparator {
 public:
  Comparator() noexcept;  ///< Creating a Comparator window.

  /// Slider's callback
  static void updateTransparency(int newTransp, void* userdata) noexcept;

  /**
  Setting the original image to be processed.
  @throw invalid_argument for an empty reference_

  Exception to be only reported, not handled
  */
  void setReference(const cv::Mat& reference_) noexcept override;

  /**
  Setting the resulted image after processing.

  @throw invalid_argument for a result_ with other dimensions than reference_
  @throw logic_error if called before setReference()

  Exceptions to be only reported, not handled
  */
  void setResult(const cv::Mat& result_,
                 int transparency = (int)
                     round(Comparator_defaultTransparency *
                           Comparator_trackMax)) noexcept override;

  using CvWin::resize;  // to remain visible after declaring the overload below
  void resize() const noexcept override;

 protected:
  /// Called from updateTransparency
  void setTransparency(double transparency) noexcept;

  /// Image to display initially (not for processing)
  static const cv::Mat noImage;

 private:
  cv::Mat initial, result;  ///< Compared items
  int trackPos = 0;         ///< Transparency value
};
#pragma warning(default : WARN_INHERITED_VIA_DOMINANCE)

// Forward declarations
class IPresentCmap;
class ISelectSymbols;

#pragma warning(disable : WARN_INHERITED_VIA_DOMINANCE)
/**
Class for displaying the symbols from the current charmap (cmap).

When there are lots of symbols, they are divided into pages which
can be browsed using the page slider.
*/
class CmapInspect : public CvWin, public virtual ICmapInspect {
 public:
  CmapInspect(const IPresentCmap& cmapPresenter_,
              const ISelectSymbols& symsSelector,
              const unsigned& fontSz_) noexcept;

  // nor move ops
  /// Slider's callback
  static void updatePageIdx(int newPage, void* userdata) noexcept;

  unsigned getCellSide() const noexcept final { return cellSide; }
  unsigned getSymsPerRow() const noexcept final { return symsPerRow; }
  unsigned getSymsPerPage() const noexcept final { return symsPerPage; }
  unsigned getPageIdx() const noexcept final { return (unsigned)page; }
  bool isBrowsable() const noexcept final { return readyToBrowse; }
  void setBrowsable(bool readyToBrowse_ = true) noexcept final {
    readyToBrowse = readyToBrowse_;
  }

  /// Display an 'early' (unofficial) version of the 1st page from the Cmap
  /// view, if the official version isn't available yet
  void showUnofficial1stPage(
      std::vector<cv::Mat>& symsOn1stPage,
      std::atomic_flag& updating1stCmapPage,
      LockFreeQueue& updateSymsActionsQueue) noexcept override;

  /// Clears the grid, the status bar and updates required fields
  void clear() noexcept override;

  /// Puts also the slider on 0
  void updatePagesCount(unsigned cmapSize) noexcept override;

  /// Changing font size must update also the grid
  void updateGrid() noexcept override;

  /// Displays page 'pageIdx'
  void showPage(unsigned pageIdx) noexcept override;

 protected:
  /// Generates the grid that separates the glyphs
  cv::Mat createGrid() noexcept;

  /// content = grid + glyphs for current page specified by a pair of iterators
  void populateGrid(const ICmapPerspective::VPSymDataCItPair& itPair,
                    const std::set<unsigned>& clusterOffsets,
                    unsigned idxOfFirstSymFromPage) noexcept;

 private:
  const IPresentCmap& cmapPresenter;  ///< presents the cmap window

  cv::Mat grid;               ///< the symbols' `hive`
  int page = 0;               ///< page slider position
  unsigned pagesCount = 0U;   ///< used for dividing the cmap
  const unsigned& fontSz;     ///< font size
  unsigned cellSide = 0U;     ///< used for dividing the cmap
  unsigned symsPerRow = 0U;   ///< used for dividing the cmap
  unsigned symsPerPage = 0U;  ///< used for dividing the cmap

  /**
  hack field

  The max of the page slider won't update unless
  issuing an additional slider move, which has to be ignored.
  */
  bool updatingPageMax = false;

  bool readyToBrowse = false;  ///< set to true only when all pages are ready
};
#pragma warning(default : WARN_INHERITED_VIA_DOMINANCE)

#endif  // H_VIEWS

#endif  // UNIT_TESTING not defined
