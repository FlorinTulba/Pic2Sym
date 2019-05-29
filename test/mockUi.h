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

#ifndef H_MOCK_UI
#define H_MOCK_UI

#ifndef UNIT_TESTING
#error Shouldn't include headers from UnitTesting project unless UNIT_TESTING is defined
#endif  // UNIT_TESTING not defined

#include "warnings.h"

#pragma warning(push, 0)

#include <memory>
#include <string>

#include <opencv2/core/core.hpp>

#pragma warning(pop)

class ICvWin /*abstract*/ {
 public:
  virtual void setTitle(const std::string&) const noexcept {}
  virtual void setOverlay(...) const noexcept {}
  virtual void setStatus(const std::string&, int = 0) const noexcept {}
  virtual void setPos(...) const noexcept {}
  virtual void permitResize(...) const noexcept {}
  virtual void resize(...) const noexcept {}

  virtual ~ICvWin() noexcept {}

  // Slicing prevention
  ICvWin(const ICvWin&) = delete;
  ICvWin(ICvWin&&) = delete;
  ICvWin& operator=(const ICvWin&) = delete;
  ICvWin& operator=(ICvWin&&) = delete;

 protected:
  constexpr ICvWin() noexcept {}
};

class CvWin /*abstract*/ : public virtual ICvWin {};

class IComparator /*abstract*/ : public virtual ICvWin {
 public:
  virtual void setReference(const cv::Mat&) noexcept {}
  virtual void setResult(...) noexcept {}
};

class ICmapInspect /*abstract*/ : public virtual ICvWin {
 public:
  virtual unsigned getCellSide() const noexcept { return 0U; }
  virtual unsigned getSymsPerRow() const noexcept { return 0U; }
  virtual unsigned getSymsPerPage() const noexcept { return 0U; }
  virtual unsigned getPageIdx() const noexcept { return 0U; }
  virtual bool isBrowsable() const noexcept { return false; }
  virtual void setBrowsable(...) noexcept {}

  /// Display an 'early' (unofficial) version of the 1st page from the Cmap
  /// view, if the official version isn't available yet
  virtual void showUnofficial1stPage(...) noexcept {}

  /// Clears the grid, the status bar and updates required fields
  virtual void clear() noexcept {}

  /// Puts also the slider on 0
  virtual void updatePagesCount(...) noexcept {}

  /// Changing font size must update also the grid
  virtual void updateGrid() noexcept {}

  virtual void showPage(...) noexcept {}  ///< displays page 'pageIdx'
};

class IPresentCmap;
class ISelectSymbols;

#pragma warning(disable : WARN_INHERITED_VIA_DOMINANCE)
class Comparator : public CvWin, public virtual IComparator {
 public:
  static void updateTransparency(...) noexcept {}
};

class CmapInspect : public CvWin, public virtual ICmapInspect {
 public:
  CmapInspect(const IPresentCmap&,
              const ISelectSymbols&,
              const unsigned&) noexcept
      : CvWin() {}

  static void updatePageIdx(...) noexcept {}
};
#pragma warning(default : WARN_INHERITED_VIA_DOMINANCE)

class ActionPermit {};
class IMatchSettings;
class IfImgSettings;

/// Interface of ControlPanel
class IControlPanel /*abstract*/ {
 public:
  virtual void restoreSliderValue(const cv::String&,
                                  const std::string&) noexcept {}
  virtual std::unique_ptr<const ActionPermit> actionDemand(
      const cv::String&) noexcept {
    return std::make_unique<const ActionPermit>();
  }
  virtual void updateEncodingsCount(...) noexcept {}
  virtual bool encMaxHack() const noexcept { return false; }
  virtual void updateSymSettings(...) noexcept {}
  virtual void updateImgSettings(const IfImgSettings&) noexcept {}
  virtual void updateMatchSettings(const IMatchSettings&) noexcept {}

  virtual ~IControlPanel() noexcept {}

  // Slicing prevention
  IControlPanel(const IControlPanel&) = delete;
  IControlPanel(IControlPanel&&) = delete;
  IControlPanel& operator=(const IControlPanel&) = delete;
  IControlPanel& operator=(IControlPanel&&) = delete;

 protected:
  constexpr IControlPanel() noexcept {}
};

class IControlPanelActions;
class ISettings;

class ControlPanel : public IControlPanel {
 public:
  ControlPanel(IControlPanelActions&, const ISettings&) noexcept {}
};

#endif  // H_MOCK_UI
