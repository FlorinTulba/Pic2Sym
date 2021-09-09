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

#ifndef H_MOCK_UI
#define H_MOCK_UI

#ifndef UNIT_TESTING
#error Shouldn't include headers from UnitTesting project unless UNIT_TESTING is defined
#endif  // UNIT_TESTING not defined

#include "controlPanelActionsBase.h"
#include "imgSettingsBase.h"
#include "matchSettingsBase.h"
#include "presentCmapBase.h"
#include "selectSymbolsBase.h"
#include "settingsBase.h"
#include "warnings.h"

#pragma warning(push, 0)

#include <memory>
#include <string>

#include <opencv2/core/core.hpp>

#pragma warning(pop)

namespace pic2sym::ui {

class ICvWin /*abstract*/ {
 public:
  virtual void setTitle(const std::string&) const noexcept {}
  virtual void setOverlay(const std::string&, int = 0) const noexcept {}
  virtual void setStatus(const std::string&, int = 0) const noexcept {}
  virtual void setPos(int, int) const noexcept {}
  virtual void permitResize(bool = true) const noexcept {}
  virtual void resize(int, int) const noexcept {}

  virtual ~ICvWin() noexcept = 0 {}
};

class CvWin /*abstract*/ : public virtual ICvWin {};

class IComparator /*abstract*/ : public virtual ICvWin {
 public:
  virtual void setReference(const cv::Mat&) noexcept {}
  virtual void setResult(const cv::Mat&, int = 0) noexcept {}

  using ICvWin::resize;  // to remain visible after declaring the overload below
  virtual void resize() const noexcept {}
};

class ICmapInspect /*abstract*/ : public virtual ICvWin {
 public:
  virtual unsigned getCellSide() const noexcept { return 0U; }
  virtual unsigned getSymsPerRow() const noexcept { return 0U; }
  virtual unsigned getSymsPerPage() const noexcept { return 0U; }
  virtual unsigned getPageIdx() const noexcept { return 0U; }
  virtual bool isBrowsable() const noexcept { return false; }
  virtual void setBrowsable(bool = true) noexcept {}

  /// Display an 'early' (unofficial) version of the 1st page from the Cmap
  /// view, if the official version isn't available yet
  virtual void showUnofficial1stPage(...) noexcept {}

  /// Clears the grid, the status bar and updates required fields
  virtual void clear() noexcept {}

  /// Puts also the slider on 0
  virtual void updatePagesCount(unsigned) noexcept {}

  /// Changing font size must update also the grid
  virtual void updateGrid() noexcept {}

  virtual void showPage(unsigned) noexcept {}  ///< displays page 'pageIdx'
};

#pragma warning(disable : WARN_INHERITED_VIA_DOMINANCE)
class Comparator : public CvWin, public virtual IComparator {
 public:
  static void updateTransparency(auto&&...) noexcept {}
};

class CmapInspect : public CvWin, public virtual ICmapInspect {
 public:
  CmapInspect(const IPresentCmap&,
              const ISelectSymbols&,
              const unsigned&) noexcept
      : CvWin() {}

  static void updatePageIdx(auto&&...) noexcept {}
};
#pragma warning(default : WARN_INHERITED_VIA_DOMINANCE)

class ActionPermit {};

/// Interface of ControlPanel
class IControlPanel /*abstract*/ {
 public:
  virtual void restoreSliderValue(const cv::String&,
                                  std::string_view) noexcept {}
  virtual std::unique_ptr<const ActionPermit> actionDemand(
      const cv::String&) noexcept {
    return std::make_unique<const ActionPermit>();
  }
  virtual void updateEncodingsCount(unsigned) noexcept {}
  virtual bool encMaxHack() const noexcept { return false; }
  virtual void updateSymSettings(unsigned, unsigned) noexcept {}
  virtual void updateImgSettings(const cfg::IfImgSettings&) noexcept {}
  virtual void updateMatchSettings(const cfg::IMatchSettings&) noexcept {}

  virtual ~IControlPanel() noexcept = 0 {}
};

class ControlPanel : public IControlPanel {
 public:
  ControlPanel(IControlPanelActions&, const cfg::ISettings&) noexcept {}
};

}  // namespace pic2sym::ui

#endif  // H_MOCK_UI
