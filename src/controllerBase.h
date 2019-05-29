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

#ifndef H_CONTROLLER_BASE
#define H_CONTROLLER_BASE

#include "misc.h"

#ifndef UNIT_TESTING
#include "pixMapSymBase.h"

#else  // UNIT_TESTING defined
#pragma warning(push, 0)

#include <memory>

#pragma warning(pop)

#endif  // UNIT_TESTING

#pragma warning(push, 0)

#include <string>

#pragma warning(pop)

// Forward declarations
class IUpdateSymSettings;
class IGlyphsProgressTracker;
class IPicTransformProgressTracker;
class IPresentCmap;
class IControlPanelActions;
class IResizedImg;

/// Base interface for the Controller.
class IController /*abstract*/ {
 public:
  virtual ~IController() noexcept {}

  // Slicing prevention
  IController(const IController&) = delete;
  IController(IController&&) = delete;
  IController& operator=(const IController&) = delete;
  IController& operator=(IController&&) = delete;

  // Group 1: 2 methods called so far only by ControlPanelActions
  /**
  Triggered by new font family / encoding / size.

  @throw logic_error if called before ensureExistenceCmapInspect() - cannot be
  handled
  @throw NormalSymsLoadingFailure when unable to load normal size symbols
  @throw TinySymsLoadingFailure when unable to load tiny size symbols

  Last 2 exceptions have handlers, so this method should not be noexcept.
  */
  virtual void symbolsChanged() = 0;

  virtual void ensureExistenceCmapInspect() noexcept = 0;

  // Group 2: 2 methods called so far only by FontEngine
  virtual const IUpdateSymSettings& getUpdateSymSettings() const noexcept = 0;

  /// The ref to unique_ptr solves a circular dependency inside the constructor
  virtual const std::unique_ptr<const IPresentCmap>& getPresentCmap() const
      noexcept = 0;

  // Last group of methods used by many different clients without an obvious
  // pattern
  virtual const IGlyphsProgressTracker& getGlyphsProgressTracker() const
      noexcept = 0;

  virtual IPicTransformProgressTracker& getPicTransformProgressTracker() const
      noexcept = 0;

  /// Font size determines grid size
  virtual const unsigned& getFontSize() const noexcept = 0;

  /// Returns true if transforming a new image or the last one, but under other
  /// image parameters
  virtual bool updateResizedImg(const IResizedImg& resizedImg_) noexcept = 0;

  /**
  Shows a 'Please wait' window and reports progress.

  @param progress the progress (0..1) as %
  @param title details about the ongoing operation
  @param async allows showing the window asynchronously

  @throw invalid_argument if progress outside 0..1

  Exception to be only reported, not handled
  */
  virtual void hourGlass(double progress,
                         const std::string& title = "",
                         bool async = false) const noexcept(!UT) = 0;

  ///< Displays the resulted image
  virtual void showResultedImage(double completionDurationS) const noexcept = 0;

  /**
  Updates the status bar from the charmap inspector window.

  @param upperSymsCount an overestimated number of symbols from the unfiltered
  set or 0 when considering the exact number of symbols from the filtered set
  @param suffix an optional status bar message suffix
  @param async allows showing the new status bar message asynchronously

  @throw logic_error if called before ensureExistenceCmapInspect()

  Exception to be only reported, not handled
  */
  virtual void updateStatusBarCmapInspect(unsigned upperSymsCount = 0U,
                                          const std::string& suffix = "",
                                          bool async = false) const
      noexcept(!UT) = 0;

  /**
  Reports the duration of loading symbols / transforming images
  @throw logic_error if called before ensureExistenceCmapInspect()

  Exception to be only reported, not handled
  */
  virtual void reportDuration(const std::string& text, double durationS) const
      noexcept(!UT) = 0;

  virtual IControlPanelActions& getControlPanelActions() const noexcept = 0;

#ifndef UNIT_TESTING
  /**
  Attempts to display 1st cmap page, when full. Called after appending each
  symbol from charmap.
  @throw logic_error if called before ensureExistenceCmapInspect()

  Exception to be only reported, not handled
  */
  virtual void display1stPageIfFull(const VPixMapSym& syms) const
      noexcept(!UT) = 0;
#endif  // UNIT_TESTING not defined

 protected:
  constexpr IController() noexcept {}
};

#endif  // H_CONTROLLER_BASE
