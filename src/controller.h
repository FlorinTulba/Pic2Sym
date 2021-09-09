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

#ifndef H_CONTROLLER
#define H_CONTROLLER

#include "controllerBase.h"

#include "cmapInspectBase.h"
#include "cmapPerspectiveBase.h"
#include "comparatorBase.h"
#include "fontEngineBase.h"
#include "jobMonitorBase.h"
#include "matchEngineBase.h"
#include "selectSymbolsBase.h"
#include "settingsBase.h"
#include "transformBase.h"
#include "transformCompletion.h"
#include "updateSymsActions.h"
#include "views.h"

#pragma warning(push, 0)

#include <atomic>
#include <future>

#pragma warning(pop)

namespace pic2sym {

/**
Base interface for the Controller.

There appear now 3 segregated groups of classes (as reported by AI Reviewer).

The realization of most of these methods involves many overlapping fields
and in the future some of the methods might:
- get split / merged
- be used by unexpected / different clients

Furthermore, 2 of the observed groups of methods contain only 2 methods each,
which is negligible.


To still address the segregation, here would be some options:

A.	For a fast implementation, good adaptability to unforeseen factors
  and with a small cost of creating a `Concentrator class`:

    IController could inherit 2 new parallel interfaces,
    say IControllerSupportForCPA and IControllerSupportForFE
    of the 2 smallest groups of methods (for ControlPanelActions and for
FontEngine).

  These interfaces can be split / merged / extended or changed rather easily.

B.	A more complex implementation and rather inflexible to unforeseen
factors but without the `Concentrator class` issue:

    Separating IController into inheritance layers:

    - IControllerCore - to contain the 3rd (largest) group of methods.
      Its implementation would encapsulate as protected most fields (which are
strictly required) from the existing Controller class

    - IControllerLayer1 - to contain the methods from say group 1.
      Its implementation would be on top of the core class and might not need
any new fields

    - IControllerLayer2 - to contain the methods from the remaining group.
      Its implementation would be on top of the previous layer and
      will encapsulate any remaining required field

  This `vertical` approach might later involve costly operations:
  - moving fields around between layers, plus splitting methods between layers
  - switching layers

C.	Aggregation of 2 providers for the interfaces for the 2 smallest groups
of methods. The interface provider of group 2 needs only 2 fields from
Controller. The interface provider of group 1 needs either to be a friend of
Controller or to receive all the required fields in the constructor.

  This approach is nothing more than `Feature Envy` and brings high maintenance
costs.
*/
class Controller : public IController {
 public:
  /// Initializes controller with ISettingsRW object s
  explicit Controller(cfg::ISettingsRW& s) noexcept;

  /// Waits until the user wants to leave and then destroys the windows
  ~Controller() noexcept override;

  // Slicing prevention
  Controller(const Controller&) = delete;
  Controller(Controller&&) = delete;
  void operator=(const Controller&) = delete;
  void operator=(Controller&&) = delete;

  // Group 1: 2 methods called so far only by ControlPanelActions
  void ensureExistenceCmapInspect() noexcept override;

  /**
  Triggered by new font family / encoding / size.

  @throw logic_error if called before ensureExistenceCmapInspect() - cannot be
  handled
  @throw NormalSymsLoadingFailure when unable to load normal size symbols
  @throw TinySymsLoadingFailure when unable to load tiny size symbols

  Last 2 exceptions have handlers, so this method should not be noexcept.
  */
  void symbolsChanged() override;

  // Group 2: 2 methods called so far only by FontEngine

  /// Access to methods for changing the font file or its enconding
  const IUpdateSymSettings& getUpdateSymSettings() const noexcept override;

  /// Access to methods controlling the displayed Cmap
  const IPresentCmap& getPresentCmap() const noexcept override;

  // Last group of methods used by many different clients without an obvious
  // pattern
  const IGlyphsProgressTracker& getGlyphsProgressTracker()
      const noexcept override;

  IPicTransformProgressTracker& getPicTransformProgressTracker()
      const noexcept override;

  IControlPanelActions& getControlPanelActions() const noexcept override;

  /// Font size determines grid size
  const unsigned& getFontSize() const noexcept override;

  /// Returns true if transforming a new image or the last one, but under other
  /// image parameters
  bool updateResizedImg(
      const input::IResizedImg& resizedImg_) noexcept override;

  /**
  Shows a 'Please wait' window and reports progress.

  @param progress the progress (0..1) as %
  @param title details about the ongoing operation
  @param async allows showing the window asynchronously

  @throw invalid_argument if progress outside 0..1

  Exception to be only reported, not handled
  */
  void hourGlass(double progress,
                 const std::string& title = "",
                 bool async = false) const noexcept(!UT) override;

  /**
  Updates the status bar from the charmap inspector window.

  @param upperSymsCount an overestimated number of symbols from the unfiltered
  set or 0 when considering the exact number of symbols from the filtered set
  @param suffix an optional status bar message suffix
  @param async allows showing the new status bar message asynchronously

  @throw logic_error if called before ensureExistenceCmapInspect()

  Exception to be only reported, not handled
  */
  void updateStatusBarCmapInspect(unsigned upperSymsCount = 0U,
                                  const std::string& suffix = "",
                                  bool async = false) const
      noexcept(!UT) override;

  /**
  Reports the duration of loading symbols / transforming images
  @throw logic_error if called before ensureExistenceCmapInspect()

  Exception to be only reported, not handled
  */
  void reportDuration(std::string_view text, double durationS) const
      noexcept(!UT) override;

  /// Displays the resulted image
  void showResultedImage(double completionDurationS) const noexcept override;

#ifndef UNIT_TESTING
  /**
  Attempts to display 1st cmap page, when full. Called after appending each
  symbol from charmap.
  @throw logic_error if called before ensureExistenceCmapInspect()

  Exception to be only reported, not handled
  */
  void display1stPageIfFull(const syms::VPixMapSym& syms) const
      noexcept(!UT) override;
#endif  // UNIT_TESTING not defined

  PROTECTED :  // Providing get<field> as public for Unit Testing

               // Methods for initialization

               static ui::IComparator&
               getComparator() noexcept;

  syms::IFontEngine& getFontEngine(const cfg::ISymSettings& ss_) noexcept;

  match::IMatchEngine& getMatchEngine(const cfg::ISettings& cfg_) noexcept;

  transform::ITransformer& getTransformer(const cfg::ISettings& cfg_) noexcept;

  /// Status bar with font information
  std::string textForCmapStatusBar(unsigned upperSymsCount = 0U) const noexcept;

  /// Progress
  std::string textHourGlass(const std::string& prefix,
                            double progress) const noexcept;

 private:
  /// The settings for the transformations
  gsl::not_null<cfg::ISettingsRW*> cfg;

  /// Responsible of updating symbol settings
  std::unique_ptr<const IUpdateSymSettings> updateSymSettings;

  /// Responsible for keeping track of the symbols loading process
  std::unique_ptr<const IGlyphsProgressTracker> glyphsProgressTracker;

  /// Responsible for monitoring the progress during an image transformation
  std::unique_ptr<IPicTransformProgressTracker> picTransformProgressTracker;

  // Control of displayed progress

  /// In charge of displaying the progress while updating the glyphs
  std::unique_ptr<ui::AbsJobMonitor> glyphsUpdateMonitor;

  /// In charge of displaying the progress while transforming images
  std::unique_ptr<ui::AbsJobMonitor> imgTransformMonitor;

  /// Reorganized symbols to be visualized within the cmap viewer
  std::unique_ptr<ui::ICmapPerspective> cmP;

  /**
  Provides read-only access to Cmap data.
  Needed by 'fe' from below.
  Uses 'cmP' and 'cfg' from above and lazyly 'me' from below
  */
  std::unique_ptr<const IPresentCmap> presentCmap;

  // Data

  /// Pointer to the resized version of most recent image that had to be
  /// transformed
  std::unique_ptr<const input::IResizedImg> resizedImg;
  gsl::not_null<syms::IFontEngine*> fe;    ///< font engine
  gsl::not_null<match::IMatchEngine*> me;  ///< matching engine

  /// Results of the transformation
  gsl::not_null<transform::ITransformCompletion*> t;

  // Views

  /// View for comparing original & result
  gsl::not_null<ui::IComparator*> comp;

  /// View for inspecting the used cmap
  std::unique_ptr<ui::ICmapInspect> pCmi;

  /// Allows saving a selection of symbols pointed within the charmap viewer
  std::unique_ptr<const ISelectSymbols> selectSymbols;

  /// Responsible for the actions triggered by the controls from Control Panel
  std::unique_ptr<IControlPanelActions> controlPanelActions;

  // synchronization items necessary while updating symbols

  /// Stores the events occurred while updating the symbols.
  mutable ui::LockFreeQueue updateSymsActionsQueue;

  /// Thread used to update the symbols; Self joins in ~Controller
  mutable std::jthread updatesSymbols{};

  /// Orders the display of the 1st unofficial Cmap page
  /// Waited for by 'updatesSymbols'
  mutable std::future<void> ordersDisplayOfUnofficial1stCmapPage{};

  /// Stays true while updating the symbols; Initially false
  std::atomic_flag updatingSymbols{};

  /// Controls concurrent attempts to update 1st page
  mutable std::atomic_flag updating1stCmapPage{};

  /// Set when the user wants to leave
  std::atomic_flag leaving{};
};

}  // namespace pic2sym

#endif  // H_CONTROLLER
