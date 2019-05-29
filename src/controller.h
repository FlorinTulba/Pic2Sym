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

#ifndef H_CONTROLLER
#define H_CONTROLLER

#include "controllerBase.h"
#include "updateSymsActions.h"

#pragma warning(push, 0)

#include <atomic>

#pragma warning(pop)

// Forward declarations
class IFontEngine;
class IMatchEngine;
class ISettings;
class ISettingsRW;
class ISymSettings;
class AbsJobMonitor;
class ITransformCompletion;
class ITransformer;
class IComparator;
class ICmapInspect;
class ICmapPerspective;
class ISelectSymbols;

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
  explicit Controller(ISettingsRW& s) noexcept;
  ~Controller() noexcept;  ///< destroys the windows

  Controller(const Controller&) = delete;
  Controller(Controller&&) = delete;
  void operator=(const Controller&) = delete;
  void operator=(Controller&&) = delete;

  /// Waits for the user to press ESC and confirm he wants to leave
  static void handleRequests() noexcept;

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
  const IUpdateSymSettings& getUpdateSymSettings() const noexcept override;

  /// The ref to unique_ptr solves a circular dependency inside the constructor
  const std::unique_ptr<const IPresentCmap>& getPresentCmap() const
      noexcept override;

  // Last group of methods used by many different clients without an obvious
  // pattern
  const IGlyphsProgressTracker& getGlyphsProgressTracker() const
      noexcept override;

  IPicTransformProgressTracker& getPicTransformProgressTracker() const
      noexcept override;

  IControlPanelActions& getControlPanelActions() const noexcept override;

  /// Font size determines grid size
  const unsigned& getFontSize() const noexcept override;

  /// Returns true if transforming a new image or the last one, but under other
  /// image parameters
  bool updateResizedImg(const IResizedImg& resizedImg_) noexcept override;

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
  void reportDuration(const std::string& text, double durationS) const
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
  void display1stPageIfFull(const VPixMapSym& syms) const
      noexcept(!UT) override;
#endif  // UNIT_TESTING not defined

  PROTECTED :  // Providing get<field> as public for Unit Testing

               // Methods for initialization

               static IComparator&
               getComparator() noexcept;

  IFontEngine& getFontEngine(const ISymSettings& ss_) noexcept;

  IMatchEngine& getMatchEngine(const ISettings& cfg_) noexcept;

  ITransformer& getTransformer(const ISettings& cfg_) noexcept;

  /// Status bar with font information
  const std::string textForCmapStatusBar(unsigned upperSymsCount = 0U) const
      noexcept;

  /// Progress
  const std::string textHourGlass(const std::string& prefix,
                                  double progress) const noexcept;

 private:
  /// Responsible of updating symbol settings
  const std::unique_ptr<const IUpdateSymSettings> updateSymSettings;

  /// Responsible for keeping track of the symbols loading process
  const std::unique_ptr<const IGlyphsProgressTracker> glyphsProgressTracker;

  /// Responsible for monitoring the progress during an image transformation
  const std::unique_ptr<IPicTransformProgressTracker>
      picTransformProgressTracker;

  // Control of displayed progress

  /// In charge of displaying the progress while updating the glyphs
  const std::unique_ptr<AbsJobMonitor> glyphsUpdateMonitor;

  /// In charge of displaying the progress while transforming images
  const std::unique_ptr<AbsJobMonitor> imgTransformMonitor;

  /// Reorganized symbols to be visualized within the cmap viewer
  const std::unique_ptr<ICmapPerspective> cmP;

  /**
  Provides read-only access to Cmap data.
  Needed by 'fe' from below.
  Uses 'me' from below and 'cmP' from above
  */
  const std::unique_ptr<const IPresentCmap> presentCmap;

  // Data

  /// Pointer to the resized version of most recent image that had to be
  /// transformed
  std::unique_ptr<const IResizedImg> resizedImg;
  IFontEngine& fe;          ///< font engine
  ISettingsRW& cfg;         ///< the settings for the transformations
  IMatchEngine& me;         ///< matching engine
  ITransformCompletion& t;  ///< results of the transformation

  // Views

  IComparator& comp;                   ///< view for comparing original & result
  std::unique_ptr<ICmapInspect> pCmi;  ///< view for inspecting the used cmap

  /// Allows saving a selection of symbols pointed within the charmap viewer
  const std::unique_ptr<const ISelectSymbols> selectSymbols;

  /// Responsible for the actions triggered by the controls from Control Panel
  const std::unique_ptr<IControlPanelActions> controlPanelActions;

  // synchronization items necessary while updating symbols

  /// Stores the events occurred while updating the symbols.
  mutable LockFreeQueue updateSymsActionsQueue;

  std::atomic_flag updatingSymbols;  ///< stays true while updating the symbols

  /// Controls concurrent attempts to update 1st page
  mutable std::atomic_flag updating1stCmapPage;
};

#endif  // H_CONTROLLER
