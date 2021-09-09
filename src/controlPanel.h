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
#include "../test/mockUi.h"

#else  // UNIT_TESTING not defined

#ifndef H_CONTROL_PANEL
#define H_CONTROL_PANEL

#include "controlPanelActionsBase.h"

#include "appState.h"
#include "settingsBase.h"
#include "sliderConversionBase.h"

#pragma warning(push, 0)

#include <memory>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <unordered_set>

#include <gsl/gsl>

#include <opencv2/core/core.hpp>

#pragma warning(pop)

extern template class std::unordered_set<cv::String, std::hash<std::string> >;

namespace pic2sym::ui {

/// Interface of ControlPanel
class IControlPanel /*abstract*/ {
 public:
  /// Restores a slider to its previous value.
  virtual void restoreSliderValue(const cv::String& trName,
                                  std::string_view errText) noexcept = 0;

  /**
  Authorizes the action of the control whose name is provided as parameter.
  @return nullptr when the action isn't allowed or the controlName is not
  recognized. Otherwise it returns a permit for the action.
  @throw invalid_argument for an unknown controlName
  */
  virtual std::unique_ptr<const ActionPermit> actionDemand(
      const cv::String& controlName) noexcept = 0;

  /// Puts also the slider on 0
  virtual void updateEncodingsCount(unsigned uniqueEncodings) noexcept = 0;

  /**
  The max of the encodings slider won't update unless
  issuing an additional slider move, which has to be ignored.

  @return ControlPanel::updatingEncMax
  */
  virtual bool encMaxHack() const noexcept = 0;

  /// updates font size & encoding sliders, if necessary
  virtual void updateSymSettings(unsigned encIdx,
                                 unsigned fontSz_) noexcept = 0;

  /// Updates sliders concerning IfImgSettings items
  virtual void updateImgSettings(const cfg::IfImgSettings& is) noexcept = 0;

  /// Updates sliders concerning IMatchSettings items
  virtual void updateMatchSettings(const cfg::IMatchSettings& ms) noexcept = 0;

  virtual ~IControlPanel() noexcept = 0 {}
};

/**
Configures the transformation settings, chooses which image to process and with
which cmap.

The range of the slider used for selecting the encoding is updated automatically
based on the selected font family.

The sliders from OpenCV are quite basic, so ranges must be integer, start from 0
and include 1. Thus, range 0..1 could also mean there's only one value (0) and
the 1 will be ignored. Furthermore, range 0..x can be invalid if the actual
range is [x+1 .. y]. In this latter case an error message will report the
problem and the user needs to correct it.
*/
class ControlPanel : public IControlPanel {
 public:
  ControlPanel(IControlPanelActions& performer_,
               const cfg::ISettings& cfg_) noexcept;
  ~ControlPanel() noexcept override = default;

  // 'cfg' is supposed to not change for the original / copy
  ControlPanel(const ControlPanel&) noexcept = default;
  ControlPanel(ControlPanel&&) noexcept = default;
  void operator=(const ControlPanel&) = delete;
  void operator=(ControlPanel&&) = delete;

  /**
  Restores a slider to its previous value when:
  - the newly set value was invalid
  - the update of the tracker happened while the application was not in the
  appropriate state for it

  Only when entering the explicit new desired value of the slider there will be
  a single setTrackPos event.

  Otherwise (when dragging the handle of a slider, or when clicking on a new
  target slider position), there will be several intermediary positions that
  generate setTrackPos events.

  So, if changing the trackbar is requested at an inappropriate moment,
  all intermediary setTrackPos events should be discarded.
  This can be accomplished as follows:
  - including trName in slidersRestoringValue during the handling of the first
  setTrackPos event (ensures trName won't be allowed to change the corresponding
  value from the settings until trName is removed from slidersRestoringValue)
  - starting a thread that:
    - displays a modal window with the error text, thus blocking any further
  user maneuvers
    - after the user closes the mentioned modal window, all the setTrackPos
  events should have been consumed and now the restoration of the previous value
  can finally proceed at the termination of this thread
  */
  void restoreSliderValue(const cv::String& trName,
                          std::string_view errText) noexcept override;

  /// Authorizes the action of the control whose name is provided as parameter.
  /// When the action isn't allowed returns nullptr.
  /// Throws invalid_argument for an unknown controlName
  std::unique_ptr<const ActionPermit> actionDemand(
      const cv::String& controlName) noexcept override;

  /// Puts also the slider on 0
  void updateEncodingsCount(unsigned uniqueEncodings) noexcept override;

  /// Used for the hack above
  bool encMaxHack() const noexcept final { return updatingEncMax; }

  /// updates font size & encoding sliders, if necessary
  void updateSymSettings(unsigned encIdx, unsigned fontSz_) noexcept override;

  /// Updates sliders concerning IfImgSettings items
  void updateImgSettings(const cfg::IfImgSettings& is) noexcept override;

  /// Updates sliders concerning IMatchSettings items
  void updateMatchSettings(const cfg::IMatchSettings& ms) noexcept override;

 protected:
  /**
  Map between:
  - the addresses of the names of the sliders corresponding to the matching
  aspects
  - the slider to/from value converter for each such slider
  */
  static const std::unordered_map<
      const cv::String*,
      const std::unique_ptr<const SliderConverter> >&
  slidersConverters() noexcept;

 private:
  /**
  Starting with OpenCV 4.0.0, trackbars with a nullptr window name trigger:
    `access violation reading from address 0`.
  Therefore the Control Panel will take "" as window name instead of nullptr.
  */
  static inline const cv::String winName;

  /// The delegate responsible to perform selected actions
  gsl::not_null<IControlPanelActions*> performer;

  /// The settings, required to (re)initialize the sliders
  gsl::not_null<const cfg::ISettings*> cfg;

  /// Displays the problem and corrects it by restoring previous slider value
  /// Self joins in ~ControlPanel
  std::jthread sliderErrorsHandler{};

  /// pointers to the names of the sliders that are undergoing value restoration
  std::unordered_set<cv::String, std::hash<std::string> > slidersRestoringValue;

  /**
  When performing Load All Settings or only Load Match Aspects Settings, the
  corresponding sliders need to be updated one by one without considering the
  state and without modifying this state. In order to reduce the chance that
  some parallel update setting event might get also a free ride, the sliders are
  authorized one by one.
  */
  const cv::String* pLuckySliderName = nullptr;

  /**
  Application state manipulated via actionDemand() method
  and the ActionPermit-s generated by that method.
  Its inspection & update are guarded by a lock mechanism.
  */
  AppStateType appState{ST(Idle)};

  // Configuration sliders' positions
  int maxHSyms, maxVSyms;
  int encoding, fontSz;
  int symsBatchSz;
  int hybridResult;
  int structuralSim, correlationCorrectness;
  int underGlyphCorrectness, glyphEdgeCorrectness, asideGlyphCorrectness;
  int moreContrast, gravity, direction, largerSym;
  int thresh4Blanks;

  /**
  hack field

  The max of the encodings slider won't update unless
  issuing an additional slider move, which has to be ignored.
  */
  bool updatingEncMax{false};
};

}  // namespace pic2sym::ui

#endif  // H_CONTROL_PANEL

#endif  // UNIT_TESTING
