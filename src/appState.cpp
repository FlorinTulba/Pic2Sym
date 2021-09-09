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

#include "precompiled.h"
// This keeps precompiled.h first; Otherwise header sorting might move it

#ifndef UNIT_TESTING

#include "appState.h"

#include "controlPanel.h"
#include "misc.h"
#include "warnings.h"

#pragma warning(push, 0)

#include <atomic>
#include <bitset>
#include <unordered_map>
#include <unordered_set>

#include <gsl/gsl>

#pragma warning(pop)

using namespace std;
using namespace gsl;
using namespace cv;

extern template class unordered_set<const String*>;
extern template class unordered_set<String, hash<string>>;
extern template class unordered_map<const String*, bool>;

namespace pic2sym {

extern const String ControlPanel_aboutLabel;
extern const String ControlPanel_instructionsLabel;
extern const String ControlPanel_symsBatchSzTrName;
extern const String ControlPanel_selectImgLabel;
extern const String ControlPanel_transformImgLabel;
extern const String ControlPanel_restoreDefaultsLabel;
extern const String ControlPanel_saveAsDefaultsLabel;
extern const String ControlPanel_loadSettingsLabel;
extern const String ControlPanel_saveSettingsLabel;
extern const String ControlPanel_selectFontLabel;
extern const String ControlPanel_encodingTrName;
extern const String ControlPanel_fontSzTrName;
extern const String ControlPanel_hybridResultTrName;
extern const String ControlPanel_structuralSimTrName;
extern const String ControlPanel_correlationTrName;
extern const String ControlPanel_underGlyphCorrectnessTrName;
extern const String ControlPanel_glyphEdgeCorrectnessTrName;
extern const String ControlPanel_asideGlyphCorrectnessTrName;
extern const String ControlPanel_moreContrastTrName;
extern const String ControlPanel_gravityTrName;
extern const String ControlPanel_directionTrName;
extern const String ControlPanel_largerSymTrName;
extern const String ControlPanel_thresh4BlanksTrName;
extern const String ControlPanel_outWTrName;
extern const String ControlPanel_outHTrName;

namespace ui {

namespace {
const unordered_set<const String*> independentActions{
    &ControlPanel_aboutLabel, &ControlPanel_instructionsLabel,
    &ControlPanel_symsBatchSzTrName};

const unordered_set<const String*> imgSettingsSliders{&ControlPanel_outWTrName,
                                                      &ControlPanel_outHTrName};

const unordered_set<const String*> matchAspectsSliders{
    &ControlPanel_hybridResultTrName,
    &ControlPanel_structuralSimTrName,
    &ControlPanel_correlationTrName,
    &ControlPanel_underGlyphCorrectnessTrName,
    &ControlPanel_glyphEdgeCorrectnessTrName,
    &ControlPanel_asideGlyphCorrectnessTrName,
    &ControlPanel_moreContrastTrName,
    &ControlPanel_gravityTrName,
    &ControlPanel_directionTrName,
    &ControlPanel_largerSymTrName,
    &ControlPanel_thresh4BlanksTrName};

// Pairs like: pointer to control's name plus a boolean stating whether the
// control is a slider or not
const unordered_map<const String*, bool> symSettingsControls{
    {&ControlPanel_selectFontLabel, false},
    {&ControlPanel_encodingTrName, true},
    {&ControlPanel_fontSzTrName, true}};
const auto itEndSymSettCtrls = symSettingsControls.cend();

atomic_flag stateAccess{};

/// Waits until nobody uses appState, then locks it and releases it after
/// inspecting/changing
class LockAppState final {
 public:
  LockAppState() noexcept {
    stateAccess.wait(true);

    // Loops until it is cleared externally
    while (stateAccess.test_and_set())
      stateAccess.wait(true);

    // stateAccess was set by this method, so the lock is owned
    owned = true;
  }

  ~LockAppState() noexcept { release(); }

  // No intention to copy / move
  LockAppState(const LockAppState&) = delete;
  LockAppState(LockAppState&&) = delete;
  void operator=(const LockAppState&) = delete;
  void operator=(LockAppState&&) = delete;

  /// Query the lock state
  bool isOwned() const noexcept { return owned; }

  /// Releases the lock only if owned
  void release() noexcept {
    if (owned) {
      stateAccess.clear();
      stateAccess.notify_one();
      owned = false;
    }
  }

 private:
  bool owned{false};  ///< true while this thread is using appState
};

/// Permit for actions that can be performed without altering the existing
/// application state
class NoTraceActionPermit : public ActionPermit {};

/// Basic permit - sets a state when the permit is acquired, and clears it when
/// the sequential action finishes
class NormalActionPermit : public ActionPermit {
 public:
  /**
  Sets the new application state as the bit-or between appState_ and
  statesToToggle_.

  @throw invalid_argument if the provided appState_ didn't need to toggle some
  of the mentioned states

  @throw logic_error if the provided lock_ isn't owned

  Exceptions to be only reported, not handled
  */
  NormalActionPermit(
      AppStateType&
          appState_,  ///< existing state before the start of this action
      AppStateType statesToToggle_,  ///< the change to inflict on the state
      const LockAppState& lock_      ///< the lock guarding the permit
      ) noexcept
      : ActionPermit(),
        // First init the appState reference; Change value afterwards in body
        appState(&appState_),
        statesToToggle(statesToToggle_) {
    Expects(lock_.isOwned());
    Expects(!(AppStateType)(appState_ & statesToToggle_));

    *appState = (AppStateType)(appState_ | statesToToggle_);
  }

  /// Reverts the state to the previous one.
  ~NormalActionPermit() noexcept override {
    // Mandatory, as the finalization of the action is not guarded
    LockAppState lock;
    *appState = (AppStateType)(*appState & ~statesToToggle);
  }

  NormalActionPermit(const NormalActionPermit&) = delete;
  NormalActionPermit(NormalActionPermit&&) = delete;
  void operator=(const NormalActionPermit&) = delete;
  void operator=(NormalActionPermit&&) = delete;

 private:
  not_null<AppStateType*> appState;  ///< application status

  /// The states to set when starting the action and clear when finishing
  AppStateType statesToToggle;
};

/// Group of parameters for update*SettingDemand() functions
struct UpdateSettingsParams {
  not_null<IControlPanel*> cp;
  not_null<AppStateType*> appState;
  not_null<unordered_set<String, hash<string>>*> slidersRestoringValue;
  const String* pLuckySliderName;
  not_null<const String*> controlName;
};

/// Shared body for the controls updating the symbols settings
unique_ptr<const ActionPermit> updateSettingDemand(
    UpdateSettingsParams& usp,
    AppStateType bannedMask,
    string_view msgWhenBanned,
    AppStateType statesToSet,
    bool isSlider = true) noexcept {
  if (isSlider && usp.slidersRestoringValue->contains(*usp.controlName))
    return nullptr;  // ignore call, as it's a value restoration maneuver

  // Authorize update of the sliders while loading settings without the casual
  // checks. Only one slider must be authorized at a time, to reduce the chances
  // of free rides.
  if (usp.pLuckySliderName == usp.controlName)
    return make_unique<const NoTraceActionPermit>();

  LockAppState lock;

  // Below using bit-and, not a mistake
  if (bannedMask & *usp.appState) {
    lock.release();

    if (isSlider)
      usp.cp->restoreSliderValue(*usp.controlName, msgWhenBanned);
    else
      errMsg(msgWhenBanned);

    return nullptr;
  }
  return make_unique<const NormalActionPermit>(*usp.appState, statesToSet,
                                               lock);
}

/// Shared body for the sliders updating the image settings
unique_ptr<const ActionPermit> updateImgSettingDemand(
    UpdateSettingsParams& usp) noexcept {
  return updateSettingDemand(
      usp,
      ST(ImgTransform) | ST(UpdateImg) | ST(LoadAllSettings) |
          ST(SaveAllSettings),
      "Please don't update the image settings "
      "while they're saved "
      "or when an image transformation is performed based on them!",
      ST(UpdateImgSettings));
}

/// Shared body for the controls updating the symbols settings
unique_ptr<const ActionPermit> updateSymSettingDemand(
    UpdateSettingsParams& usp,
    bool isSlider = true) noexcept {
  return updateSettingDemand(
      usp,
      ST(ImgTransform) | ST(UpdateSymSettings) | ST(LoadAllSettings) |
          ST(SaveAllSettings),
      "Please don't update the symbols settings "
      "before the completion of similar previous updates, "
      "nor while they're saved and "
      "neither when an image transformation is performed based on them!",
      ST(UpdateSymSettings), isSlider);
}

/// Shared body for the sliders updating the match settings
unique_ptr<const ActionPermit> updateMatchSettingDemand(
    UpdateSettingsParams& usp) noexcept {
  return updateSettingDemand(
      usp,
      ST(ImgTransform) | ST(LoadAllSettings) | ST(LoadMatchSettings) |
          ST(SaveAllSettings) | ST(SaveMatchSettings),
      "Please don't update the match aspects settings "
      "before the completion of similar previous updates, "
      "nor while they're saved and "
      "neither when an image transformation is performed based on them!",
      ST(UpdateMatchSettings));
}
}  // anonymous namespace

#pragma warning(disable : WARN_THROWS_ALTHOUGH_NOEXCEPT)
unique_ptr<const ActionPermit> ControlPanel::actionDemand(
    const String& controlName) noexcept {
  if (independentActions.contains(&controlName))
    return make_unique<const NoTraceActionPermit>();

  UpdateSettingsParams usp{.cp = this,
                           .appState = &appState,
                           .slidersRestoringValue = &slidersRestoringValue,
                           .pLuckySliderName = pLuckySliderName,
                           .controlName = &controlName};

  if (imgSettingsSliders.contains(&controlName))
    return updateImgSettingDemand(usp);

  if (const auto it = symSettingsControls.find(&controlName);
      it != itEndSymSettCtrls)
    return updateSymSettingDemand(usp, it->second);

  if (matchAspectsSliders.contains(&controlName))
    return updateMatchSettingDemand(usp);

  // The actions from above who affect appState are guarded individually
  LockAppState lock;

  if (&controlName == &ControlPanel_selectImgLabel) {
    if ((ST(ImgTransform) | ST(UpdateImg)) & appState) {
      lock.release();

      errMsg(
          "Please don't load a new image "
          "while one is being loaded / transformed!");

      return nullptr;
    }

    return make_unique<const NormalActionPermit>(appState, ST(UpdateImg), lock);
  }

  if (&controlName == &ControlPanel_transformImgLabel) {
    if ((ST(ImgTransform) | ST(UpdateImg) | ST(LoadAllSettings) |
         ST(LoadMatchSettings) | ST(UpdateSymSettings) | ST(UpdateImgSettings) |
         ST(UpdateMatchSettings)) &
        appState) {
      lock.release();

      errMsg(
          "Please don't demand a transformation "
          "while one is being performed, "
          "nor while the settings are changing and "
          "neither while a new image is loaded!");

      return nullptr;
    }
    return make_unique<const NormalActionPermit>(appState, ST(ImgTransform),
                                                 lock);
  }

  if (&controlName == &ControlPanel_restoreDefaultsLabel) {
    if ((ST(ImgTransform) | ST(UpdateMatchSettings) | ST(LoadAllSettings) |
         ST(LoadMatchSettings) | ST(SaveAllSettings) | ST(SaveMatchSettings)) &
        appState) {
      lock.release();

      errMsg(
          "Please don't update the match aspects settings "
          "before the completion of similar previous updates, "
          "nor while they're saved and "
          "neither when an image transformation is performed based on them!");

      return nullptr;
    }
    return make_unique<const NormalActionPermit>(
        appState, ST(LoadMatchSettings) | ST(UpdateMatchSettings), lock);
  }

  if (&controlName == &ControlPanel_saveAsDefaultsLabel) {
    if ((ST(UpdateMatchSettings) | ST(SaveMatchSettings) | ST(LoadAllSettings) |
         ST(LoadMatchSettings)) &
        appState) {
      lock.release();

      errMsg(
          "Please don't save the match aspects settings "
          "before the completion of the same command issued previously, "
          "nor when these settings are being updated!");

      return nullptr;
    }
    return make_unique<const NormalActionPermit>(appState,
                                                 ST(SaveMatchSettings), lock);
  }

  if (&controlName == &ControlPanel_loadSettingsLabel) {
    if ((ST(ImgTransform) | ST(UpdateImgSettings) | ST(UpdateSymSettings) |
         ST(UpdateMatchSettings) | ST(LoadAllSettings) | ST(LoadMatchSettings) |
         ST(SaveAllSettings) | ST(SaveMatchSettings)) &
        appState) {
      lock.release();

      errMsg(
          "Please don't update the settings "
          "before the completion of similar previous updates, "
          "nor while they're saved and "
          "neither when an image transformation is performed based on them!");

      return nullptr;
    }
    return make_unique<const NormalActionPermit>(
        appState,
        ST(LoadAllSettings) | ST(UpdateImgSettings) | ST(UpdateSymSettings) |
            ST(UpdateMatchSettings),
        lock);
  }

  if (&controlName == &ControlPanel_saveSettingsLabel) {
    if ((ST(UpdateImgSettings) | ST(UpdateSymSettings) |
         ST(UpdateMatchSettings) | ST(LoadAllSettings) | ST(LoadMatchSettings) |
         ST(SaveAllSettings)) &
        appState) {
      lock.release();

      errMsg(
          "Please don't save the settings "
          "before the completion of the same command issued previously, "
          "nor when these settings are being updated!");

      return nullptr;
    }
    return make_unique<const NormalActionPermit>(appState, ST(SaveAllSettings),
                                                 lock);
  }

  lock.release();

  reportAndThrow<invalid_argument>("Control named '"s + controlName +
                                   "' is not handled by "s +
                                   HERE.function_name() + "!"s);
  return nullptr;  // avoids warning about code paths not returning
}
#pragma warning(default : WARN_THROWS_ALTHOUGH_NOEXCEPT)

}  // namespace ui
}  // namespace pic2sym

#endif  // UNIT_TESTING not defined
