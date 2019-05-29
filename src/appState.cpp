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

#include "precompiled.h"

#ifndef UNIT_TESTING

#include "controlPanel.h"
#include "misc.h"
#include "warnings.h"

#pragma warning(push, 0)

#include <atomic>
#include <bitset>
#include <unordered_map>
#include <unordered_set>

#pragma warning(pop)

using namespace std;
using namespace cv;

extern template class unordered_set<const String*>;
extern template class unordered_set<String, hash<string>>;
extern template class unordered_map<const String*, bool>;

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

namespace {
const unordered_set<const String*> independentActions{
    &ControlPanel_aboutLabel, &ControlPanel_instructionsLabel,
    &ControlPanel_symsBatchSzTrName};
const auto itEndIndepActions = independentActions.cend();

const unordered_set<const String*> imgSettingsSliders{&ControlPanel_outWTrName,
                                                      &ControlPanel_outHTrName};
const auto itEndImgSettSliders = imgSettingsSliders.cend();

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
const auto itEndMatchAspSliders = matchAspectsSliders.cend();

// Pairs like: pointer to control's name plus a boolean stating whether the
// control is a slider or not
const unordered_map<const String*, bool> symSettingsControls{
    {&ControlPanel_selectFontLabel, false},
    {&ControlPanel_encodingTrName, true},
    {&ControlPanel_fontSzTrName, true}};
const auto itEndSymSettCtrls = symSettingsControls.cend();

atomic_flag stateAccess = ATOMIC_FLAG_INIT;

/// Waits until nobody uses appState, then locks it and releases it after
/// inspecting/changing
class LockAppState final {
 public:
  LockAppState() noexcept {
    while (stateAccess.test_and_set())
      ;
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

  /// Releases the lock
  void release() noexcept {
    if (owned) {
      stateAccess.clear();
      owned = false;
    }
  }

 private:
  bool owned = false;  ///< true while this thread is using appState
};

/// Permit for actions that can be performed without altering the existing
/// application state
class NoTraceActionPermit : public ActionPermit {};

/// Basic permit - sets a state when the permit is acquired, and clears it when
/// the sequential action finishes
class NormalActionPermit : public ActionPermit {
 public:
#pragma warning(disable : WARN_THROWS_ALTHOUGH_NOEXCEPT)
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
      :  // First init the appState reference; Change value afterwards in body
        appState(appState_),
        statesToToggle(statesToToggle_) {
    if (!lock_.isOwned())
      THROW_WITH_CONST_MSG(__FUNCTION__ " provided a lock that is not owned!",
                           logic_error);
    if (0ULL != (AppStateType)(appState_ & statesToToggle_))
      THROW_WITH_VAR_MSG(
          __FUNCTION__ " - appState_ (" +
              bitset<sizeof(AppStateType)>(appState_).to_string() +
              ") & statesToToggle_ (" +
              bitset<sizeof(AppStateType)>(statesToToggle_).to_string() +
              ") != 0!",
          invalid_argument);

    appState = (AppStateType)(appState_ | statesToToggle_);
  }
#pragma warning(default : WARN_THROWS_ALTHOUGH_NOEXCEPT)

  /// Reverts the state to the previous one.
  ~NormalActionPermit() noexcept {
    // Mandatory, as the finalization of the action is not guarded
    LockAppState lock;
    appState = (AppStateType)(appState & ~statesToToggle);
  }

  NormalActionPermit(const NormalActionPermit&) = delete;
  NormalActionPermit(NormalActionPermit&&) = delete;
  void operator=(const NormalActionPermit&) = delete;
  void operator=(NormalActionPermit&&) = delete;

 private:
  AppStateType& appState;  ///< application status

  /// The states to set when starting the action and clear when finishing
  const AppStateType statesToToggle;
};

/// Group of parameters for update*SettingDemand() functions
struct UpdateSettingsParams {
  IControlPanel& cp;
  AppStateType& appState;
  unordered_set<String, hash<string>>& slidersRestoringValue;
  const String* const pLuckySliderName;
  const String& controlName;
};

/// Shared body for the controls updating the symbols settings
unique_ptr<const ActionPermit> updateSettingDemand(
    UpdateSettingsParams& usp,
    AppStateType bannedMask,
    const string& msgWhenBanned,
    AppStateType statesToSet,
    bool isSlider = true) noexcept {
  if (isSlider && usp.slidersRestoringValue.end() !=
                      usp.slidersRestoringValue.find(usp.controlName))
    return nullptr;  // ignore call, as it's a value restoration maneuver

  // Authorize update of the sliders while loading settings without the casual
  // checks. Only one slider must be authorized at a time, to reduce the chances
  // of free rides.
  if (usp.pLuckySliderName == &usp.controlName)
    return make_unique<const NoTraceActionPermit>();

  LockAppState lock;

  if (0U != (bannedMask & usp.appState)) {
    lock.release();

    if (isSlider)
      usp.cp.restoreSliderValue(usp.controlName, msgWhenBanned);
    else
      errMsg(msgWhenBanned);

    return nullptr;
  }
  return make_unique<const NormalActionPermit>(usp.appState, statesToSet, lock);
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

unique_ptr<const ActionPermit> ControlPanel::actionDemand(
    const String& controlName) noexcept {
  if (independentActions.find(&controlName) != itEndIndepActions)
    return make_unique<const NoTraceActionPermit>();

  UpdateSettingsParams usp{*this, appState, slidersRestoringValue,
                           pLuckySliderName, controlName};

  if (imgSettingsSliders.find(&controlName) != itEndImgSettSliders)
    return updateImgSettingDemand(usp);

  if (const auto it = symSettingsControls.find(&controlName);
      it != itEndSymSettCtrls)
    return updateSymSettingDemand(usp, it->second);

  if (matchAspectsSliders.find(&controlName) != itEndMatchAspSliders)
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
  cerr << "No handling yet for " << controlName << " in " __FUNCTION__ << endl;
  assert(false);
  return nullptr;
}

#endif  // UNIT_TESTING not defined
