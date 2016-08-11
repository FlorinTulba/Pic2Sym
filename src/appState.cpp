/****************************************************************************************
 The application Pic2Sym approximates images by a
 grid of colored symbols with colored backgrounds.

 This file belongs to the Pic2Sym project.

 Copyrights from the libraries used by the program:
 - (c) 2015 Boost (www.boost.org)
   License: <http://www.boost.org/LICENSE_1_0.txt>
            or doc/licenses/Boost.lic
 - (c) 2015 The FreeType Project (www.freetype.org)
   License: <http://git.savannah.gnu.org/cgit/freetype/freetype2.git/plain/docs/FTL.TXT>
	        or doc/licenses/FTL.txt
 - (c) 2015 OpenCV (www.opencv.org)
   License: <http://opencv.org/license.html>
            or doc/licenses/OpenCV.lic
 - (c) 1997-2002 OpenMP Architecture Review Board (www.openmp.org)
   (c) Microsoft Corporation (Visual C++ implementation for OpenMP C/C++ Version 2.0 March 2002)
   See: <https://msdn.microsoft.com/en-us/library/8y6825x5(v=vs.140).aspx>
 
 (c) 2016 Florin Tulba <florintulba@yahoo.com>

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
 If not, see <http://www.gnu.org/licenses/agpl-3.0.txt>.
 ****************************************************************************************/

#include "controlPanel.h"
#include "misc.h"

#include <set>
#include <map>

using namespace std;
using namespace cv;

/// Permit for actions that can be performed without altering the existing application state
struct NoTraceActionPermit : ActionPermit {};

/// Basic permit - sets a state when the permit is acquired, and clears it when the sequential action finishes
class NormalActionPermit : public ActionPermit {
protected:
	volatile AppStateType &appState;	///< application status
	const AppStateType statesToToggle;	///< the states to set when starting the action and clear when finishing

public:
	NormalActionPermit(volatile AppStateType &appState_,	///< existing state before the start of this action
					   AppStateType statesToToggle_			///< the change to inflict on the state
					   ) : appState(appState_), statesToToggle(statesToToggle_) {
		appState = appState | statesToToggle;
	}

	~NormalActionPermit() {
		appState = appState & ~statesToToggle;
	}
};

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
	/// Shared body for the controls updating the symbols settings
	unique_ptr<ActionPermit> updateSettingDemand(ControlPanel &cp,
												 volatile AppStateType &appState,
												 set<const cv::String*> &slidersRestoringValue,
												 const cv::String * const pLuckySliderName,
												 const String &controlName,
												 AppStateType bannedMask,
												 const string &msgWhenBanned,
												 AppStateType statesToSet,
												 bool isSlider = true) {
		if(isSlider && slidersRestoringValue.end() != slidersRestoringValue.find(&controlName))
			return nullptr; // ignore call, as it's a value restoration maneuver

		// Authorize update of the sliders while loading settings without the casual checks.
		// Only one slider must be authorized at a time, to reduce the chances of free rides.
		if(pLuckySliderName == &controlName)
			return std::move(make_unique<NoTraceActionPermit>());

		if(0U != (bannedMask & appState)) {
			errMsg(msgWhenBanned);
			if(isSlider)
				cp.restoreSliderValue(controlName);
			return nullptr;
		}
		return std::move(make_unique<NormalActionPermit>(appState, statesToSet));
	}

	/// Shared body for the sliders updating the image settings
	unique_ptr<ActionPermit> updateImgSettingDemand(ControlPanel &cp,
													volatile AppStateType &appState,
													set<const cv::String*> &slidersRestoringValue,
													const cv::String * const pLuckySliderName,
													const String &controlName) {
		return std::move(
			updateSettingDemand(cp, appState, slidersRestoringValue, pLuckySliderName, controlName,
			(AppStateType)AppState::ImgTransform | (AppStateType)AppState::UpdateImg
			| (AppStateType)AppState::LoadAllSettings | (AppStateType)AppState::SaveAllSettings,
			"Please don't update the image settings while they're saved or when an image transformation is performed based on them!",
			(AppStateType)AppState::UpdateImgSettings));
	}

	/// Shared body for the controls updating the symbols settings
	unique_ptr<ActionPermit> updateSymSettingDemand(ControlPanel &cp,
													volatile AppStateType &appState,
													set<const cv::String*> &slidersRestoringValue,
													const cv::String * const pLuckySliderName,
													const String &controlName,
													bool isSlider = true) {
		return std::move(
			updateSettingDemand(cp, appState, slidersRestoringValue, pLuckySliderName, controlName,
			(AppStateType)AppState::ImgTransform  | (AppStateType)AppState::UpdateSymSettings
			| (AppStateType)AppState::LoadAllSettings | (AppStateType)AppState::SaveAllSettings,
			"Please don't update the symbols settings before the completion of similar previous updates, nor while they're saved and neither when an image transformation is performed based on them!",
			(AppStateType)AppState::UpdateSymSettings, isSlider));
	}

	/// Shared body for the sliders updating the match settings
	unique_ptr<ActionPermit> updateMatchSettingDemand(ControlPanel &cp,
													  volatile AppStateType &appState,
													  set<const cv::String*> &slidersRestoringValue,
													  const cv::String * const pLuckySliderName,
													  const String &controlName) {
		return std::move(
			updateSettingDemand(cp, appState, slidersRestoringValue, pLuckySliderName, controlName,
			(AppStateType)AppState::ImgTransform
			| (AppStateType)AppState::LoadAllSettings | (AppStateType)AppState::LoadMatchSettings
			| (AppStateType)AppState::SaveAllSettings | (AppStateType)AppState::SaveMatchSettings,
			"Please don't update the match aspects settings before the completion of similar previous updates, nor while they're saved and neither when an image transformation is performed based on them!",
			(AppStateType)AppState::UpdateMatchSettings));
	}

} // anonymous namespace

unique_ptr<ActionPermit> ControlPanel::actionDemand(const String &controlName) {
	static const set<const String*>
		independentActions {
			&ControlPanel_aboutLabel,
			&ControlPanel_instructionsLabel,
			&ControlPanel_symsBatchSzTrName
		},
		imgSettingsSliders {
			&ControlPanel_outWTrName,
			&ControlPanel_outHTrName
		},
		matchAspectsSliders {
			&ControlPanel_hybridResultTrName,
			&ControlPanel_structuralSimTrName,
			&ControlPanel_underGlyphCorrectnessTrName,
			&ControlPanel_glyphEdgeCorrectnessTrName,
			&ControlPanel_asideGlyphCorrectnessTrName,
			&ControlPanel_moreContrastTrName,
			&ControlPanel_gravityTrName,
			&ControlPanel_directionTrName,
			&ControlPanel_largerSymTrName,
			&ControlPanel_thresh4BlanksTrName
		};
	static const auto itEndIndepActions = independentActions.cend(),
					itEndImgSettSliders = imgSettingsSliders.cend(),
					itEndMatchAspSliders = matchAspectsSliders.cend();

	// Pairs like: pointer to control's name plus a boolean stating whether the control is a slider or not
	static const map<const String*, bool> symSettingsControls {
		{&ControlPanel_selectFontLabel, false},
		{&ControlPanel_encodingTrName, true},
		{&ControlPanel_fontSzTrName, true}
	};
	static const auto itEndSymSettCtrls = symSettingsControls.cend();

	if(independentActions.find(&controlName) != itEndIndepActions)
		return std::move(make_unique<NoTraceActionPermit>());
	if(imgSettingsSliders.find(&controlName) != itEndImgSettSliders)
		return std::move(updateImgSettingDemand(*this, appState, slidersRestoringValue, pLuckySliderName, controlName));
	const auto it = symSettingsControls.find(&controlName);
	if(it != itEndSymSettCtrls)
		return std::move(updateSymSettingDemand(*this, appState, slidersRestoringValue, pLuckySliderName, controlName, it->second));
	if(matchAspectsSliders.find(&controlName) != itEndMatchAspSliders)
		return std::move(updateMatchSettingDemand(*this, appState, slidersRestoringValue, pLuckySliderName, controlName));

	if(&controlName == &ControlPanel_selectImgLabel) {
		if(((AppStateType)AppState::ImgTransform 
			| (AppStateType)AppState::UpdateImg | (AppStateType)AppState::UpdateImgSettings) & appState) {
			ostringstream oss;
			oss<<"Please don't load a new image while one is being loaded / transformed!";
			errMsg(oss.str());
			return nullptr;
		}
		return std::move(make_unique<NormalActionPermit>(appState, (AppStateType)AppState::UpdateImg));

	} else if(&controlName == &ControlPanel_transformImgLabel) {
		if(((AppStateType)AppState::ImgTransform | (AppStateType)AppState::UpdateImg 
			| (AppStateType)AppState::LoadAllSettings | (AppStateType)AppState::LoadMatchSettings
			| (AppStateType)AppState::UpdateSymSettings | (AppStateType)AppState::UpdateImgSettings | (AppStateType)AppState::UpdateMatchSettings) & appState) {
			ostringstream oss;
			oss<<"Please don't demand a transformation while one is being performed, nor while the settings are changing and neither while a new image is loaded!";
			errMsg(oss.str());
			return nullptr;
		}
		return std::move(make_unique<NormalActionPermit>(appState, (AppStateType)AppState::ImgTransform));

	} else if(&controlName == &ControlPanel_restoreDefaultsLabel) {
		if(((AppStateType)AppState::ImgTransform | (AppStateType)AppState::UpdateMatchSettings
			| (AppStateType)AppState::LoadAllSettings | (AppStateType)AppState::LoadMatchSettings
			| (AppStateType)AppState::SaveAllSettings | (AppStateType)AppState::SaveMatchSettings) & appState) {
			ostringstream oss;
			oss<<"Please don't update the match aspects settings before the completion of similar previous updates, nor while they're saved and neither when an image transformation is performed based on them!";
			errMsg(oss.str());
			return nullptr;
		}
		return std::move(make_unique<NormalActionPermit>(appState, (AppStateType)AppState::LoadMatchSettings 
			| (AppStateType)AppState::UpdateMatchSettings));

	} else if(&controlName == &ControlPanel_saveAsDefaultsLabel) {
		if(((AppStateType)AppState::UpdateMatchSettings | (AppStateType)AppState::SaveMatchSettings
			| (AppStateType)AppState::LoadAllSettings | (AppStateType)AppState::LoadMatchSettings) & appState) {
			ostringstream oss;
			oss<<"Please don't save the match aspects settings before the completion of the same command issued previously, nor when these settings are being updated!";
			errMsg(oss.str());
			return nullptr;
		}
		return std::move(make_unique<NormalActionPermit>(appState, (AppStateType)AppState::SaveMatchSettings));

	} else if(&controlName == &ControlPanel_loadSettingsLabel) {
		if(((AppStateType)AppState::ImgTransform
			| (AppStateType)AppState::UpdateImgSettings | (AppStateType)AppState::UpdateSymSettings | (AppStateType)AppState::UpdateMatchSettings
			| (AppStateType)AppState::LoadAllSettings | (AppStateType)AppState::LoadMatchSettings
			| (AppStateType)AppState::SaveAllSettings | (AppStateType)AppState::SaveMatchSettings) & appState) {
			ostringstream oss;
			oss<<"Please don't update the settings before the completion of similar previous updates, nor while they're saved and neither when an image transformation is performed based on them!";
			errMsg(oss.str());
			return nullptr;
		}
		return std::move(make_unique<NormalActionPermit>(appState, (AppStateType)AppState::LoadAllSettings 
			| (AppStateType)AppState::UpdateImgSettings | (AppStateType)AppState::UpdateSymSettings | (AppStateType)AppState::UpdateMatchSettings));

	} else if(&controlName == &ControlPanel_saveSettingsLabel) {
		if(((AppStateType)AppState::UpdateImgSettings | (AppStateType)AppState::UpdateSymSettings | (AppStateType)AppState::UpdateMatchSettings
			| (AppStateType)AppState::LoadAllSettings | (AppStateType)AppState::LoadMatchSettings 
			| (AppStateType)AppState::SaveAllSettings) & appState) {
			ostringstream oss;
			oss<<"Please don't save the settings before the completion of the same command issued previously, nor when these settings are being updated!";
			errMsg(oss.str());
			return nullptr;
		}
		return std::move(make_unique<NormalActionPermit>(appState, (AppStateType)AppState::SaveAllSettings));

	} else throw domain_error("No handling yet for " + controlName + " in " __FUNCTION__);
}