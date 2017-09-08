/************************************************************************************************
 The application Pic2Sym approximates images by a
 grid of colored symbols with colored backgrounds.

 Copyrights from the libraries used by the program:
 - (c) 2016 Boost (www.boost.org)
		License: <http://www.boost.org/LICENSE_1_0.txt>
			or doc/licenses/Boost.lic
 - (c) 2015 OpenCV (www.opencv.org)
		License: <http://opencv.org/license.html>
            or doc/licenses/OpenCV.lic
 - (c) 2015 The FreeType Project (www.freetype.org)
		License: <http://git.savannah.gnu.org/cgit/freetype/freetype2.git/plain/docs/FTL.TXT>
	        or doc/licenses/FTL.txt
 - (c) 1997-2002 OpenMP Architecture Review Board (www.openmp.org)
   (c) Microsoft Corporation (Visual C++ implementation for OpenMP C/C++ Version 2.0 March 2002)
		See: <https://msdn.microsoft.com/en-us/library/8y6825x5(v=vs.140).aspx>
 - (c) 1995-2013 zlib software (Jean-loup Gailly and Mark Adler - see: www.zlib.net)
		License: <http://www.zlib.net/zlib_license.html>
            or doc/licenses/zlib.lic
 
 (c) 2016, 2017 Florin Tulba <florintulba@yahoo.com>

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
 ***********************************************************************************************/

#ifndef UNIT_TESTING

#include "controlPanel.h"
#include "misc.h"

#pragma warning ( push, 0 )

#include <unordered_set>
#include <unordered_map>
#include <sstream>
#include <atomic>

#pragma warning ( pop )

using namespace std;
using namespace cv;

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
	const unordered_set<const String*> independentActions {
		&ControlPanel_aboutLabel,
		&ControlPanel_instructionsLabel,
		&ControlPanel_symsBatchSzTrName
	};
	const auto itEndIndepActions = independentActions.cend();

	const unordered_set<const String*> imgSettingsSliders {
		&ControlPanel_outWTrName,
		&ControlPanel_outHTrName
	};
	const auto itEndImgSettSliders = imgSettingsSliders.cend();

	const unordered_set<const String*> matchAspectsSliders {
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
	const auto itEndMatchAspSliders = matchAspectsSliders.cend();

	// Pairs like: pointer to control's name plus a boolean stating whether the control is a slider or not
	const unordered_map<const String*, bool> symSettingsControls {
		{ &ControlPanel_selectFontLabel, false },
		{ &ControlPanel_encodingTrName, true },
		{ &ControlPanel_fontSzTrName, true }
	};
	const auto itEndSymSettCtrls = symSettingsControls.cend();
			
	atomic_flag stateAccess = ATOMIC_FLAG_INIT;

	/// Waits until nobody uses appState, then locks it and releases it after inspecting/changing
	class LockAppState {
		bool owned = false;	///< true while this thread is using appState

	public:
		LockAppState() {
			while(stateAccess.test_and_set());
			owned = true;
		}

		~LockAppState() {
			release();
		}

		/// Handy for holding the lock over appState as short as possible
		void release() {
			if(owned) {
				stateAccess.clear();
				owned = false;
			}
		}
	};

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
			// No need to guard appState change from below, as the permit generation occurs
			// already in a guarded region where:
			// appState is first consulted, the permit is issued and then the lock is released
			appState = (AppStateType)(appState | statesToToggle);
		}

		NormalActionPermit(const NormalActionPermit&) = delete;
		NormalActionPermit(NormalActionPermit&&) = delete;
		void operator=(const NormalActionPermit&) = delete;
		void operator=(NormalActionPermit&&) = delete;

		~NormalActionPermit() {
			LockAppState lock; // Mandatory, as the finalization of the action is not guarded
			appState = (AppStateType)(appState & ~statesToToggle);
		}
	};

	/// Group of parameters for update*SettingDemand() functions
	struct UpdateSettingsParams {
		IControlPanel &cp;
		volatile AppStateType &appState;
		unordered_set<const String, hash<stringType>> &slidersRestoringValue;
		const String * const pLuckySliderName;
		const String &controlName;

		UpdateSettingsParams(IControlPanel &cp_,
							 volatile AppStateType &appState_,
							 unordered_set<const String, hash<stringType>> &slidersRestoringValue_,
							 const String * const pLuckySliderName_,
							 const String &controlName_) :
			cp(cp_), appState(appState_), slidersRestoringValue(slidersRestoringValue_),
			pLuckySliderName(pLuckySliderName_), controlName(controlName_) {}
		UpdateSettingsParams(const UpdateSettingsParams&) = delete;
		UpdateSettingsParams(UpdateSettingsParams&&) = delete;
		void operator=(const UpdateSettingsParams&) = delete;
		void operator=(UpdateSettingsParams&&) = delete;
	};

	/// Shared body for the controls updating the symbols settings
	uniquePtr<const ActionPermit> updateSettingDemand(UpdateSettingsParams &usp,
													  AppStateType bannedMask,
													  const stringType &msgWhenBanned,
													  AppStateType statesToSet,
													  bool isSlider = true) {
		if(isSlider &&
				usp.slidersRestoringValue.end() != usp.slidersRestoringValue.find(usp.controlName))
			return nullptr; // ignore call, as it's a value restoration maneuver

		// Authorize update of the sliders while loading settings without the casual checks.
		// Only one slider must be authorized at a time, to reduce the chances of free rides.
		if(usp.pLuckySliderName == &usp.controlName)
			return makeUnique<const NoTraceActionPermit>();

		LockAppState lock;

		if(0U != (bannedMask & usp.appState)) {
			lock.release();

			if(isSlider)
				usp.cp.restoreSliderValue(usp.controlName, msgWhenBanned);
			else
				errMsg(msgWhenBanned);

			return nullptr;
		}
		return makeUnique<const NormalActionPermit>(usp.appState, statesToSet);
	}

	/// Shared body for the sliders updating the image settings
	uniquePtr<const ActionPermit> updateImgSettingDemand(UpdateSettingsParams &usp) {
		return std::move(
			updateSettingDemand(usp,
				ST(ImgTransform) | ST(UpdateImg) | ST(LoadAllSettings) | ST(SaveAllSettings),
				"Please don't update the image settings "
					"while they're saved "
					"or when an image transformation is performed based on them!",
				ST(UpdateImgSettings)));
	}

	/// Shared body for the controls updating the symbols settings
	uniquePtr<const ActionPermit> updateSymSettingDemand(UpdateSettingsParams &usp,
															   bool isSlider = true) {
		return std::move(
			updateSettingDemand(usp,
				ST(ImgTransform)  | ST(UpdateSymSettings) | ST(LoadAllSettings) | ST(SaveAllSettings),
				"Please don't update the symbols settings "
					"before the completion of similar previous updates, "
					"nor while they're saved and "
					"neither when an image transformation is performed based on them!",
				ST(UpdateSymSettings), isSlider));
	}

	/// Shared body for the sliders updating the match settings
	uniquePtr<const ActionPermit> updateMatchSettingDemand(UpdateSettingsParams &usp) {
		return std::move(
			updateSettingDemand(usp,
				ST(ImgTransform) | ST(LoadAllSettings) | ST(LoadMatchSettings)
					| ST(SaveAllSettings) | ST(SaveMatchSettings),
				"Please don't update the match aspects settings "
					"before the completion of similar previous updates, "
					"nor while they're saved and "
					"neither when an image transformation is performed based on them!",
				ST(UpdateMatchSettings)));
	}
} // anonymous namespace

uniquePtr<const ActionPermit> ControlPanel::actionDemand(const String &controlName) {
	if(independentActions.find(&controlName) != itEndIndepActions)
		return makeUnique<const NoTraceActionPermit>();

	UpdateSettingsParams usp(*this, appState, slidersRestoringValue, pLuckySliderName, controlName);

	if(imgSettingsSliders.find(&controlName) != itEndImgSettSliders)
		return std::move(
			updateImgSettingDemand(usp));

	const auto it = symSettingsControls.find(&controlName);
	if(it != itEndSymSettCtrls)
		return std::move(
			updateSymSettingDemand(usp, it->second));

	if(matchAspectsSliders.find(&controlName) != itEndMatchAspSliders)
		return std::move(
			updateMatchSettingDemand(usp));

	LockAppState lock; // the actions from above who affect appState are guarded individually

	if(&controlName == &ControlPanel_selectImgLabel) {
		if((ST(ImgTransform) | ST(UpdateImg)) & appState) {
			lock.release();

			ostringstream oss;
			oss<<"Please don't load a new image "
				"while one is being loaded / transformed!";
			errMsg(oss.str());

			return nullptr;
		}

		return makeUnique<const NormalActionPermit>(appState, ST(UpdateImg));
	}

	if(&controlName == &ControlPanel_transformImgLabel) {
		if((ST(ImgTransform) | ST(UpdateImg)
				| ST(LoadAllSettings) | ST(LoadMatchSettings)
				| ST(UpdateSymSettings) | ST(UpdateImgSettings) | ST(UpdateMatchSettings)) & appState) {
			lock.release();

			ostringstream oss;
			oss<<"Please don't demand a transformation "
				"while one is being performed, "
				"nor while the settings are changing and "
				"neither while a new image is loaded!";
			errMsg(oss.str());

			return nullptr;
		}
		return makeUnique<const NormalActionPermit>(appState, ST(ImgTransform));
	}
	
	if(&controlName == &ControlPanel_restoreDefaultsLabel) {
		if((ST(ImgTransform) | ST(UpdateMatchSettings)
				| ST(LoadAllSettings) | ST(LoadMatchSettings)
				| ST(SaveAllSettings) | ST(SaveMatchSettings)) & appState) {
			lock.release();

			ostringstream oss;
			oss<<"Please don't update the match aspects settings "
				"before the completion of similar previous updates, "
				"nor while they're saved and "
				"neither when an image transformation is performed based on them!";
			errMsg(oss.str());

			return nullptr;
		}
		return makeUnique<const NormalActionPermit>(appState,
				ST(LoadMatchSettings) | ST(UpdateMatchSettings));
	}
	
	if(&controlName == &ControlPanel_saveAsDefaultsLabel) {
		if((ST(UpdateMatchSettings) | ST(SaveMatchSettings)
				| ST(LoadAllSettings) | ST(LoadMatchSettings)) & appState) {
			lock.release();

			ostringstream oss;
			oss<<"Please don't save the match aspects settings "
				"before the completion of the same command issued previously, "
				"nor when these settings are being updated!";
			errMsg(oss.str());

			return nullptr;
		}
		return makeUnique<const NormalActionPermit>(appState, ST(SaveMatchSettings));
	}
	
	if(&controlName == &ControlPanel_loadSettingsLabel) {
		if((ST(ImgTransform) | ST(UpdateImgSettings) | ST(UpdateSymSettings) | ST(UpdateMatchSettings)
				| ST(LoadAllSettings) | ST(LoadMatchSettings)
				| ST(SaveAllSettings) | ST(SaveMatchSettings)) & appState) {
			lock.release();

			ostringstream oss;
			oss<<"Please don't update the settings "
				"before the completion of similar previous updates, "
				"nor while they're saved and "
				"neither when an image transformation is performed based on them!";
			errMsg(oss.str());

			return nullptr;
		}
		return makeUnique<const NormalActionPermit>(appState,
				ST(LoadAllSettings) | ST(UpdateImgSettings)
					| ST(UpdateSymSettings) | ST(UpdateMatchSettings));
	}
	
	if(&controlName == &ControlPanel_saveSettingsLabel) {
		if((ST(UpdateImgSettings) | ST(UpdateSymSettings) | ST(UpdateMatchSettings)
				| ST(LoadAllSettings) | ST(LoadMatchSettings)
				| ST(SaveAllSettings)) & appState) {
			lock.release();

			ostringstream oss;
			oss<<"Please don't save the settings "
				"before the completion of the same command issued previously, "
				"nor when these settings are being updated!";
			errMsg(oss.str());

			return nullptr;
		}
		return makeUnique<const NormalActionPermit>(appState, ST(SaveAllSettings));
	}
	
	THROW_WITH_VAR_MSG("No handling yet for " + controlName + " in " __FUNCTION__, domain_error);
}

#endif // UNIT_TESTING not defined
