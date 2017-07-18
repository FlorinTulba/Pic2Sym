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
#include "controlPanelActions.h"
#include "settingsBase.h"
#include "symSettings.h"
#include "imgSettings.h"
#include "matchSettings.h"
#include "sliderConversion.h"
#include "dlgs.h"
#include "misc.h"

#pragma warning ( push, 0 )

#include <thread>

#include <opencv2/highgui/highgui.hpp>

#pragma warning ( pop )

using namespace std;
using namespace cv;

extern const int ControlPanel_Converter_StructuralSim_maxSlider;
extern const double ControlPanel_Converter_StructuralSim_maxReal;
extern const int ControlPanel_Converter_Contrast_maxSlider;
extern const double ControlPanel_Converter_Contrast_maxReal;
extern const int ControlPanel_Converter_Correctness_maxSlider;
extern const double ControlPanel_Converter_Correctness_maxReal;
extern const int ControlPanel_Converter_Direction_maxSlider;
extern const double ControlPanel_Converter_Direction_maxReal;
extern const int ControlPanel_Converter_Gravity_maxSlider;
extern const double ControlPanel_Converter_Gravity_maxReal;
extern const int ControlPanel_Converter_LargerSym_maxSlider;
extern const double ControlPanel_Converter_LargerSym_maxReal;

extern const String ControlPanel_selectImgLabel;
extern const String ControlPanel_transformImgLabel;
extern const String ControlPanel_selectFontLabel;
extern const String ControlPanel_restoreDefaultsLabel;
extern const String ControlPanel_saveAsDefaultsLabel;
extern const String ControlPanel_aboutLabel;
extern const String ControlPanel_instructionsLabel;
extern const String ControlPanel_loadSettingsLabel;
extern const String ControlPanel_saveSettingsLabel;
extern const String ControlPanel_fontSzTrName;
extern const String ControlPanel_encodingTrName;
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
extern const String ControlPanel_symsBatchSzTrName;
extern const wstring ControlPanel_aboutText;
extern const wstring ControlPanel_instructionsText;
extern const unsigned SymsBatch_defaultSz;

const map<const String*, std::shared_ptr<const SliderConverter>>& ControlPanel::slidersConverters() {
#pragma warning ( disable : WARN_THREAD_UNSAFE )
	static std::shared_ptr<const map<const String*, std::shared_ptr<const SliderConverter>>> result;
	static bool initialized = false;
#pragma warning ( default : WARN_THREAD_UNSAFE )
	if(!initialized) {
		result = std::make_shared<const map<const String*, std::shared_ptr<const SliderConverter>>>(
			std::move(map<const String*, std::shared_ptr<const SliderConverter>> {
				{ &ControlPanel_structuralSimTrName,
					std::make_shared<ProportionalSliderValue>(std::make_unique<const ProportionalSliderValue::Params>(
					ControlPanel_Converter_StructuralSim_maxSlider, ControlPanel_Converter_StructuralSim_maxReal)) },

				{ &ControlPanel_underGlyphCorrectnessTrName,
					std::make_shared<ProportionalSliderValue>(std::make_unique<const ProportionalSliderValue::Params>(
					ControlPanel_Converter_Correctness_maxSlider, ControlPanel_Converter_Correctness_maxReal)) },

				{ &ControlPanel_glyphEdgeCorrectnessTrName,
					std::make_shared<ProportionalSliderValue>(std::make_unique<const ProportionalSliderValue::Params>(
					ControlPanel_Converter_Correctness_maxSlider, ControlPanel_Converter_Correctness_maxReal)) },

				{ &ControlPanel_asideGlyphCorrectnessTrName,
					std::make_shared<ProportionalSliderValue>(std::make_unique<const ProportionalSliderValue::Params>(
					ControlPanel_Converter_Correctness_maxSlider, ControlPanel_Converter_Correctness_maxReal)) },

				{ &ControlPanel_moreContrastTrName,
					std::make_shared<ProportionalSliderValue>(std::make_unique<const ProportionalSliderValue::Params>(
					ControlPanel_Converter_Contrast_maxSlider, ControlPanel_Converter_Contrast_maxReal)) },

				{ &ControlPanel_gravityTrName,
					std::make_shared<ProportionalSliderValue>(std::make_unique<const ProportionalSliderValue::Params>(
					ControlPanel_Converter_Gravity_maxSlider, ControlPanel_Converter_Gravity_maxReal)) },

				{ &ControlPanel_directionTrName,
					std::make_shared<ProportionalSliderValue>(std::make_unique<const ProportionalSliderValue::Params>(
					ControlPanel_Converter_Direction_maxSlider, ControlPanel_Converter_Direction_maxReal)) },

				{ &ControlPanel_largerSymTrName,
					std::make_shared<ProportionalSliderValue>(std::make_unique<const ProportionalSliderValue::Params>(
					ControlPanel_Converter_LargerSym_maxSlider, ControlPanel_Converter_LargerSym_maxReal)) }
				
				}));
		initialized = true;
	}

	return *result;
}

void ControlPanel::updateMatchSettings(const MatchSettings &ms) {
	int newVal = ms.isHybridResult();
	while(hybridResult != newVal)
		setTrackbarPos(*(pLuckySliderName = &ControlPanel_hybridResultTrName), nullptr, newVal);

	newVal = slidersConverters().at(&ControlPanel_structuralSimTrName)->toSlider(ms.get_kSsim());
	while(structuralSim != newVal)
		setTrackbarPos(*(pLuckySliderName = &ControlPanel_structuralSimTrName), nullptr, newVal);

	newVal = slidersConverters().at(&ControlPanel_underGlyphCorrectnessTrName)->toSlider(ms.get_kSdevFg());
	while(underGlyphCorrectness != newVal)
		setTrackbarPos(*(pLuckySliderName = &ControlPanel_underGlyphCorrectnessTrName), nullptr, newVal);

	newVal = slidersConverters().at(&ControlPanel_glyphEdgeCorrectnessTrName)->toSlider(ms.get_kSdevEdge());
	while(glyphEdgeCorrectness != newVal)
		setTrackbarPos(*(pLuckySliderName = &ControlPanel_glyphEdgeCorrectnessTrName), nullptr, newVal);

	newVal = slidersConverters().at(&ControlPanel_asideGlyphCorrectnessTrName)->toSlider(ms.get_kSdevBg());
	while(asideGlyphCorrectness != newVal)
		setTrackbarPos(*(pLuckySliderName = &ControlPanel_asideGlyphCorrectnessTrName), nullptr, newVal);

	newVal = slidersConverters().at(&ControlPanel_moreContrastTrName)->toSlider(ms.get_kContrast());
	while(moreContrast != newVal)
		setTrackbarPos(*(pLuckySliderName = &ControlPanel_moreContrastTrName), nullptr, newVal);

	newVal = slidersConverters().at(&ControlPanel_gravityTrName)->toSlider(ms.get_kMCsOffset());
	while(gravity != newVal)
		setTrackbarPos(*(pLuckySliderName = &ControlPanel_gravityTrName), nullptr, newVal);

	newVal = slidersConverters().at(&ControlPanel_directionTrName)->toSlider(ms.get_kCosAngleMCs());
	while(direction != newVal)
		setTrackbarPos(*(pLuckySliderName = &ControlPanel_directionTrName), nullptr, newVal);

	newVal = slidersConverters().at(&ControlPanel_largerSymTrName)->toSlider(ms.get_kSymDensity());
	while(largerSym != newVal)
		setTrackbarPos(*(pLuckySliderName = &ControlPanel_largerSymTrName), nullptr, newVal);

	newVal = (int)ms.getBlankThreshold();
	while(thresh4Blanks != newVal)
		setTrackbarPos(*(pLuckySliderName = &ControlPanel_thresh4BlanksTrName), nullptr, newVal);

	pLuckySliderName = nullptr;
}

void ControlPanel::updateImgSettings(const ImgSettings &is) {
	int newVal = (int)is.getMaxHSyms();
	while(maxHSyms != newVal)
		setTrackbarPos(*(pLuckySliderName = &ControlPanel_outWTrName), nullptr, newVal);

	newVal = (int)is.getMaxVSyms();
	while(maxVSyms != newVal)
		setTrackbarPos(*(pLuckySliderName = &ControlPanel_outHTrName), nullptr, newVal);

	pLuckySliderName = nullptr;
}

void ControlPanel::updateSymSettings(unsigned encIdx, unsigned fontSz_) {
	while(encoding != (int)encIdx)
		setTrackbarPos(*(pLuckySliderName = &ControlPanel_encodingTrName), nullptr, (int)encIdx);

	while(fontSz != (int)fontSz_)
		setTrackbarPos(*(pLuckySliderName = &ControlPanel_fontSzTrName), nullptr, (int)fontSz_);

	pLuckySliderName = nullptr;
}

void ControlPanel::updateEncodingsCount(unsigned uniqueEncodings) {
	updatingEncMax = true;
	setTrackbarMax(ControlPanel_encodingTrName, nullptr, max(1, int(uniqueEncodings-1U)));

	// Sequence from below is required to really update the trackbar max & pos
	// The controller should prevent them to trigger Controller::newFontEncoding
	setTrackbarPos(*(pLuckySliderName = &ControlPanel_encodingTrName), nullptr, 1);
	updatingEncMax = false;
	setTrackbarPos(ControlPanel_encodingTrName, nullptr, 0);
	pLuckySliderName = nullptr;
}

void ControlPanel::restoreSliderValue(const String &trName, const string &errText) {
	// Determine previous value
	int prevVal = 0;
	if(&trName == &ControlPanel_outWTrName) {
		prevVal = (int)cfg.getIS().getMaxHSyms();
	} else if(&trName == &ControlPanel_outHTrName) {
		prevVal = (int)cfg.getIS().getMaxVSyms();
	} else if(&trName == &ControlPanel_encodingTrName) {
		prevVal = (int)performer.getFontEncodingIdx();
	} else if(&trName == &ControlPanel_fontSzTrName) {
		prevVal = (int)cfg.getSS().getFontSz();
	} else if(&trName == &ControlPanel_symsBatchSzTrName) {
		prevVal = symsBatchSz; // no change needed for Symbols Batch Size!
	} else if(&trName == &ControlPanel_hybridResultTrName) {
		prevVal = cfg.getMS().isHybridResult() ? 1 : 0;
	} else if(&trName == &ControlPanel_structuralSimTrName) {
		prevVal = slidersConverters().at(&ControlPanel_structuralSimTrName)->toSlider(cfg.getMS().get_kSsim());
	} else if(&trName == &ControlPanel_underGlyphCorrectnessTrName) {
		prevVal = slidersConverters().at(&ControlPanel_underGlyphCorrectnessTrName)->toSlider(cfg.getMS().get_kSdevFg());
	} else if(&trName == &ControlPanel_glyphEdgeCorrectnessTrName) {
		prevVal = slidersConverters().at(&ControlPanel_glyphEdgeCorrectnessTrName)->toSlider(cfg.getMS().get_kSdevEdge());
	} else if(&trName == &ControlPanel_asideGlyphCorrectnessTrName) {
		prevVal = slidersConverters().at(&ControlPanel_asideGlyphCorrectnessTrName)->toSlider(cfg.getMS().get_kSdevBg());
	} else if(&trName == &ControlPanel_moreContrastTrName) {
		prevVal = slidersConverters().at(&ControlPanel_moreContrastTrName)->toSlider(cfg.getMS().get_kContrast());
	} else if(&trName == &ControlPanel_gravityTrName) {
		prevVal = slidersConverters().at(&ControlPanel_gravityTrName)->toSlider(cfg.getMS().get_kMCsOffset());
	} else if(&trName == &ControlPanel_directionTrName) {
		prevVal = slidersConverters().at(&ControlPanel_directionTrName)->toSlider(cfg.getMS().get_kCosAngleMCs());
	} else if(&trName == &ControlPanel_largerSymTrName) {
		prevVal = slidersConverters().at(&ControlPanel_largerSymTrName)->toSlider(cfg.getMS().get_kSymDensity());
	} else if(&trName == &ControlPanel_thresh4BlanksTrName) {
		prevVal = (int)cfg.getMS().getBlankThreshold();
	} else THROW_WITH_VAR_MSG("Code for " + trName + " must be added within " __FUNCTION__, domain_error);

	// Deals with the case when the value was already restored / not modified at all
	if(getTrackbarPos(trName, nullptr) == prevVal)
		return;

	slidersRestoringValue.insert(trName);

	thread([&] (const String sliderName, int previousVal, const string errorText) {
				errMsg(errorText);

				// Set the previous value
				while(getTrackbarPos(sliderName, nullptr) != previousVal)
					setTrackbarPos(sliderName, nullptr, previousVal);

				slidersRestoringValue.erase(sliderName);
			},
			trName, prevVal, errText
	).detach();
}

ControlPanel::ControlPanel(IControlPanelActions &performer_, const ISettings &cfg_) :
		performer(performer_), cfg(cfg_),
		maxHSyms((int)cfg_.getIS().getMaxHSyms()), maxVSyms((int)cfg_.getIS().getMaxVSyms()),
		encoding(0), fontSz((int)cfg_.getSS().getFontSz()),
		symsBatchSz((int)SymsBatch_defaultSz),
		hybridResult(cfg_.getMS().isHybridResult() ? 1 : 0),
		structuralSim(slidersConverters().at(&ControlPanel_structuralSimTrName)->toSlider(cfg_.getMS().get_kSsim())),
		underGlyphCorrectness(slidersConverters().at(&ControlPanel_underGlyphCorrectnessTrName)->toSlider(cfg_.getMS().get_kSdevFg())),
		glyphEdgeCorrectness(slidersConverters().at(&ControlPanel_glyphEdgeCorrectnessTrName)->toSlider(cfg_.getMS().get_kSdevEdge())),
		asideGlyphCorrectness(slidersConverters().at(&ControlPanel_asideGlyphCorrectnessTrName)->toSlider(cfg_.getMS().get_kSdevBg())),
		moreContrast(slidersConverters().at(&ControlPanel_moreContrastTrName)->toSlider(cfg_.getMS().get_kContrast())),
		gravity(slidersConverters().at(&ControlPanel_gravityTrName)->toSlider(cfg_.getMS().get_kMCsOffset())),
		direction(slidersConverters().at(&ControlPanel_directionTrName)->toSlider(cfg_.getMS().get_kCosAngleMCs())),
		largerSym(slidersConverters().at(&ControlPanel_largerSymTrName)->toSlider(cfg_.getMS().get_kSymDensity())),
		thresh4Blanks((int)cfg_.getMS().getBlankThreshold()) {
	extern const unsigned Settings_MAX_THRESHOLD_FOR_BLANKS;
	extern const unsigned Settings_MAX_H_SYMS;
	extern const unsigned Settings_MAX_V_SYMS;
	extern const unsigned Settings_MAX_FONT_SIZE;
	extern const unsigned SymsBatch_trackMax;
	extern const int ControlPanel_Converter_StructuralSim_maxSlider;
	extern const int ControlPanel_Converter_Contrast_maxSlider;
	extern const int ControlPanel_Converter_Correctness_maxSlider;
	extern const int ControlPanel_Converter_Direction_maxSlider;
	extern const int ControlPanel_Converter_Gravity_maxSlider;
	extern const int ControlPanel_Converter_LargerSym_maxSlider;

	extern const String ControlPanel_selectImgLabel;
	extern const String ControlPanel_transformImgLabel;
	extern const String ControlPanel_selectFontLabel;
	extern const String ControlPanel_restoreDefaultsLabel;
	extern const String ControlPanel_saveAsDefaultsLabel;
	extern const String ControlPanel_loadSettingsLabel;
	extern const String ControlPanel_saveSettingsLabel;
	extern const String ControlPanel_fontSzTrName;
	extern const String ControlPanel_encodingTrName;
	extern const String ControlPanel_symsBatchSzTrName;
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

	createButton(ControlPanel_selectImgLabel,
				 [] (int, void *userdata) {
		IControlPanelActions *pActions = reinterpret_cast<IControlPanelActions*>(userdata);
#pragma warning ( disable : WARN_THREAD_UNSAFE )
		static ImgSelector is;
#pragma warning ( default : WARN_THREAD_UNSAFE )

		if(is.promptForUserChoice())
			pActions->newImage(is.selection());
	}, reinterpret_cast<void*>(&performer));
	createButton(ControlPanel_transformImgLabel,
				 [] (int, void *userdata) {
		IControlPanelActions *pActions = reinterpret_cast<IControlPanelActions*>(userdata);
		pActions->performTransformation();
	}, reinterpret_cast<void*>(&performer));

	createTrackbar(ControlPanel_outWTrName, nullptr, &maxHSyms, (int)Settings_MAX_H_SYMS,
				   [] (int val, void *userdata) {
		IControlPanelActions *pActions = reinterpret_cast<IControlPanelActions*>(userdata);
		pActions->newHmaxSyms(val);
	}, reinterpret_cast<void*>(&performer));
	createTrackbar(ControlPanel_outHTrName, nullptr, &maxVSyms, (int)Settings_MAX_V_SYMS,
				   [] (int val, void *userdata) {
		IControlPanelActions *pActions = reinterpret_cast<IControlPanelActions*>(userdata);
		pActions->newVmaxSyms(val);
	}, reinterpret_cast<void*>(&performer));

	createButton(ControlPanel_selectFontLabel,
				 [] (int, void *userdata) {
		IControlPanelActions *pActions = reinterpret_cast<IControlPanelActions*>(userdata);

#pragma warning ( disable : WARN_THREAD_UNSAFE )
		static SelectFont sf;
#pragma warning ( default : WARN_THREAD_UNSAFE )

		try {
			if(sf.promptForUserChoice())
				pActions->newFontFamily(sf.selection());
		} catch(FontLocationFailure&) {
			pActions->invalidateFont();
			extern const string CannotLoadFontErrSuffix;
			infoMsg("Couldn't locate the selected font!" + CannotLoadFontErrSuffix, "Manageable Error");
		}
	}, reinterpret_cast<void*>(&performer));

	createTrackbar(ControlPanel_encodingTrName, nullptr, &encoding, 1,
				   [] (int val, void *userdata) {
		IControlPanelActions *pActions = reinterpret_cast<IControlPanelActions*>(userdata);
		pActions->newFontEncoding(val);
	}, reinterpret_cast<void*>(&performer));
	createTrackbar(ControlPanel_fontSzTrName, nullptr, &fontSz, (int)Settings_MAX_FONT_SIZE,
				   [] (int val, void *userdata) {
		IControlPanelActions *pActions = reinterpret_cast<IControlPanelActions*>(userdata);
		pActions->newFontSize(val);
	}, reinterpret_cast<void*>(&performer));

	createTrackbar(ControlPanel_symsBatchSzTrName, nullptr, &symsBatchSz, (int)SymsBatch_trackMax,
				   [] (int val, void *userdata) {
		IControlPanelActions *pActions = reinterpret_cast<IControlPanelActions*>(userdata);
		pActions->newSymsBatchSize(val);
	}, reinterpret_cast<void*>(&performer));

	createButton(ControlPanel_restoreDefaultsLabel, [] (int, void *userdata) {
		IControlPanelActions *pActions = reinterpret_cast<IControlPanelActions*>(userdata);
		pActions->restoreUserDefaultMatchSettings();
	}, reinterpret_cast<void*>(&performer));
	createButton(ControlPanel_saveAsDefaultsLabel, [] (int, void *userdata) {
		IControlPanelActions *pActions = reinterpret_cast<IControlPanelActions*>(userdata);
		pActions->setUserDefaultMatchSettings();
	}, reinterpret_cast<void*>(&performer));

	createTrackbar(ControlPanel_hybridResultTrName, nullptr, &hybridResult, 1,
				   [] (int state, void *userdata) {
		IControlPanelActions *pActions = reinterpret_cast<IControlPanelActions*>(userdata);
		pActions->setResultMode(state != 0);
	}, reinterpret_cast<void*>(&performer));

	createTrackbar(ControlPanel_structuralSimTrName, nullptr, &structuralSim, ControlPanel_Converter_StructuralSim_maxSlider,
				   [] (int val, void *userdata) {
		extern const String ControlPanel_structuralSimTrName; // redeclared within lambda, since no capture is allowed
		IControlPanelActions *pActions = reinterpret_cast<IControlPanelActions*>(userdata);
		pActions->newStructuralSimilarityFactor(slidersConverters().at(&ControlPanel_structuralSimTrName)->fromSlider(val));
	}, reinterpret_cast<void*>(&performer));
	createTrackbar(ControlPanel_underGlyphCorrectnessTrName, nullptr, &underGlyphCorrectness, ControlPanel_Converter_Correctness_maxSlider,
				   [] (int val, void *userdata) {
		extern const String ControlPanel_underGlyphCorrectnessTrName; // redeclared within lambda, since no capture is allowed
		IControlPanelActions *pActions = reinterpret_cast<IControlPanelActions*>(userdata);
		pActions->newUnderGlyphCorrectnessFactor(slidersConverters().at(&ControlPanel_underGlyphCorrectnessTrName)->fromSlider(val));
	}, reinterpret_cast<void*>(&performer));
	createTrackbar(ControlPanel_glyphEdgeCorrectnessTrName, nullptr, &glyphEdgeCorrectness, ControlPanel_Converter_Correctness_maxSlider,
				   [] (int val, void *userdata) {
		extern const String ControlPanel_glyphEdgeCorrectnessTrName; // redeclared within lambda, since no capture is allowed
		IControlPanelActions *pActions = reinterpret_cast<IControlPanelActions*>(userdata);
		pActions->newGlyphEdgeCorrectnessFactor(slidersConverters().at(&ControlPanel_glyphEdgeCorrectnessTrName)->fromSlider(val));
	}, reinterpret_cast<void*>(&performer));
	createTrackbar(ControlPanel_asideGlyphCorrectnessTrName, nullptr, &asideGlyphCorrectness, ControlPanel_Converter_Correctness_maxSlider,
				   [] (int val, void *userdata) {
		extern const String ControlPanel_asideGlyphCorrectnessTrName; // redeclared within lambda, since no capture is allowed
		IControlPanelActions *pActions = reinterpret_cast<IControlPanelActions*>(userdata);
		pActions->newAsideGlyphCorrectnessFactor(slidersConverters().at(&ControlPanel_asideGlyphCorrectnessTrName)->fromSlider(val));
	}, reinterpret_cast<void*>(&performer));

	createTrackbar(ControlPanel_moreContrastTrName, nullptr, &moreContrast, ControlPanel_Converter_Contrast_maxSlider,
				   [] (int val, void *userdata) {
		extern const String ControlPanel_moreContrastTrName; // redeclared within lambda, since no capture is allowed
		IControlPanelActions *pActions = reinterpret_cast<IControlPanelActions*>(userdata);
		pActions->newContrastFactor(slidersConverters().at(&ControlPanel_moreContrastTrName)->fromSlider(val));
	}, reinterpret_cast<void*>(&performer));

	createTrackbar(ControlPanel_gravityTrName, nullptr, &gravity, ControlPanel_Converter_Gravity_maxSlider,
				   [] (int val, void *userdata) {
		extern const String ControlPanel_gravityTrName; // redeclared within lambda, since no capture is allowed
		IControlPanelActions *pActions = reinterpret_cast<IControlPanelActions*>(userdata);
		pActions->newGravitationalSmoothnessFactor(slidersConverters().at(&ControlPanel_gravityTrName)->fromSlider(val));
	}, reinterpret_cast<void*>(&performer));
	createTrackbar(ControlPanel_directionTrName, nullptr, &direction, ControlPanel_Converter_Direction_maxSlider,
				   [] (int val, void *userdata) {
		extern const String ControlPanel_directionTrName; // redeclared within lambda, since no capture is allowed
		IControlPanelActions *pActions = reinterpret_cast<IControlPanelActions*>(userdata);
		pActions->newDirectionalSmoothnessFactor(slidersConverters().at(&ControlPanel_directionTrName)->fromSlider(val));
	}, reinterpret_cast<void*>(&performer));

	createTrackbar(ControlPanel_largerSymTrName, nullptr, &largerSym, ControlPanel_Converter_LargerSym_maxSlider,
				   [] (int val, void *userdata) {
		extern const String ControlPanel_largerSymTrName; // redeclared within lambda, since no capture is allowed
		IControlPanelActions *pActions = reinterpret_cast<IControlPanelActions*>(userdata);
		pActions->newGlyphWeightFactor(slidersConverters().at(&ControlPanel_largerSymTrName)->fromSlider(val));
	}, reinterpret_cast<void*>(&performer));

	createTrackbar(ControlPanel_thresh4BlanksTrName, nullptr, &thresh4Blanks, (int)Settings_MAX_THRESHOLD_FOR_BLANKS,
				   [] (int val, void *userdata) {
		IControlPanelActions *pActions = reinterpret_cast<IControlPanelActions*>(userdata);
		pActions->newThreshold4BlanksFactor((unsigned)val);
	}, reinterpret_cast<void*>(&performer));

	createButton(ControlPanel_aboutLabel, [] (int, void *userdata) {
		IControlPanelActions *pActions = reinterpret_cast<IControlPanelActions*>(userdata);
		pActions->showAboutDlg(ControlPanel_aboutLabel, ControlPanel_aboutText);
	}, reinterpret_cast<void*>(&performer));
	createButton(ControlPanel_instructionsLabel, [] (int, void *userdata) {
		IControlPanelActions *pActions = reinterpret_cast<IControlPanelActions*>(userdata);
		pActions->showInstructionsDlg(ControlPanel_instructionsLabel, ControlPanel_instructionsText);
	}, reinterpret_cast<void*>(&performer));
	createButton(ControlPanel_loadSettingsLabel, [] (int, void *userdata) {
		IControlPanelActions *pActions = reinterpret_cast<IControlPanelActions*>(userdata);
		pActions->loadSettings();
	}, reinterpret_cast<void*>(&performer));
	createButton(ControlPanel_saveSettingsLabel, [] (int, void *userdata) {
		IControlPanelActions *pActions = reinterpret_cast<IControlPanelActions*>(userdata);
		pActions->saveSettings();
	}, reinterpret_cast<void*>(&performer));
}

#endif // UNIT_TESTING not defined
