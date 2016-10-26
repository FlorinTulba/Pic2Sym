/************************************************************************************************
 The application Pic2Sym approximates images by a
 grid of colored symbols with colored backgrounds.

 This file belongs to the Pic2Sym project.

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
 ***********************************************************************************************/

#include "controlPanel.h"
#include "controlPanelActions.h"
#include "settings.h"
#include "misc.h"

#include <thread>

#include <opencv2/highgui.hpp>

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

double ControlPanel::Converter::proportionRule(double x, double xMax, double yMax) {
	return x * yMax / xMax;
}

int ControlPanel::Converter::StructuralSim::toSlider(double ssim) {
	return (int)round(proportionRule(ssim, ControlPanel_Converter_StructuralSim_maxReal, ControlPanel_Converter_StructuralSim_maxSlider));
}

double ControlPanel::Converter::StructuralSim::fromSlider(int ssim) {
	return proportionRule(ssim, ControlPanel_Converter_StructuralSim_maxSlider, ControlPanel_Converter_StructuralSim_maxReal);
}

int ControlPanel::Converter::Contrast::toSlider(double contrast) {
	return (int)round(proportionRule(contrast, ControlPanel_Converter_Contrast_maxReal, ControlPanel_Converter_Contrast_maxSlider));
}

double ControlPanel::Converter::Contrast::fromSlider(int contrast) {
	return proportionRule(contrast, ControlPanel_Converter_Contrast_maxSlider, ControlPanel_Converter_Contrast_maxReal);
}

int ControlPanel::Converter::Correctness::toSlider(double correctness) {
	return (int)round(proportionRule(correctness, ControlPanel_Converter_Correctness_maxReal, ControlPanel_Converter_Correctness_maxSlider));
}

double ControlPanel::Converter::Correctness::fromSlider(int correctness) {
	return proportionRule(correctness, ControlPanel_Converter_Correctness_maxSlider, ControlPanel_Converter_Correctness_maxReal);
}

int ControlPanel::Converter::Direction::toSlider(double direction) {
	return (int)round(proportionRule(direction, ControlPanel_Converter_Direction_maxReal, ControlPanel_Converter_Direction_maxSlider));
}

double ControlPanel::Converter::Direction::fromSlider(int direction) {
	return proportionRule(direction, ControlPanel_Converter_Direction_maxSlider, ControlPanel_Converter_Direction_maxReal);
}

int ControlPanel::Converter::Gravity::toSlider(double gravity) {
	return (int)round(proportionRule(gravity, ControlPanel_Converter_Gravity_maxReal, ControlPanel_Converter_Gravity_maxSlider));
}

double ControlPanel::Converter::Gravity::fromSlider(int gravity) {
	return proportionRule(gravity, ControlPanel_Converter_Gravity_maxSlider, ControlPanel_Converter_Gravity_maxReal);
}

int ControlPanel::Converter::LargerSym::toSlider(double largerSym) {
	return (int)round(proportionRule(largerSym, ControlPanel_Converter_LargerSym_maxReal, ControlPanel_Converter_LargerSym_maxSlider));
}

double ControlPanel::Converter::LargerSym::fromSlider(int largerSym) {
	return proportionRule(largerSym, ControlPanel_Converter_LargerSym_maxSlider, ControlPanel_Converter_LargerSym_maxReal);
}

void ControlPanel::updateMatchSettings(const MatchSettings &ms) {
	int newVal = ms.isHybridResult();
	while(hybridResult != newVal)
		setTrackbarPos(*(pLuckySliderName = &ControlPanel_hybridResultTrName), nullptr, newVal);

	newVal = Converter::StructuralSim::toSlider(ms.get_kSsim());
	while(structuralSim != newVal)
		setTrackbarPos(*(pLuckySliderName = &ControlPanel_structuralSimTrName), nullptr, newVal);

	newVal = Converter::Correctness::toSlider(ms.get_kSdevFg());
	while(underGlyphCorrectness != newVal)
		setTrackbarPos(*(pLuckySliderName = &ControlPanel_underGlyphCorrectnessTrName), nullptr, newVal);

	newVal = Converter::Correctness::toSlider(ms.get_kSdevEdge());
	while(glyphEdgeCorrectness != newVal)
		setTrackbarPos(*(pLuckySliderName = &ControlPanel_glyphEdgeCorrectnessTrName), nullptr, newVal);

	newVal = Converter::Correctness::toSlider(ms.get_kSdevBg());
	while(asideGlyphCorrectness != newVal)
		setTrackbarPos(*(pLuckySliderName = &ControlPanel_asideGlyphCorrectnessTrName), nullptr, newVal);

	newVal = Converter::Contrast::toSlider(ms.get_kContrast());
	while(moreContrast != newVal)
		setTrackbarPos(*(pLuckySliderName = &ControlPanel_moreContrastTrName), nullptr, newVal);

	newVal = Converter::Gravity::toSlider(ms.get_kMCsOffset());
	while(gravity != newVal)
		setTrackbarPos(*(pLuckySliderName = &ControlPanel_gravityTrName), nullptr, newVal);

	newVal = Converter::Direction::toSlider(ms.get_kCosAngleMCs());
	while(direction != newVal)
		setTrackbarPos(*(pLuckySliderName = &ControlPanel_directionTrName), nullptr, newVal);

	newVal = Converter::LargerSym::toSlider(ms.get_kSymDensity());
	while(largerSym != newVal)
		setTrackbarPos(*(pLuckySliderName = &ControlPanel_largerSymTrName), nullptr, newVal);

	newVal = ms.getBlankThreshold();
	while(thresh4Blanks != newVal)
		setTrackbarPos(*(pLuckySliderName = &ControlPanel_thresh4BlanksTrName), nullptr, newVal);

	pLuckySliderName = nullptr;
}

void ControlPanel::updateImgSettings(const ImgSettings &is) {
	int newVal = is.getMaxHSyms();
	while(maxHSyms != newVal)
		setTrackbarPos(*(pLuckySliderName = &ControlPanel_outWTrName), nullptr, newVal);

	newVal = is.getMaxVSyms();
	while(maxVSyms != newVal)
		setTrackbarPos(*(pLuckySliderName = &ControlPanel_outHTrName), nullptr, newVal);

	pLuckySliderName = nullptr;
}

void ControlPanel::updateSymSettings(unsigned encIdx, unsigned fontSz_) {
	while(encoding != encIdx)
		setTrackbarPos(*(pLuckySliderName = &ControlPanel_encodingTrName), nullptr, encIdx);

	while(fontSz != fontSz_)
		setTrackbarPos(*(pLuckySliderName = &ControlPanel_fontSzTrName), nullptr, fontSz_);

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
		prevVal = cfg.imgSettings().getMaxHSyms();
	} else if(&trName == &ControlPanel_outHTrName) {
		prevVal = cfg.imgSettings().getMaxVSyms();
	} else if(&trName == &ControlPanel_encodingTrName) {
		prevVal = performer.getFontEncodingIdx();
	} else if(&trName == &ControlPanel_fontSzTrName) {
		prevVal = cfg.symSettings().getFontSz();
	} else if(&trName == &ControlPanel_symsBatchSzTrName) {
		prevVal = symsBatchSz; // no change needed for Symbols Batch Size!
	} else if(&trName == &ControlPanel_hybridResultTrName) {
		prevVal = cfg.matchSettings().isHybridResult() ? 1 : 0;
	} else if(&trName == &ControlPanel_structuralSimTrName) {
		prevVal = Converter::StructuralSim::toSlider(cfg.matchSettings().get_kSsim());
	} else if(&trName == &ControlPanel_underGlyphCorrectnessTrName) {
		prevVal = Converter::Correctness::toSlider(cfg.matchSettings().get_kSdevFg());
	} else if(&trName == &ControlPanel_glyphEdgeCorrectnessTrName) {
		prevVal = Converter::Correctness::toSlider(cfg.matchSettings().get_kSdevEdge());
	} else if(&trName == &ControlPanel_asideGlyphCorrectnessTrName) {
		prevVal = Converter::Correctness::toSlider(cfg.matchSettings().get_kSdevBg());
	} else if(&trName == &ControlPanel_moreContrastTrName) {
		prevVal = Converter::Contrast::toSlider(cfg.matchSettings().get_kContrast());
	} else if(&trName == &ControlPanel_gravityTrName) {
		prevVal = Converter::Gravity::toSlider(cfg.matchSettings().get_kMCsOffset());
	} else if(&trName == &ControlPanel_directionTrName) {
		prevVal = Converter::Direction::toSlider(cfg.matchSettings().get_kCosAngleMCs());
	} else if(&trName == &ControlPanel_largerSymTrName) {
		prevVal = Converter::LargerSym::toSlider(cfg.matchSettings().get_kSymDensity());
	} else if(&trName == &ControlPanel_thresh4BlanksTrName) {
		prevVal = cfg.matchSettings().getBlankThreshold();
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
