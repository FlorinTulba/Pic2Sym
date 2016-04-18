/****************************************************************************************
 The application Pic2Sym approximates images by a
 grid of colored symbols with colored backgrounds.

 This file was created on 2016-4-9
 and belongs to the Pic2Sym project.

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
#include "settings.h"

#include <opencv2/highgui.hpp>

using namespace std;
using namespace cv;

extern const int ControlPanel_Converter_StructuralSim_maxSlider,
	ControlPanel_Converter_Contrast_maxSlider,
	ControlPanel_Converter_Correctness_maxSlider,
	ControlPanel_Converter_Direction_maxSlider,
	ControlPanel_Converter_Gravity_maxSlider,
	ControlPanel_Converter_LargerSym_maxSlider;
extern const double ControlPanel_Converter_StructuralSim_maxReal,
	ControlPanel_Converter_Contrast_maxReal,
	ControlPanel_Converter_Correctness_maxReal,
	ControlPanel_Converter_Direction_maxReal,
	ControlPanel_Converter_Gravity_maxReal,
	ControlPanel_Converter_LargerSym_maxReal;
extern const String ControlPanel_selectImgLabel,
	ControlPanel_transformImgLabel,
	ControlPanel_selectFontLabel,
	ControlPanel_restoreDefaultsLabel,
	ControlPanel_saveAsDefaultsLabel,
	ControlPanel_aboutLabel,
	ControlPanel_instructionsLabel,
	ControlPanel_loadSettingsLabel,
	ControlPanel_saveSettingsLabel,
	ControlPanel_fontSzTrName,
	ControlPanel_encodingTrName,
	ControlPanel_hybridResultTrName,
	ControlPanel_structuralSimTrName,
	ControlPanel_underGlyphCorrectnessTrName,
	ControlPanel_glyphEdgeCorrectnessTrName,
	ControlPanel_asideGlyphCorrectnessTrName,
	ControlPanel_moreContrastTrName,
	ControlPanel_gravityTrName,
	ControlPanel_directionTrName,
	ControlPanel_largerSymTrName,
	ControlPanel_thresh4BlanksTrName,
	ControlPanel_outWTrName,
	ControlPanel_outHTrName;

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
		setTrackbarPos(ControlPanel_hybridResultTrName, nullptr, newVal);

	newVal = Converter::StructuralSim::toSlider(ms.get_kSsim());
	while(structuralSim != newVal)
		setTrackbarPos(ControlPanel_structuralSimTrName, nullptr, newVal);

	newVal = Converter::Correctness::toSlider(ms.get_kSdevFg());
	while(underGlyphCorrectness != newVal)
		setTrackbarPos(ControlPanel_underGlyphCorrectnessTrName, nullptr, newVal);

	newVal = Converter::Correctness::toSlider(ms.get_kSdevEdge());
	while(glyphEdgeCorrectness != newVal)
		setTrackbarPos(ControlPanel_glyphEdgeCorrectnessTrName, nullptr, newVal);

	newVal = Converter::Correctness::toSlider(ms.get_kSdevBg());
	while(asideGlyphCorrectness != newVal)
		setTrackbarPos(ControlPanel_asideGlyphCorrectnessTrName, nullptr, newVal);

	newVal = Converter::Contrast::toSlider(ms.get_kContrast());
	while(moreContrast != newVal)
		setTrackbarPos(ControlPanel_moreContrastTrName, nullptr, newVal);

	newVal = Converter::Gravity::toSlider(ms.get_kMCsOffset());
	while(gravity != newVal)
		setTrackbarPos(ControlPanel_gravityTrName, nullptr, newVal);

	newVal = Converter::Direction::toSlider(ms.get_kCosAngleMCs());
	while(direction != newVal)
		setTrackbarPos(ControlPanel_directionTrName, nullptr, newVal);

	newVal = Converter::LargerSym::toSlider(ms.get_kSymDensity());
	while(largerSym != newVal)
		setTrackbarPos(ControlPanel_largerSymTrName, nullptr, newVal);

	newVal = ms.getBlankThreshold();
	while(thresh4Blanks != newVal)
		setTrackbarPos(ControlPanel_thresh4BlanksTrName, nullptr, newVal);
}

void ControlPanel::updateImgSettings(const ImgSettings &is) {
	int newVal = is.getMaxHSyms();
	while(maxHSyms != newVal)
		setTrackbarPos(ControlPanel_outWTrName, nullptr, newVal);

	newVal = is.getMaxVSyms();
	while(maxVSyms != newVal)
		setTrackbarPos(ControlPanel_outHTrName, nullptr, newVal);
}

void ControlPanel::updateSymSettings(unsigned encIdx, unsigned fontSz_) {
	while(encoding != encIdx)
		setTrackbarPos(ControlPanel_encodingTrName, nullptr, encIdx);

	while(fontSz != fontSz_)
		setTrackbarPos(ControlPanel_fontSzTrName, nullptr, fontSz_);
}

void ControlPanel::updateEncodingsCount(unsigned uniqueEncodings) {
	updatingEncMax = true;
	setTrackbarMax(ControlPanel_encodingTrName, nullptr, max(1, int(uniqueEncodings-1U)));

	// Sequence from below is required to really update the trackbar max & pos
	// The controller should prevent them to trigger Controller::newFontEncoding
	setTrackbarPos(ControlPanel_encodingTrName, nullptr, 1);
	updatingEncMax = false;
	setTrackbarPos(ControlPanel_encodingTrName, nullptr, 0);
}
