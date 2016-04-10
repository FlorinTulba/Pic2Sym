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

#include "controller.h"
#include "controlPanel.h"
#include "matchParams.h"
#include "misc.h"
#include "propsReader.h"
#include "settings.h"
#include "views.h"

#include <boost/algorithm/string/replace.hpp>

using namespace std;
using namespace cv;
using namespace boost::algorithm;

/// Replaces all instances of a pattern in a text with a different string.
static string replacePlaceholder(const string &text,	///< initial text
								 const string &placeholder = "$(PIC2SYM_VERSION)",	///< pattern to be replaced
								 const string &replacement = PIC2SYM_VERSION	///< replacement string
								 ) {
	string text_(text);
	replace_all(text_, placeholder, replacement);
	return std::move(text_);
}

/// parser for reading various texts and constants concerning
static PropsReader varConfig("res/varConfig.txt");

// Macros for reading the properties from varConfig
#define READ_PROP(prop, type) prop = varConfig.read<type>(#prop)
#define READ_INT_PROP(prop) READ_PROP(prop, int)
#define READ_UINT_PROP(prop) READ_PROP(prop, unsigned)
#define READ_DOUBLE_PROP(prop) READ_PROP(prop, double)
#define READ_STR_PROP(prop) READ_PROP(prop, string)
#define READ_WSTR_PROP(prop) prop = str2wstr(varConfig.read<string>(#prop))

// Reading data
const unsigned READ_UINT_PROP(Settings::MIN_FONT_SIZE);
const unsigned READ_UINT_PROP(Settings::MAX_FONT_SIZE);
const unsigned READ_UINT_PROP(Settings::DEF_FONT_SIZE);
const unsigned READ_UINT_PROP(Settings::MAX_THRESHOLD_FOR_BLANKS);
const unsigned READ_UINT_PROP(Settings::MIN_H_SYMS);
const unsigned READ_UINT_PROP(Settings::MAX_H_SYMS);
const unsigned READ_UINT_PROP(Settings::MIN_V_SYMS);
const unsigned READ_UINT_PROP(Settings::MAX_V_SYMS);

const double READ_DOUBLE_PROP(PmsCont::SMALL_GLYPHS_PERCENT);
extern const double READ_DOUBLE_PROP(MatchEngine_updateSymbols_STILL_BG);
extern const double READ_DOUBLE_PROP(Transformer_run_THRESHOLD_CONTRAST_BLURRED);

static const int READ_INT_PROP(StructuralSimilarity_RecommendedWindowSide);
const cv::Size StructuralSimilarity::WIN_SIZE(StructuralSimilarity_RecommendedWindowSide, StructuralSimilarity_RecommendedWindowSide);
const double READ_DOUBLE_PROP(StructuralSimilarity::SIGMA);
const double READ_DOUBLE_PROP(StructuralSimilarity::C1);
const double READ_DOUBLE_PROP(StructuralSimilarity::C2);

#ifndef UNIT_TESTING
const int READ_INT_PROP(Comparator::trackMax);
const double READ_DOUBLE_PROP(Comparator::defaultTransparency);

static const int READ_INT_PROP(CmapInspect_width);
static const int READ_INT_PROP(CmapInspect_height);
const Size CmapInspect::pageSz(CmapInspect_width, CmapInspect_height);

const int READ_INT_PROP(ControlPanel::Converter::StructuralSim::maxSlider);
const double READ_DOUBLE_PROP(ControlPanel::Converter::StructuralSim::maxReal);
const int READ_INT_PROP(ControlPanel::Converter::Correctness::maxSlider);
const double READ_DOUBLE_PROP(ControlPanel::Converter::Correctness::maxReal);
const int READ_INT_PROP(ControlPanel::Converter::Contrast::maxSlider);
const double READ_DOUBLE_PROP(ControlPanel::Converter::Contrast::maxReal);
const int READ_INT_PROP(ControlPanel::Converter::Gravity::maxSlider);
const double READ_DOUBLE_PROP(ControlPanel::Converter::Gravity::maxReal);
const int READ_INT_PROP(ControlPanel::Converter::Direction::maxSlider);
const double READ_DOUBLE_PROP(ControlPanel::Converter::Direction::maxReal);
const int READ_INT_PROP(ControlPanel::Converter::LargerSym::maxSlider);
const double READ_DOUBLE_PROP(ControlPanel::Converter::LargerSym::maxReal);
const String READ_STR_PROP(ControlPanel::selectImgLabel);
const String READ_STR_PROP(ControlPanel::transformImgLabel);
const String READ_STR_PROP(ControlPanel::selectFontLabel);
const String READ_STR_PROP(ControlPanel::restoreDefaultsLabel);
const String READ_STR_PROP(ControlPanel::saveAsDefaultsLabel);
const String READ_STR_PROP(ControlPanel::aboutLabel);
const String READ_STR_PROP(ControlPanel::instructionsLabel);
const String READ_STR_PROP(ControlPanel::loadSettingsLabel);
const String READ_STR_PROP(ControlPanel::saveSettingsLabel);
const String READ_STR_PROP(ControlPanel::fontSzTrName);
const String READ_STR_PROP(ControlPanel::encodingTrName);
const String READ_STR_PROP(ControlPanel::hybridResultTrName);
const String READ_STR_PROP(ControlPanel::structuralSimTrName);
const String READ_STR_PROP(ControlPanel::underGlyphCorrectnessTrName);
const String READ_STR_PROP(ControlPanel::glyphEdgeCorrectnessTrName);
const String READ_STR_PROP(ControlPanel::asideGlyphCorrectnessTrName);
const String READ_STR_PROP(ControlPanel::moreContrastTrName);
const String READ_STR_PROP(ControlPanel::gravityTrName);
const String READ_STR_PROP(ControlPanel::directionTrName);
const String READ_STR_PROP(ControlPanel::largerSymTrName);
const String READ_STR_PROP(ControlPanel::thresh4BlanksTrName);
const String READ_STR_PROP(ControlPanel::outWTrName);
const String READ_STR_PROP(ControlPanel::outHTrName);

const String READ_STR_PROP(Comparator::transpTrackName);

const String READ_STR_PROP(CmapInspect::pageTrackName);

const wstring ControlPanel::aboutText = std::move(str2wstr(replacePlaceholder(varConfig.read<string>("ControlPanel::aboutText"))));
const wstring READ_WSTR_PROP(ControlPanel::instructionsText);
#endif

extern const string Comparator_initial_title = replacePlaceholder(varConfig.read<string>("Comparator_initial_title"));
extern const string READ_STR_PROP(Comparator_statusBar);

const string READ_STR_PROP(Controller::PREFIX_GLYPH_PROGRESS);
const string READ_STR_PROP(Controller::PREFIX_TRANSFORMATION_PROGRESS);

extern const string READ_STR_PROP(copyrightText);

#if defined _DEBUG || defined UNIT_TESTING
const wstring READ_WSTR_PROP(MatchParams::HEADER);
const wstring READ_WSTR_PROP(BestMatch::HEADER) + MatchParams::HEADER;
#endif // _DEBUG || UNIT_TESTING

#undef READ_PROP
#undef READ_INT_PROP
#undef READ_UINT_PROP
#undef READ_DOUBLE_PROP
#undef READ_STR_PROP
#undef READ_WSTR_PROP
