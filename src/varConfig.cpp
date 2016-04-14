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

#include "misc.h"
#include "propsReader.h"

#include <boost/algorithm/string/replace.hpp>
#include <opencv2/core/core.hpp>

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
#define READ_BOOL_PROP(prop) READ_PROP(prop, bool)
#define READ_INT_PROP(prop) READ_PROP(prop, int)
#define READ_UINT_PROP(prop) READ_PROP(prop, unsigned)
#define READ_DOUBLE_PROP(prop) READ_PROP(prop, double)
#define READ_STR_PROP(prop) READ_PROP(prop, string)
#define READ_WSTR_PROP(prop) prop = str2wstr(varConfig.read<string>(#prop))

// Reading data
extern const bool READ_BOOL_PROP(Transform_BlurredPatches_InsteadOf_Originals);

extern const unsigned READ_UINT_PROP(Settings_MIN_FONT_SIZE);
extern const unsigned READ_UINT_PROP(Settings_MAX_FONT_SIZE);
extern const unsigned READ_UINT_PROP(Settings_DEF_FONT_SIZE);
extern const unsigned READ_UINT_PROP(Settings_MAX_THRESHOLD_FOR_BLANKS);
extern const unsigned READ_UINT_PROP(Settings_MIN_H_SYMS);
extern const unsigned READ_UINT_PROP(Settings_MAX_H_SYMS);
extern const unsigned READ_UINT_PROP(Settings_MIN_V_SYMS);
extern const unsigned READ_UINT_PROP(Settings_MAX_V_SYMS);

extern const double READ_DOUBLE_PROP(PmsCont_SMALL_GLYPHS_PERCENT);
extern const double READ_DOUBLE_PROP(MatchEngine_updateSymbols_STILL_BG);
extern const double READ_DOUBLE_PROP(Transformer_run_THRESHOLD_CONTRAST_BLURRED);

static const int READ_INT_PROP(StructuralSimilarity_RecommendedWindowSide);
extern const Size StructuralSimilarity_WIN_SIZE(StructuralSimilarity_RecommendedWindowSide, StructuralSimilarity_RecommendedWindowSide);
extern const double READ_DOUBLE_PROP(StructuralSimilarity_SIGMA);
extern const double READ_DOUBLE_PROP(StructuralSimilarity_C1);
extern const double READ_DOUBLE_PROP(StructuralSimilarity_C2);

static const int READ_INT_PROP(BlurWindowSize);
extern const Size BlurWinSize(BlurWindowSize, BlurWindowSize);
extern const double READ_DOUBLE_PROP(BlurStandardDeviation);

#ifndef UNIT_TESTING
extern const int READ_INT_PROP(Comparator_trackMax);
extern const double READ_DOUBLE_PROP(Comparator_defaultTransparency);

static const int READ_INT_PROP(CmapInspect_width);
static const int READ_INT_PROP(CmapInspect_height);
extern const Size CmapInspect_pageSz(CmapInspect_width, CmapInspect_height);

extern const int READ_INT_PROP(ControlPanel_Converter_StructuralSim_maxSlider);
extern const double READ_DOUBLE_PROP(ControlPanel_Converter_StructuralSim_maxReal);
extern const int READ_INT_PROP(ControlPanel_Converter_Correctness_maxSlider);
extern const double READ_DOUBLE_PROP(ControlPanel_Converter_Correctness_maxReal);
extern const int READ_INT_PROP(ControlPanel_Converter_Contrast_maxSlider);
extern const double READ_DOUBLE_PROP(ControlPanel_Converter_Contrast_maxReal);
extern const int READ_INT_PROP(ControlPanel_Converter_Gravity_maxSlider);
extern const double READ_DOUBLE_PROP(ControlPanel_Converter_Gravity_maxReal);
extern const int READ_INT_PROP(ControlPanel_Converter_Direction_maxSlider);
extern const double READ_DOUBLE_PROP(ControlPanel_Converter_Direction_maxReal);
extern const int READ_INT_PROP(ControlPanel_Converter_LargerSym_maxSlider);
extern const double READ_DOUBLE_PROP(ControlPanel_Converter_LargerSym_maxReal);

extern const String READ_STR_PROP(ControlPanel_selectImgLabel);
extern const String READ_STR_PROP(ControlPanel_transformImgLabel);
extern const String READ_STR_PROP(ControlPanel_selectFontLabel);
extern const String READ_STR_PROP(ControlPanel_restoreDefaultsLabel);
extern const String READ_STR_PROP(ControlPanel_saveAsDefaultsLabel);
extern const String READ_STR_PROP(ControlPanel_aboutLabel);
extern const String READ_STR_PROP(ControlPanel_instructionsLabel);
extern const String READ_STR_PROP(ControlPanel_loadSettingsLabel);
extern const String READ_STR_PROP(ControlPanel_saveSettingsLabel);
extern const String READ_STR_PROP(ControlPanel_fontSzTrName);
extern const String READ_STR_PROP(ControlPanel_encodingTrName);
extern const String READ_STR_PROP(ControlPanel_hybridResultTrName);
extern const String READ_STR_PROP(ControlPanel_structuralSimTrName);
extern const String READ_STR_PROP(ControlPanel_underGlyphCorrectnessTrName);
extern const String READ_STR_PROP(ControlPanel_glyphEdgeCorrectnessTrName);
extern const String READ_STR_PROP(ControlPanel_asideGlyphCorrectnessTrName);
extern const String READ_STR_PROP(ControlPanel_moreContrastTrName);
extern const String READ_STR_PROP(ControlPanel_gravityTrName);
extern const String READ_STR_PROP(ControlPanel_directionTrName);
extern const String READ_STR_PROP(ControlPanel_largerSymTrName);
extern const String READ_STR_PROP(ControlPanel_thresh4BlanksTrName);
extern const String READ_STR_PROP(ControlPanel_outWTrName);
extern const String READ_STR_PROP(ControlPanel_outHTrName);

extern const String READ_STR_PROP(Comparator_transpTrackName);

extern const String READ_STR_PROP(CmapInspect_pageTrackName);

extern const wstring ControlPanel_aboutText = std::move(str2wstr(replacePlaceholder(varConfig.read<string>("ControlPanel_aboutText"))));
extern const wstring READ_WSTR_PROP(ControlPanel_instructionsText);
#endif

extern const string Comparator_initial_title = replacePlaceholder(varConfig.read<string>("Comparator_initial_title"));
extern const string READ_STR_PROP(Comparator_statusBar);

extern const string READ_STR_PROP(Controller_PREFIX_GLYPH_PROGRESS);
extern const string READ_STR_PROP(Controller_PREFIX_TRANSFORMATION_PROGRESS);

extern const string READ_STR_PROP(copyrightText);

#if defined _DEBUG || defined UNIT_TESTING
extern const wstring READ_WSTR_PROP(MatchParams_HEADER);
extern const wstring READ_WSTR_PROP(BestMatch_HEADER) + MatchParams_HEADER;
#endif // _DEBUG || UNIT_TESTING

#undef READ_PROP
#undef READ_BOOL_PROP
#undef READ_INT_PROP
#undef READ_UINT_PROP
#undef READ_DOUBLE_PROP
#undef READ_STR_PROP
#undef READ_WSTR_PROP
