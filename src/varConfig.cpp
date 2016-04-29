/****************************************************************************************
 The application Pic2Sym approximates images by a
 grid of colored symbols with colored backgrounds.

 This file was created on 2016-4-10
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

/// parser for reading various texts and constants customizing the runtime look and behavior
static PropsReader varConfig("res/varConfig.txt");

// Macros for reading the properties from 'varConfig.txt'
#define READ_PROP(prop, type) \
	const type prop = varConfig.read<type>(#prop)

#define READ_PROP_COND(prop, type, cond, defaultVal) \
	const type prop = (cond) ? varConfig.read<type>(#prop) : (defaultVal)

#define READ_BOOL_PROP(prop) \
	READ_PROP(prop, bool)

#define READ_BOOL_PROP_COND(prop, cond) \
	READ_PROP_COND(prop, bool, cond, false)

#define READ_INT_PROP(prop) \
	READ_PROP(prop, int)

#define READ_INT_PROP_COND(prop, cond, defaultVal) \
	READ_PROP_COND(prop, int, cond, defaultVal)

#define READ_UINT_PROP(prop) \
	READ_PROP(prop, unsigned)

#define READ_DOUBLE_PROP(prop) \
	READ_PROP(prop, double)

#define READ_STR_PROP(prop) \
	READ_PROP(prop, string)

#define READ_STR_PROP_CONVERT(prop, destStringType) \
	const destStringType prop = varConfig.read<string>(#prop)

#define READ_WSTR_PROP(prop) \
	const wstring prop = str2wstr(varConfig.read<string>(#prop))

// Reading data
extern READ_BOOL_PROP(Transform_BlurredPatches_InsteadOf_Originals);

extern READ_BOOL_PROP(UsingOMP); // global OpenMP switch
extern READ_BOOL_PROP_COND(ParallelizePixMapStatistics, UsingOMP);
extern READ_BOOL_PROP_COND(ParallelizeHorVertGlyphFitting, UsingOMP);
extern READ_BOOL_PROP_COND(ParallelizeHorVertGlyphReductions, UsingOMP);
extern READ_BOOL_PROP_COND(ParallelizeGridCreation, UsingOMP);
extern READ_BOOL_PROP_COND(ParallelizeGridPopulation, UsingOMP);
extern READ_INT_PROP_COND(MinRowsToParallelizeGlyphBitmapExtraction, UsingOMP, INT_MAX);
extern READ_BOOL_PROP_COND(PrepareMoreGlyphsAtOnce, UsingOMP);
extern READ_BOOL_PROP_COND(ParallelizeTr_PatchRowLoops, UsingOMP);

extern READ_BOOL_PROP_COND(ParallelizeGlyphMasks, UsingOMP);
extern READ_BOOL_PROP_COND(ParallelizeTr_PatchColLoops, UsingOMP);
extern READ_BOOL_PROP_COND(ParallelizeMp_MassCenters, UsingOMP);
extern READ_BOOL_PROP_COND(ParallelizeMp_GlyphSumAndReductions, UsingOMP);
extern READ_BOOL_PROP_COND(ParallelizeMp_ContrastAndDensity, UsingOMP);
extern READ_BOOL_PROP_COND(ParallelizeMp_FgBgMeans, UsingOMP);
extern READ_BOOL_PROP_COND(ParallelizeMp_VarianceAndPatchApprox, UsingOMP);
extern READ_BOOL_PROP_COND(ParallelizeMp_BPAS_VPA, UsingOMP);
extern READ_BOOL_PROP_COND(ParallelizeMp_PAP_BPBPA, UsingOMP);
extern READ_BOOL_PROP_COND(ParallelizeMp_ssimFactors, UsingOMP);
extern READ_BOOL_PROP_COND(ParallelizeMp_CheckBPA_VPA, UsingOMP);
extern READ_BOOL_PROP_COND(ParallelizeLoggingAndResultAssembly, UsingOMP);
extern READ_BOOL_PROP_COND(ParallelizeBm_HybridStdDevs, UsingOMP);
extern READ_BOOL_PROP_COND(ParallelizeBm_ColorPatchApprox, UsingOMP);
extern READ_BOOL_PROP_COND(ParallelizeBm_ColorPatchFgBgMeans, UsingOMP);

extern READ_UINT_PROP(Settings_MIN_FONT_SIZE);
extern READ_UINT_PROP(Settings_MAX_FONT_SIZE);
extern READ_UINT_PROP(Settings_DEF_FONT_SIZE);
extern READ_UINT_PROP(Settings_MAX_THRESHOLD_FOR_BLANKS);
extern READ_UINT_PROP(Settings_MIN_H_SYMS);
extern READ_UINT_PROP(Settings_MAX_H_SYMS);
extern READ_UINT_PROP(Settings_MIN_V_SYMS);
extern READ_UINT_PROP(Settings_MAX_V_SYMS);

extern READ_DOUBLE_PROP(PmsCont_SMALL_GLYPHS_PERCENT);
extern READ_DOUBLE_PROP(MatchEngine_updateSymbols_STILL_BG);
extern READ_DOUBLE_PROP(Transformer_run_THRESHOLD_CONTRAST_BLURRED);

static READ_INT_PROP(StructuralSimilarity_RecommendedWindowSide);
extern const Size StructuralSimilarity_WIN_SIZE(StructuralSimilarity_RecommendedWindowSide, StructuralSimilarity_RecommendedWindowSide);
extern READ_DOUBLE_PROP(StructuralSimilarity_SIGMA);
extern READ_DOUBLE_PROP(StructuralSimilarity_C1);
extern READ_DOUBLE_PROP(StructuralSimilarity_C2);

static READ_INT_PROP(BlurWindowSize);
extern const Size BlurWinSize(BlurWindowSize, BlurWindowSize);
extern READ_DOUBLE_PROP(BlurStandardDeviation);

#ifndef UNIT_TESTING

extern READ_INT_PROP(Comparator_trackMax);
extern READ_DOUBLE_PROP(Comparator_defaultTransparency);

static READ_INT_PROP(CmapInspect_width);
static READ_INT_PROP(CmapInspect_height);
extern const Size CmapInspect_pageSz(CmapInspect_width, CmapInspect_height);

extern READ_INT_PROP(ControlPanel_Converter_StructuralSim_maxSlider);
extern READ_DOUBLE_PROP(ControlPanel_Converter_StructuralSim_maxReal);
extern READ_INT_PROP(ControlPanel_Converter_Correctness_maxSlider);
extern READ_DOUBLE_PROP(ControlPanel_Converter_Correctness_maxReal);
extern READ_INT_PROP(ControlPanel_Converter_Contrast_maxSlider);
extern READ_DOUBLE_PROP(ControlPanel_Converter_Contrast_maxReal);
extern READ_INT_PROP(ControlPanel_Converter_Gravity_maxSlider);
extern READ_DOUBLE_PROP(ControlPanel_Converter_Gravity_maxReal);
extern READ_INT_PROP(ControlPanel_Converter_Direction_maxSlider);
extern READ_DOUBLE_PROP(ControlPanel_Converter_Direction_maxReal);
extern READ_INT_PROP(ControlPanel_Converter_LargerSym_maxSlider);
extern READ_DOUBLE_PROP(ControlPanel_Converter_LargerSym_maxReal);

extern READ_STR_PROP_CONVERT(ControlPanel_selectImgLabel, String);
extern READ_STR_PROP_CONVERT(ControlPanel_transformImgLabel, String);
extern READ_STR_PROP_CONVERT(ControlPanel_selectFontLabel, String);
extern READ_STR_PROP_CONVERT(ControlPanel_restoreDefaultsLabel, String);
extern READ_STR_PROP_CONVERT(ControlPanel_saveAsDefaultsLabel, String);
extern READ_STR_PROP_CONVERT(ControlPanel_aboutLabel, String);
extern READ_STR_PROP_CONVERT(ControlPanel_instructionsLabel, String);
extern READ_STR_PROP_CONVERT(ControlPanel_loadSettingsLabel, String);
extern READ_STR_PROP_CONVERT(ControlPanel_saveSettingsLabel, String);
extern READ_STR_PROP_CONVERT(ControlPanel_fontSzTrName, String);
extern READ_STR_PROP_CONVERT(ControlPanel_encodingTrName, String);
extern READ_STR_PROP_CONVERT(ControlPanel_hybridResultTrName, String);
extern READ_STR_PROP_CONVERT(ControlPanel_structuralSimTrName, String);
extern READ_STR_PROP_CONVERT(ControlPanel_underGlyphCorrectnessTrName, String);
extern READ_STR_PROP_CONVERT(ControlPanel_glyphEdgeCorrectnessTrName, String);
extern READ_STR_PROP_CONVERT(ControlPanel_asideGlyphCorrectnessTrName, String);
extern READ_STR_PROP_CONVERT(ControlPanel_moreContrastTrName, String);
extern READ_STR_PROP_CONVERT(ControlPanel_gravityTrName, String);
extern READ_STR_PROP_CONVERT(ControlPanel_directionTrName, String);
extern READ_STR_PROP_CONVERT(ControlPanel_largerSymTrName, String);
extern READ_STR_PROP_CONVERT(ControlPanel_thresh4BlanksTrName, String);
extern READ_STR_PROP_CONVERT(ControlPanel_outWTrName, String);
extern READ_STR_PROP_CONVERT(ControlPanel_outHTrName, String);

extern READ_STR_PROP_CONVERT(Comparator_transpTrackName, String);

extern READ_STR_PROP_CONVERT(CmapInspect_pageTrackName, String);

extern const wstring ControlPanel_aboutText = std::move(str2wstr(replacePlaceholder(varConfig.read<string>("ControlPanel_aboutText"))));
extern READ_WSTR_PROP(ControlPanel_instructionsText);

#endif // ifndef UNIT_TESTING

extern const string Comparator_initial_title = replacePlaceholder(varConfig.read<string>("Comparator_initial_title"));
extern READ_STR_PROP(Comparator_statusBar);

extern READ_STR_PROP(Controller_PREFIX_GLYPH_PROGRESS);
extern READ_STR_PROP(Controller_PREFIX_TRANSFORMATION_PROGRESS);

extern READ_STR_PROP(copyrightText);

#if defined _DEBUG || defined UNIT_TESTING

extern READ_WSTR_PROP(MatchParams_HEADER);
extern READ_WSTR_PROP(BestMatch_HEADER) + MatchParams_HEADER;

#endif // _DEBUG || UNIT_TESTING

#undef READ_PROP
#undef READ_PROP_COND
#undef READ_BOOL_PROP
#undef READ_BOOL_PROP_COND
#undef READ_INT_PROP
#undef READ_INT_PROP_COND
#undef READ_UINT_PROP
#undef READ_DOUBLE_PROP
#undef READ_STR_PROP
#undef READ_WSTR_PROP
