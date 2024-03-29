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

#include "boxBlur.h"
#include "extBoxBlur.h"
#include "gaussBlur.h"
#include "misc.h"
#include "propsReader.h"
#include "structuralSimilarity.h"
#include "warnings.h"

#pragma warning(push, 0)

#include <string_view>

#include <boost/algorithm/string/replace.hpp>

#pragma warning(pop)

using namespace std;
using namespace cv;
using namespace boost::algorithm;

extern template class unordered_set<string>;

namespace pic2sym {

namespace {
/// Replaces all instances of a pattern in a text with a different string.
string replacePlaceholder(string text,              ///< initial text
                          string_view placeholder,  ///< pattern to be replaced
                          string_view replacement   ///< replacement string
                          ) noexcept {
  replace_all(text, placeholder, replacement);
  return text;
}

/**
Parser for reading various texts and constants customizing the runtime look
and behavior
@throw boost::property_tree::info_parser_error only in UnitTesting if the config
file cannot be found / parsed

Exception to be caught only in UnitTesting
*/
PropsReader& varConfigRef() noexcept(!UT) {
  static PropsReader varConfig("res/varConfig.txt");
  // If config file cannot be found / parsed => info_parser_error
  // UnitTesting propagates the exception up to here.
  // Otherwise the program terminates

  return varConfig;
}

}  // anonymous namespace

// Macros for reading the properties from 'varConfig.txt'
#define READ_PROP(prop, type, defaultValue, ...)                         \
  const type prop {                                                      \
    varConfigRef().read<type>(#prop, __VA_ARGS__).value_or(defaultValue) \
  }

#define READ_PROP_COND(prop, type, cond, defaultVal)               \
  const type prop {                                                \
    (cond) ? varConfigRef().read<type>(#prop).value_or(defaultVal) \
           : (defaultVal)                                          \
  }

#define READ_BOOL_PROP(prop) READ_PROP(prop, bool, false)

#define READ_STR_PROP(prop, ...) READ_PROP(prop, string, ""s, __VA_ARGS__)

#define READ_BOOL_PROP_COND(prop, cond) READ_PROP_COND(prop, bool, cond, false)

#define READ_INT_PROP(prop, ...) READ_PROP(prop, int, 0, __VA_ARGS__)

#define READ_UINT_PROP(prop, ...) READ_PROP(prop, unsigned, 0U, __VA_ARGS__)

#define READ_DOUBLE_PROP(prop, ...) READ_PROP(prop, double, 0., __VA_ARGS__)

/// Reads a string from the config file and converts it to a constant of type
/// stringLikeType, for instance cv::String
#define READ_STR_PROP_CONVERT(prop, stringLikeType) \
  const stringLikeType prop { varConfigRef().read<string>(#prop).value_or(""s) }

/// Reads a string from the config file and converts it to a constant of type
/// std::wstring; Allows appending the value with + after calling the macro
#define READ_WSTR_PROP(prop) \
  const wstring prop =       \
      str2wstr(varConfigRef().read<string>(#prop).value_or(""s))

static VALIDATOR(oddI, IsOdd, int);
static VALIDATOR(oddU, IsOdd, unsigned);

#pragma warning(disable : WARN_UNREFERENCED_FUNCTION_REMOVED)
static VALIDATOR(lessThan20i, IsLessThan, int, 20);
static VALIDATOR(lessThan600i, IsLessThan, int, 600, true);
static VALIDATOR(lessThan800i, IsLessThan, int, 800, true);
static VALIDATOR(lessThan1000i, IsLessThan, int, 1'000, true);
static VALIDATOR(atMost9U, IsLessThan, unsigned, 9U, true);
static VALIDATOR(atMost50U, IsLessThan, unsigned, 50U, true);
static VALIDATOR(atMost768U, IsLessThan, unsigned, 768U, true);
static VALIDATOR(lessThan1000U, IsLessThan, unsigned, 1'000U, true);
static VALIDATOR(atMost1024U, IsLessThan, unsigned, 1'024U, true);
static VALIDATOR(lessThan235D, IsLessThan, double, 235.);
static VALIDATOR(atMost50D, IsLessThan, double, 50., true);
static VALIDATOR(lessThan26D, IsLessThan, double, 26.);
static VALIDATOR(lessThan20D, IsLessThan, double, 20.);
static VALIDATOR(lessThan10D, IsLessThan, double, 10, true);
static VALIDATOR(lessThan5D, IsLessThan, double, 5., true);
static VALIDATOR(lessThan1D, IsLessThan, double, 1., true);
static VALIDATOR(lessThan0dot1, IsLessThan, double, 0.1);
static VALIDATOR(lessThan0dot4, IsLessThan, double, 0.4);
static VALIDATOR(lessThan0dot04, IsLessThan, double, 0.04);

static VALIDATOR(atLeast3i, IsGreaterThan, int, 3, true);
static VALIDATOR(atLeast5i, IsGreaterThan, int, 5, true);
static VALIDATOR(atLeast10i, IsGreaterThan, int, 10, true);
static VALIDATOR(atLeast480i, IsGreaterThan, int, 480, true);
static VALIDATOR(atLeast640i, IsGreaterThan, int, 640, true);
static VALIDATOR(atLeast1U, IsGreaterThan, unsigned, 1U, true);
static VALIDATOR(atLeast5U, IsGreaterThan, unsigned, 5U, true);
static VALIDATOR(atLeast3U, IsGreaterThan, unsigned, 3U, true);
static VALIDATOR(atLeast14D, IsGreaterThan, double, 14.);
static VALIDATOR(atLeast1dot6, IsGreaterThan, double, 1.6);
static VALIDATOR(atLeast1D, IsGreaterThan, double, 1., true);
static VALIDATOR(atLeast0dot8, IsGreaterThan, double, 0.8, true);
static VALIDATOR(atLeast0dot15, IsGreaterThan, double, 0.15, true);
static VALIDATOR(atLeast0dot01, IsGreaterThan, double, 0.01, true);
static VALIDATOR(atLeast0dot05, IsGreaterThan, double, 0.05, true);
static VALIDATOR(atLeast0dot001, IsGreaterThan, double, 0.001);
static VALIDATOR(positiveD, IsGreaterThan, double, 0.);
static VALIDATOR(nonNegativeD, IsGreaterThan, double, 0., true);

static VALIDATOR(availableClusterAlgs,
                 IsOneOf,
                 string,
                 {"None"s, "Partition"s, "TTSAS"s});
static VALIDATOR(availBlurAlgsForStrSim,
                 IsOneOf,
                 string,
                 {"box"s, "ext_box"s, "gaussian"s});
#pragma warning(default : WARN_UNREFERENCED_FUNCTION_REMOVED)

// Reading data before the main function - see static variable
// FileValidationResult from below.
// Initialization-order fiasco is avoided by using the extern constants from
// within functions/methods or by lazy evaluation, thus after the main function
extern READ_BOOL_PROP(Transform_BlurredPatches_InsteadOf_Originals);
extern READ_BOOL_PROP(ViewSymWeightsHistogram);

extern READ_BOOL_PROP(UsingOMP);  // global OpenMP switch
extern READ_BOOL_PROP_COND(ParallelizePixMapStatistics, UsingOMP);
extern READ_BOOL_PROP_COND(ParallelizeGridCreation, UsingOMP);
extern READ_BOOL_PROP_COND(ParallelizeGridPopulation, UsingOMP);
extern READ_BOOL_PROP_COND(PrepareMoreGlyphsAtOnce, UsingOMP);
extern READ_BOOL_PROP_COND(ParallelizeTr_PatchRowLoops, UsingOMP);

extern READ_UINT_PROP(Settings_MAX_THRESHOLD_FOR_BLANKS, atMost50U());

static const unsigned minFontSize() noexcept {
  static READ_UINT_PROP(Settings_MIN_FONT_SIZE, atLeast5U());
  return Settings_MIN_FONT_SIZE;
}
extern const unsigned Settings_MIN_FONT_SIZE{minFontSize()};

static const unsigned minHSyms() noexcept {
  static READ_UINT_PROP(Settings_MIN_H_SYMS, atLeast3U());
  return Settings_MIN_H_SYMS;
}
extern const unsigned Settings_MIN_H_SYMS{minHSyms()};

static const unsigned minVSyms() noexcept {
  static READ_UINT_PROP(Settings_MIN_V_SYMS, atLeast3U());
  return Settings_MIN_V_SYMS;
}
extern const unsigned Settings_MIN_V_SYMS{minVSyms()};

static VALIDATOR(moreThanMinFontSize,
                 IsGreaterThan,
                 unsigned,
                 minFontSize(),
                 true);
static VALIDATOR(moreThanMinHSyms, IsGreaterThan, unsigned, minHSyms(), true);
static VALIDATOR(moreThanMinVSyms, IsGreaterThan, unsigned, minVSyms(), true);

static const unsigned maxFontSize() noexcept {
  static READ_UINT_PROP(Settings_MAX_FONT_SIZE, atMost50U(),
                        moreThanMinFontSize());
  return Settings_MAX_FONT_SIZE;
}
extern const unsigned Settings_MAX_FONT_SIZE{maxFontSize()};

extern READ_UINT_PROP(Settings_MAX_H_SYMS, atMost1024U(), moreThanMinHSyms());
extern READ_UINT_PROP(Settings_MAX_V_SYMS, atMost768U(), moreThanMinVSyms());

static VALIDATOR(lessThanMaxFontSize,
                 IsLessThan,
                 unsigned,
                 maxFontSize(),
                 true);
extern READ_UINT_PROP(Settings_DEF_FONT_SIZE,
                      moreThanMinFontSize(),
                      lessThanMaxFontSize());

extern READ_BOOL_PROP(UseSkipMatchAspectsHeuristic);
extern READ_DOUBLE_PROP(EnableSkipAboveMatchRatio,
                        nonNegativeD(),
                        lessThan1D());

extern READ_DOUBLE_PROP(MinAverageClusterSize, atLeast1D());
extern READ_STR_PROP(ClusterAlgName, availableClusterAlgs());
extern READ_BOOL_PROP(FastDistSymToClusterComputation);
extern READ_DOUBLE_PROP(InvestigateClusterEvenForInferiorScoreFactor,
                        lessThan1D(),
                        atLeast0dot8());
extern READ_DOUBLE_PROP(MaxAvgProjErrForPartitionClustering,
                        positiveD(),
                        lessThan0dot1());
extern READ_DOUBLE_PROP(StillForegroundThreshold,
                        atLeast0dot001(),
                        lessThan0dot04());
extern READ_DOUBLE_PROP(ForegroundThresholdDelta, nonNegativeD(), atMost50D());
extern READ_DOUBLE_PROP(MaxRelMcOffsetForPartitionClustering,
                        atLeast0dot001(),
                        lessThan0dot1());
extern READ_DOUBLE_PROP(MaxRelMcOffsetForTTSAS_Clustering,
                        atLeast0dot001(),
                        lessThan0dot1());
extern READ_DOUBLE_PROP(MaxDiffAvgPixelValForPartitionClustering,
                        atLeast0dot001(),
                        lessThan0dot1());
extern READ_DOUBLE_PROP(MaxDiffAvgPixelValForTTSAS_Clustering,
                        atLeast0dot001(),
                        lessThan0dot1());
extern READ_BOOL_PROP(TTSAS_Accept1stClusterThatQualifiesAsParent);
extern READ_DOUBLE_PROP(TTSAS_Threshold_Member,
                        atLeast0dot01(),
                        lessThan0dot1());

extern READ_DOUBLE_PROP(PmsCont_SMALL_GLYPHS_PERCENT,
                        atLeast0dot05(),
                        lessThan0dot4());
extern READ_DOUBLE_PROP(SymData_computeFields_STILL_BG,
                        nonNegativeD(),
                        lessThan0dot04());
extern READ_DOUBLE_PROP(Transformer_run_THRESHOLD_CONTRAST_BLURRED,
                        nonNegativeD(),
                        lessThan20D());

extern READ_BOOL_PROP(BulkySymsFilterEnabled);
extern READ_BOOL_PROP(UnreadableSymsFilterEnabled);
extern READ_BOOL_PROP(SievesSymsFilterEnabled);
extern READ_BOOL_PROP(FilledRectanglesFilterEnabled);
extern READ_BOOL_PROP(GridBarsFilterEnabled);

extern READ_BOOL_PROP(PreserveRemovableSymbolsForExamination);

extern READ_DOUBLE_PROP(MinAreaRatioForUnreadableSymsBB,
                        atLeast0dot15(),
                        lessThan1D());

extern READ_DOUBLE_PROP(DirSmooth_DesiredBaseForCenterAndCornerMcs,
                        atLeast0dot8(),
                        lessThan1D());

extern READ_STR_PROP(StructuralSimilarity_BlurType, availBlurAlgsForStrSim());
extern READ_INT_PROP(StructuralSimilarity_RecommendedWindowSide,
                     oddI(),
                     atLeast3i(),
                     lessThan20i());
extern READ_DOUBLE_PROP(StructuralSimilarity_SIGMA,
                        atLeast0dot8(),
                        lessThan5D());
extern READ_DOUBLE_PROP(StructuralSimilarity_C1, atLeast1dot6(), lessThan26D());
extern READ_DOUBLE_PROP(StructuralSimilarity_C2, atLeast14D(), lessThan235D());

// Keep all cir fields before StructuralSimilarity::supportBlur
blur::BlurEngine::ConfInstRegistrator blur::BoxBlur::cir{
    "box", BoxBlur::configuredInstance()};
blur::BlurEngine::ConfInstRegistrator blur::ExtBoxBlur::cir{
    "ext_box", ExtBoxBlur::configuredInstance()};
blur::BlurEngine::ConfInstRegistrator blur::GaussBlur::cir{
    "gaussian", GaussBlur::configuredInstance()};

// Keep this after StructuralSimilarity_BlurType and below all defined cir
// static fields
const blur::IBlurEngine& match::StructuralSimilarity::supportBlur{
    blur::BlurEngine::byName(StructuralSimilarity_BlurType)};

static READ_INT_PROP(BlurWindowSize, oddI(), atLeast3i(), lessThan20i());
extern const Size BlurWinSize{BlurWindowSize, BlurWindowSize};
extern READ_DOUBLE_PROP(BlurStandardDeviation, atLeast0dot8(), lessThan5D());

extern READ_DOUBLE_PROP(Transform_ProgressReportsIncrement,
                        atLeast0dot01(),
                        lessThan1D());
extern READ_DOUBLE_PROP(SymbolsProcessing_ProgressReportsIncrement,
                        atLeast0dot01(),
                        lessThan1D());

extern READ_BOOL_PROP(PreselectionByTinySyms);
extern READ_UINT_PROP(ShortListLength, atLeast1U());
extern READ_DOUBLE_PROP(AdmitOnShortListEvenForInferiorScoreFactor,
                        atLeast0dot8(),
                        lessThan1D());
extern unsigned TinySymsSz() noexcept {
  static READ_UINT_PROP(TinySymsSize, oddU(), atLeast5U(), atMost9U());
  return TinySymsSize;
}

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
extern READ_STR_PROP_CONVERT(ControlPanel_symsBatchSzTrName, String);
extern READ_STR_PROP_CONVERT(ControlPanel_hybridResultTrName, String);
extern READ_STR_PROP_CONVERT(ControlPanel_structuralSimTrName, String);
extern READ_STR_PROP_CONVERT(ControlPanel_correlationTrName, String);
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

#ifndef UNIT_TESTING

extern READ_UINT_PROP(SymsBatch_trackMax, atLeast5U(), lessThan1000U());
extern READ_UINT_PROP(SymsBatch_defaultSz, atLeast1U(), atMost50U());

extern READ_INT_PROP(Comparator_trackMax, atLeast5i(), lessThan1000i());
extern READ_DOUBLE_PROP(Comparator_defaultTransparency,
                        nonNegativeD(),
                        lessThan1D());

static READ_INT_PROP(CmapInspect_width, atLeast640i(), lessThan800i());
static READ_INT_PROP(CmapInspect_height, atLeast480i(), lessThan600i());
extern const Size CmapInspect_pageSz(CmapInspect_width, CmapInspect_height);

extern READ_INT_PROP(ControlPanel_Converter_StructuralSim_maxSlider,
                     atLeast10i(),
                     lessThan1000i());
extern READ_DOUBLE_PROP(ControlPanel_Converter_StructuralSim_maxReal,
                        positiveD(),
                        lessThan10D());
extern READ_INT_PROP(ControlPanel_Converter_Correlation_maxSlider,
                     atLeast10i(),
                     lessThan1000i());
extern READ_DOUBLE_PROP(ControlPanel_Converter_Correlation_maxReal,
                        positiveD(),
                        lessThan1D());
extern READ_INT_PROP(ControlPanel_Converter_Correctness_maxSlider,
                     atLeast10i(),
                     lessThan1000i());
extern READ_DOUBLE_PROP(ControlPanel_Converter_Correctness_maxReal,
                        positiveD(),
                        lessThan10D());
extern READ_INT_PROP(ControlPanel_Converter_Contrast_maxSlider,
                     atLeast10i(),
                     lessThan1000i());
extern READ_DOUBLE_PROP(ControlPanel_Converter_Contrast_maxReal,
                        positiveD(),
                        lessThan10D());
extern READ_INT_PROP(ControlPanel_Converter_Gravity_maxSlider,
                     atLeast10i(),
                     lessThan1000i());
extern READ_DOUBLE_PROP(ControlPanel_Converter_Gravity_maxReal,
                        positiveD(),
                        lessThan10D());
extern READ_INT_PROP(ControlPanel_Converter_Direction_maxSlider,
                     atLeast10i(),
                     lessThan1000i());
extern READ_DOUBLE_PROP(ControlPanel_Converter_Direction_maxReal,
                        positiveD(),
                        lessThan10D());
extern READ_INT_PROP(ControlPanel_Converter_LargerSym_maxSlider,
                     atLeast10i(),
                     lessThan1000i());
extern READ_DOUBLE_PROP(ControlPanel_Converter_LargerSym_maxReal,
                        positiveD(),
                        lessThan10D());

extern READ_STR_PROP_CONVERT(Comparator_transpTrackName, String);

extern READ_STR_PROP_CONVERT(CmapInspect_pageTrackName, String);

extern const wstring ControlPanel_aboutText{str2wstr(replacePlaceholder(
    varConfigRef().read<string>("ControlPanel_aboutText"s).value_or(""s),
    "$(PIC2SYM_VERSION)",
    PIC2SYM_VERSION))};
extern READ_WSTR_PROP(ControlPanel_instructionsText);

#endif  // UNIT_TESTING not defined

extern const string Comparator_initial_title{replacePlaceholder(
    varConfigRef().read<string>("Comparator_initial_title"s).value_or(""s),
    "$(PIC2SYM_VERSION)",
    PIC2SYM_VERSION)};
extern READ_STR_PROP(Comparator_statusBar);

extern READ_STR_PROP(Controller_PREFIX_GLYPH_PROGRESS);
extern READ_STR_PROP(Controller_PREFIX_TRANSFORMATION_PROGRESS);

extern READ_STR_PROP(CopyrightText);

extern READ_STR_PROP(CannotLoadFontErrSuffix);

#if defined _DEBUG || defined UNIT_TESTING

extern READ_WSTR_PROP(MatchParams_HEADER);
extern READ_WSTR_PROP(BestMatch_HEADER) + MatchParams_HEADER;

#endif  // _DEBUG || UNIT_TESTING

#undef READ_PROP
#undef READ_PROP_COND
#undef READ_BOOL_PROP
#undef READ_BOOL_PROP_COND
#undef READ_INT_PROP
#undef READ_UINT_PROP
#undef READ_DOUBLE_PROP
#undef READ_STR_PROP
#undef READ_WSTR_PROP

/**
@throw domain_error for invalid configuration file

Exception to be reported only, not handled
*/
static bool validateConfig() {
  if (varConfigRef().anyError())
    REPORT_AND_THROW_CONST_MSG(domain_error, "Invalid configuration items!");
  return true;
}

// Ensures the configuration is valid and read before starting main function
static const bool FileValidationResult{validateConfig()};

}  // namespace pic2sym
