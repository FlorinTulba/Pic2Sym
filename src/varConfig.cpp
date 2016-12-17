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

#include "propsReader.h"
#include "gaussBlur.h"
#include "stackBlur.h"
#include "extBoxBlur.h"
#include "boxBlur.h"
#include "structuralSimilarity.h"
#include "misc.h"

#pragma warning ( push, 0 )

#include <set>

#include <boost/algorithm/string/replace.hpp>
#include <opencv2/core/core.hpp>

#pragma warning ( pop )

using namespace std;
using namespace cv;
using namespace boost::algorithm;

namespace {
	/// Replaces all instances of a pattern in a text with a different string.
	string replacePlaceholder(const string &text,	///< initial text
							  const string &placeholder = "$(PIC2SYM_VERSION)",	///< pattern to be replaced
							  const string &replacement = PIC2SYM_VERSION	///< replacement string
							  ) {
		string text_(text);
		replace_all(text_, placeholder, replacement);
		return text_; // NRVO
	}

	/// parser for reading various texts and constants customizing the runtime look and behavior
	PropsReader& varConfigRef() {
#pragma warning ( disable : WARN_THREAD_UNSAFE )
		static PropsReader varConfig("res/varConfig.txt");
#pragma warning ( default : WARN_THREAD_UNSAFE )
		return varConfig;
	}

	/// Base class for validators of the configuration items from 'res/varConfig.txt'
	template<class Type>
	struct ConfigItemValidator /*abstract*/ {
		/// Should throw an appropriate exception when itemVal is wrong for itemName
		virtual void examine(const string &itemName, const Type &itemVal) const = 0;
		virtual ~ConfigItemValidator() = 0 {}
	};

	/// Base class for IsOdd and IsEven from below
	template<class Type, typename = enable_if_t<is_integral<Type>::value>>
	struct IsOddOrEven : ConfigItemValidator<Type> {
		void examine(const string &itemName, const Type &itemVal) const override {
#define REQUIRE_PARITY(ParityType, Mod2Remainder) \
			if((Type)Mod2Remainder != (itemVal & (Type)1)) { \
				THROW_WITH_VAR_MSG("Configuration item '" + itemName + \
									"' needs to be " ParityType "!", invalid_argument); \
			}

			if(isOdd) { REQUIRE_PARITY("odd", 1); }
			else { REQUIRE_PARITY("even", 0); }

#undef REQUIRE_PARITY
		}

	protected:
		const bool isOdd;	///< true for IsOdd, false for IsEven

		IsOddOrEven(bool isOdd_) : isOdd(isOdd_) {}
		void operator=(const IsOddOrEven&) = delete;
	};

	/// Throws for non-odd configuration items
	template<class Type>
	struct IsOdd : IsOddOrEven<Type> {
		IsOdd() : IsOddOrEven(true) {}
		void operator=(const IsOdd&) = delete;
	};

	/// Throws for non-even configuration items
	template<class Type>
	struct IsEven : IsOddOrEven<Type> {
		IsEven() : IsOddOrEven(false) {}
		void operator=(const IsEven&) = delete;
	};

	/// Base class for IsLessThan and IsGreaterThan from below
	template<class Type, typename = enable_if_t<is_arithmetic<Type>::value>>
	struct IsLessOrGreaterThan : ConfigItemValidator<Type> {
		void examine(const string &itemName, const Type &itemVal) const override {
#define REQUIRE_REL(RelationType) \
			if(!(itemVal RelationType refVal)) { \
				THROW_WITH_VAR_MSG("Configuration item '" + itemName + \
					"' needs to be " #RelationType " " + to_string(refVal) + "!", out_of_range); \
			}

			if(isLess) {
				if(orEqual) { REQUIRE_REL(<=); }
				else { REQUIRE_REL(<); }

			} else { // isGreater
				if(orEqual) { REQUIRE_REL(>=); }
				else { REQUIRE_REL(>); }
			}

#undef REQUIRE_REL
		}

	protected:
		const Type refVal;	///< the value to compare against
		const bool isLess;	///< true for IsLessThan, false for IsGreaterThan
		const bool orEqual;	///< compare strictly or not

		IsLessOrGreaterThan(bool isLess_, const Type &refVal_, bool orEqual_ = false) :
			isLess(isLess_), refVal(refVal_), orEqual(orEqual_) {}
		void operator=(const IsLessOrGreaterThan&) = delete;
	};

	/// Throws for values > or >= than refVal_
	template<class Type>
	struct IsLessThan : IsLessOrGreaterThan<Type> {
		IsLessThan(const Type &refVal_, bool orEqual_ = false) : IsLessOrGreaterThan(true, refVal_, orEqual_) {}
		void operator=(const IsLessThan&) = delete;
	};

	/// Throws for values < or <= than refVal_
	template<class Type>
	struct IsGreaterThan : IsLessOrGreaterThan<Type> {
		IsGreaterThan(const Type &refVal_, bool orEqual_ = false) : IsLessOrGreaterThan(false, refVal_, orEqual_) {}
		void operator=(const IsGreaterThan&) = delete;
	};

	/// Checks that the provided value for a configuration item is within a given set of accepted values.
	template<class Type>
	struct IsOneOf : ConfigItemValidator<Type> {
		IsOneOf(const set<Type> &allowedSet_) : allowedSet(allowedSet_), allowedSetStr(setAsString(allowedSet_)) {
			if(allowedSet_.empty())
				THROW_WITH_CONST_MSG(__FUNCTION__ " should get a non-empty set of allowed values!", invalid_argument);
		}
		void operator=(const IsOneOf&) = delete;

		void examine(const string &itemName, const Type &itemVal) const override {
			if(allowedSet.cend() == allowedSet.find(itemVal))
				THROW_WITH_VAR_MSG("Configuration item '" + itemName +
					"' needs to be among these values: " + allowedSetStr + "!", invalid_argument);
		}

	protected:
		/// Helper to initialize allowedSetStr in initialization list
		static const string setAsString(const set<Type> &allowedSet_) {
			ostringstream oss;
			copy(CBOUNDS(allowedSet_), ostream_iterator<Type>(oss, ", "));
			oss<<"\b\b ";
			return oss.str();
		}

		const set<Type> allowedSet;	///< allowed set of values
		const string allowedSetStr;	///< same set in string format
	};

	/// Checks that itemName's value (itemValue) is approved by all validators, in which case it returns it.
	/// (Non-template-recursive solution of the function)
	template<class ItemType, class ... ValidatorTypes>
	const ItemType& checkItem(const string &itemName, const ItemType &itemVal,
							  const ValidatorTypes& ... validators) {
		// Declaring a dummy non-empty array that is ending in the validators' expansion
		int dummyArray[] {
			0, // a first element to ensure the array won't be empty

			// Series of 2 terms comma-expressions, each getting evaluated to 0,
			// but every one also calling 'examine' for a different validator
			(validators.examine(itemName, itemVal), 0)...
		};
		// Avoids warning about unused parameters / variable
		(void)itemName;
		(void)dummyArray;

		return itemVal;
	}
} // anonymous namespace

// Macros for reading the properties from 'varConfig.txt'
#define READ_PROP(prop, type, ...) \
	const type prop = checkItem(#prop, varConfigRef().read<type>(#prop), __VA_ARGS__);

#define READ_PROP_COND(prop, type, cond, defaultVal) \
	const type prop = (cond) ? varConfigRef().read<type>(#prop) : (defaultVal)

#define READ_BOOL_PROP(prop) \
	READ_PROP(prop, bool)

#define READ_BOOL_PROP_COND(prop, cond) \
	READ_PROP_COND(prop, bool, cond, false)

#define READ_INT_PROP(prop, ...) \
	READ_PROP(prop, int, __VA_ARGS__)

#define READ_INT_PROP_COND(prop, cond, defaultVal) \
	READ_PROP_COND(prop, int, cond, defaultVal)

#define READ_UINT_PROP(prop, ...) \
	READ_PROP(prop, unsigned, __VA_ARGS__)

#define READ_DOUBLE_PROP(prop, ...) \
	READ_PROP(prop, double, __VA_ARGS__)

#define READ_STR_PROP(prop, ...) \
	READ_PROP(prop, string, __VA_ARGS__)

#define READ_STR_PROP_CONVERT(prop, destStringType) \
	const destStringType prop = varConfigRef().read<string>(#prop)

#define READ_WSTR_PROP(prop) \
	const wstring prop = str2wstr(varConfigRef().read<string>(#prop))

// Limits for read data
#define VALIDATOR_NO_ARGS(Name, Kind, Type) \
	const Kind<Type>& Name() { \
		__pragma( warning( disable : WARN_THREAD_UNSAFE ) ) \
		static const Kind<Type> validator; \
		__pragma( warning( default : WARN_THREAD_UNSAFE ) ) \
		return validator; \
	} 

#define VALIDATOR(Name, Kind, Type, ...) \
	const Kind<Type>& Name() { \
		__pragma( warning( disable : WARN_THREAD_UNSAFE ) ) \
		static const Kind<Type> validator(__VA_ARGS__); \
		__pragma( warning( default : WARN_THREAD_UNSAFE ) ) \
		return validator; \
	} 

static VALIDATOR_NO_ARGS(oddI, IsOdd, int);
static VALIDATOR_NO_ARGS(oddU, IsOdd, unsigned);

static VALIDATOR(lessThan20i,	IsLessThan, int, 20);
static VALIDATOR(lessThan600i,	IsLessThan, int, 600, true);
static VALIDATOR(lessThan800i,	IsLessThan, int, 800, true);
static VALIDATOR(lessThan1000i, IsLessThan, int, 1000, true);
static VALIDATOR(atMost9U,		IsLessThan, unsigned, 9U, true);
static VALIDATOR(atMost50U,		IsLessThan, unsigned, 50U, true);
static VALIDATOR(atMost768U,	IsLessThan, unsigned, 768U, true);
static VALIDATOR(lessThan1000U, IsLessThan, unsigned, 1000U, true);
static VALIDATOR(atMost1024U,	IsLessThan, unsigned, 1024U, true);
static VALIDATOR(lessThan235D,	IsLessThan, double, 235.);
static VALIDATOR(atMost50D,		IsLessThan, double, 50., true);
static VALIDATOR(lessThan26D,	IsLessThan, double, 26.);
static VALIDATOR(lessThan20D,	IsLessThan, double, 20.);
static VALIDATOR(lessThan10D,	IsLessThan, double, 10, true);
static VALIDATOR(lessThan5D,	IsLessThan, double, 5., true);
static VALIDATOR(lessThan1D,	IsLessThan, double, 1., true);
static VALIDATOR(lessThan0dot1, IsLessThan, double, 0.1);
static VALIDATOR(lessThan0dot4, IsLessThan, double, 0.4);
static VALIDATOR(lessThan0dot04,IsLessThan, double, 0.04);

static VALIDATOR(atLeast3i,		IsGreaterThan, int, 3, true);
static VALIDATOR(atLeast5i,		IsGreaterThan, int, 5, true);
static VALIDATOR(atLeast10i,	IsGreaterThan, int, 10, true);
static VALIDATOR(atLeast480i,	IsGreaterThan, int, 480, true);
static VALIDATOR(atLeast640i,	IsGreaterThan, int, 640, true);
static VALIDATOR(atLeast1U,		IsGreaterThan, unsigned, 1U, true);
static VALIDATOR(atLeast5U,		IsGreaterThan, unsigned, 5U, true);
static VALIDATOR(atLeast3U,		IsGreaterThan, unsigned, 3U, true);
static VALIDATOR(atLeast14D,	IsGreaterThan, double, 14.);
static VALIDATOR(atLeast1dot6,	IsGreaterThan, double, 1.6);
static VALIDATOR(atLeast1D,		IsGreaterThan, double, 1., true);
static VALIDATOR(atLeast0dot8,	IsGreaterThan, double, 0.8, true);
static VALIDATOR(atLeast0dot15,	IsGreaterThan, double, 0.15, true);
static VALIDATOR(atLeast0dot01, IsGreaterThan, double, 0.01, true);
static VALIDATOR(atLeast0dot05, IsGreaterThan, double, 0.05, true);
static VALIDATOR(atLeast0dot001,IsGreaterThan, double, 0.001);
static VALIDATOR(positiveD,		IsGreaterThan, double, 0.);
static VALIDATOR(nonNegativeD,	IsGreaterThan, double, 0., true);

static VALIDATOR(availableClusterAlgs,	IsOneOf, string, { "None", "Partition", "TTSAS" });
static VALIDATOR(availBlurAlgsForStrSim,IsOneOf, string, { "box", "ext_box", "stack", "gaussian" });

// Reading data
extern READ_BOOL_PROP(Transform_BlurredPatches_InsteadOf_Originals);
extern READ_BOOL_PROP(ViewSymWeightsHistogram);

extern READ_BOOL_PROP(UsingOMP); // global OpenMP switch
extern READ_BOOL_PROP_COND(ParallelizePixMapStatistics, UsingOMP);
extern READ_BOOL_PROP_COND(ParallelizeGridCreation, UsingOMP);
extern READ_BOOL_PROP_COND(ParallelizeGridPopulation, UsingOMP);
extern READ_BOOL_PROP_COND(PrepareMoreGlyphsAtOnce, UsingOMP);
extern READ_BOOL_PROP_COND(ParallelizeTr_PatchRowLoops, UsingOMP);

extern READ_UINT_PROP(Settings_MAX_THRESHOLD_FOR_BLANKS, atMost50U());

static const unsigned minFontSize() {
#pragma warning ( disable : WARN_THREAD_UNSAFE )
	static READ_UINT_PROP(Settings_MIN_FONT_SIZE, atLeast5U());
#pragma warning ( default : WARN_THREAD_UNSAFE )

	return Settings_MIN_FONT_SIZE;
}
extern const unsigned Settings_MIN_FONT_SIZE = minFontSize();

static const unsigned minHSyms() {
#pragma warning ( disable : WARN_THREAD_UNSAFE )
	static READ_UINT_PROP(Settings_MIN_H_SYMS, atLeast3U());
#pragma warning ( default : WARN_THREAD_UNSAFE )

	return Settings_MIN_H_SYMS;
}
extern const unsigned Settings_MIN_H_SYMS = minHSyms();

static const unsigned minVSyms() {
#pragma warning ( disable : WARN_THREAD_UNSAFE )
	static READ_UINT_PROP(Settings_MIN_V_SYMS, atLeast3U());
#pragma warning ( default : WARN_THREAD_UNSAFE )

	return Settings_MIN_V_SYMS;
}
extern const unsigned Settings_MIN_V_SYMS = minVSyms();

static VALIDATOR(moreThanMinFontSize,	IsGreaterThan, unsigned, minFontSize(), true);
static VALIDATOR(moreThanMinHSyms,		IsGreaterThan, unsigned, minHSyms(), true);
static VALIDATOR(moreThanMinVSyms,		IsGreaterThan, unsigned, minVSyms(), true);

static const unsigned maxFontSize() {
#pragma warning ( disable : WARN_THREAD_UNSAFE )
	static READ_UINT_PROP(Settings_MAX_FONT_SIZE, atMost50U(), moreThanMinFontSize());
#pragma warning ( default : WARN_THREAD_UNSAFE )

	return Settings_MAX_FONT_SIZE;
}
extern const unsigned Settings_MAX_FONT_SIZE = maxFontSize();

extern READ_UINT_PROP(Settings_MAX_H_SYMS, atMost1024U(), moreThanMinHSyms());
extern READ_UINT_PROP(Settings_MAX_V_SYMS, atMost768U(), moreThanMinVSyms());

static VALIDATOR(lessThanMaxFontSize,	IsLessThan, unsigned, maxFontSize(), true);
extern READ_UINT_PROP(Settings_DEF_FONT_SIZE, moreThanMinFontSize(), lessThanMaxFontSize());

extern READ_BOOL_PROP(UseSkipMatchAspectsHeuristic);
extern READ_DOUBLE_PROP(EnableSkipAboveMatchRatio, nonNegativeD(), lessThan1D());

extern READ_DOUBLE_PROP(MinAverageClusterSize, atLeast1D());
extern READ_STR_PROP(ClusterAlgName, availableClusterAlgs());
extern READ_BOOL_PROP(FastDistSymToClusterComputation);
extern READ_DOUBLE_PROP(InvestigateClusterEvenForInferiorScoreFactor, lessThan1D(), atLeast0dot8());
extern READ_DOUBLE_PROP(MaxAvgProjErrForPartitionClustering, positiveD(), lessThan0dot1());
extern READ_DOUBLE_PROP(StillForegroundThreshold, atLeast0dot001(), lessThan0dot04());
extern READ_DOUBLE_PROP(ForegroundThresholdDelta, nonNegativeD(), atMost50D());
extern READ_DOUBLE_PROP(MaxRelMcOffsetForPartitionClustering, atLeast0dot001(), lessThan0dot1());
extern READ_DOUBLE_PROP(MaxRelMcOffsetForTTSAS_Clustering, atLeast0dot001(), lessThan0dot1());
extern READ_DOUBLE_PROP(MaxDiffAvgPixelValForPartitionClustering, atLeast0dot001(), lessThan0dot1());
extern READ_DOUBLE_PROP(MaxDiffAvgPixelValForTTSAS_Clustering, atLeast0dot001(), lessThan0dot1());
extern READ_BOOL_PROP(TTSAS_Accept1stClusterThatQualifiesAsParent);
extern READ_DOUBLE_PROP(TTSAS_Threshold_Member, atLeast0dot01(), lessThan0dot1());

extern READ_DOUBLE_PROP(PmsCont_SMALL_GLYPHS_PERCENT, atLeast0dot05(), lessThan0dot4());
extern READ_DOUBLE_PROP(SymData_computeFields_STILL_BG, nonNegativeD(), lessThan0dot04());
extern READ_DOUBLE_PROP(Transformer_run_THRESHOLD_CONTRAST_BLURRED, nonNegativeD(), lessThan20D());

extern READ_BOOL_PROP(BulkySymsFilterEnabled);
extern READ_BOOL_PROP(UnreadableSymsFilterEnabled);
extern READ_BOOL_PROP(SievesSymsFilterEnabled);
extern READ_BOOL_PROP(FilledRectanglesFilterEnabled);
extern READ_BOOL_PROP(GridBarsFilterEnabled);

extern READ_BOOL_PROP(PreserveRemovableSymbolsForExamination);

extern READ_DOUBLE_PROP(MinAreaRatioForUnreadableSymsBB, atLeast0dot15(), lessThan1D());

extern READ_DOUBLE_PROP(DirSmooth_DesiredBaseForCenterAndCornerMcs, atLeast0dot8(), lessThan1D());

extern READ_STR_PROP(StructuralSimilarity_BlurType, availBlurAlgsForStrSim());
extern READ_INT_PROP(StructuralSimilarity_RecommendedWindowSide, oddI(), atLeast3i(), lessThan20i());
extern READ_DOUBLE_PROP(StructuralSimilarity_SIGMA, atLeast0dot8(), lessThan5D());
extern READ_DOUBLE_PROP(StructuralSimilarity_C1, atLeast1dot6(), lessThan26D());
extern READ_DOUBLE_PROP(StructuralSimilarity_C2, atLeast14D(), lessThan235D());

// Keep all cir fields before StructuralSimilarity::supportBlur
BlurEngine::ConfInstRegistrator BoxBlur::cir("box", BoxBlur::configuredInstance()); 
BlurEngine::ConfInstRegistrator ExtBoxBlur::cir("ext_box", ExtBoxBlur::configuredInstance());
BlurEngine::ConfInstRegistrator StackBlur::cir("stack", StackBlur::configuredInstance());
BlurEngine::ConfInstRegistrator GaussBlur::cir("gaussian", GaussBlur::configuredInstance());

// Keep this after StructuralSimilarity_BlurType and below all defined cir static fields
const BlurEngine& StructuralSimilarity::supportBlur = BlurEngine::byName(StructuralSimilarity_BlurType);

static READ_INT_PROP(BlurWindowSize, oddI(), atLeast3i(), lessThan20i());
extern const Size BlurWinSize(BlurWindowSize, BlurWindowSize);
extern READ_DOUBLE_PROP(BlurStandardDeviation, atLeast0dot8(), lessThan5D());

extern READ_DOUBLE_PROP(Transform_ProgressReportsIncrement, atLeast0dot01(), lessThan1D());
extern READ_DOUBLE_PROP(SymbolsProcessing_ProgressReportsIncrement, atLeast0dot01(), lessThan1D());

extern READ_BOOL_PROP(PreselectionByTinySyms);
extern READ_UINT_PROP(ShortListLength, atLeast1U());
extern READ_DOUBLE_PROP(AdmitOnShortListEvenForInferiorScoreFactor, atLeast0dot8(), lessThan1D());
extern unsigned TinySymsSz() {
#pragma warning ( disable : WARN_THREAD_UNSAFE )
	static READ_UINT_PROP(TinySymsSize, oddU(), atLeast5U(), atMost9U());
#pragma warning ( default : WARN_THREAD_UNSAFE )

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
extern READ_DOUBLE_PROP(Comparator_defaultTransparency, nonNegativeD(), lessThan1D());

static READ_INT_PROP(CmapInspect_width, atLeast640i(), lessThan800i());
static READ_INT_PROP(CmapInspect_height, atLeast480i(), lessThan600i());
extern const Size CmapInspect_pageSz(CmapInspect_width, CmapInspect_height);

extern READ_INT_PROP(ControlPanel_Converter_StructuralSim_maxSlider, atLeast10i(), lessThan1000i());
extern READ_DOUBLE_PROP(ControlPanel_Converter_StructuralSim_maxReal, positiveD(), lessThan10D());
extern READ_INT_PROP(ControlPanel_Converter_Correctness_maxSlider, atLeast10i(), lessThan1000i());
extern READ_DOUBLE_PROP(ControlPanel_Converter_Correctness_maxReal, positiveD(), lessThan10D());
extern READ_INT_PROP(ControlPanel_Converter_Contrast_maxSlider, atLeast10i(), lessThan1000i());
extern READ_DOUBLE_PROP(ControlPanel_Converter_Contrast_maxReal, positiveD(), lessThan10D());
extern READ_INT_PROP(ControlPanel_Converter_Gravity_maxSlider, atLeast10i(), lessThan1000i());
extern READ_DOUBLE_PROP(ControlPanel_Converter_Gravity_maxReal, positiveD(), lessThan10D());
extern READ_INT_PROP(ControlPanel_Converter_Direction_maxSlider, atLeast10i(), lessThan1000i());
extern READ_DOUBLE_PROP(ControlPanel_Converter_Direction_maxReal, positiveD(), lessThan10D());
extern READ_INT_PROP(ControlPanel_Converter_LargerSym_maxSlider, atLeast10i(), lessThan1000i());
extern READ_DOUBLE_PROP(ControlPanel_Converter_LargerSym_maxReal, positiveD(), lessThan10D());

extern READ_STR_PROP_CONVERT(Comparator_transpTrackName, String);

extern READ_STR_PROP_CONVERT(CmapInspect_pageTrackName, String);

extern const wstring ControlPanel_aboutText = str2wstr(replacePlaceholder(varConfigRef().read<string>("ControlPanel_aboutText")));
extern READ_WSTR_PROP(ControlPanel_instructionsText);

#endif // ifndef UNIT_TESTING

extern const string Comparator_initial_title = replacePlaceholder(varConfigRef().read<string>("Comparator_initial_title"));
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

#undef VALIDATOR_NO_ARGS
#undef VALIDATOR	
