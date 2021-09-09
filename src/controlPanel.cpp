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

#ifndef UNIT_TESTING

#include "controlPanel.h"

#include "controlPanelActions.h"
#include "dlgs.h"
#include "imgSettingsBase.h"
#include "matchSettingsBase.h"
#include "misc.h"
#include "settingsBase.h"
#include "sliderConversion.h"
#include "symSettingsBase.h"

#pragma warning(push, 0)

#include <thread>
#include <tuple>

#include <gsl/gsl>

#include <opencv2/highgui/highgui.hpp>

#pragma warning(pop)

using namespace std;
using namespace gsl;
using namespace cv;

namespace pic2sym {

extern const int ControlPanel_Converter_StructuralSim_maxSlider;
extern const double ControlPanel_Converter_StructuralSim_maxReal;
extern const int ControlPanel_Converter_Correlation_maxSlider;
extern const double ControlPanel_Converter_Correlation_maxReal;
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
extern const String ControlPanel_correlationTrName;
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
extern const unsigned Settings_MAX_THRESHOLD_FOR_BLANKS;
extern const unsigned Settings_MAX_H_SYMS;
extern const unsigned Settings_MAX_V_SYMS;
extern const unsigned Settings_MAX_FONT_SIZE;
extern const unsigned SymsBatch_trackMax;
extern const string CannotLoadFontErrSuffix;

namespace ui {

const unordered_map<const String*,
                    const std::unique_ptr<const SliderConverter>>&
ControlPanel::slidersConverters() noexcept {
  static unordered_map<const String*,
                       const std::unique_ptr<const SliderConverter>>
      result;
  static bool initialized{false};

  if (!initialized) {
    result.emplace(&ControlPanel_structuralSimTrName,
                   std::make_unique<const ProportionalSliderValue>(
                       std::make_unique<const ProportionalSliderValue::Params>(
                           ControlPanel_Converter_StructuralSim_maxSlider,
                           ControlPanel_Converter_StructuralSim_maxReal)));
    result.emplace(&ControlPanel_correlationTrName,
                   std::make_unique<const ProportionalSliderValue>(
                       std::make_unique<const ProportionalSliderValue::Params>(
                           ControlPanel_Converter_Correlation_maxSlider,
                           ControlPanel_Converter_Correlation_maxReal)));
    result.emplace(&ControlPanel_underGlyphCorrectnessTrName,
                   std::make_unique<const ProportionalSliderValue>(
                       std::make_unique<const ProportionalSliderValue::Params>(
                           ControlPanel_Converter_Correctness_maxSlider,
                           ControlPanel_Converter_Correctness_maxReal)));
    result.emplace(&ControlPanel_glyphEdgeCorrectnessTrName,
                   std::make_unique<const ProportionalSliderValue>(
                       std::make_unique<const ProportionalSliderValue::Params>(
                           ControlPanel_Converter_Correctness_maxSlider,
                           ControlPanel_Converter_Correctness_maxReal)));
    result.emplace(&ControlPanel_asideGlyphCorrectnessTrName,
                   std::make_unique<const ProportionalSliderValue>(
                       std::make_unique<const ProportionalSliderValue::Params>(
                           ControlPanel_Converter_Correctness_maxSlider,
                           ControlPanel_Converter_Correctness_maxReal)));
    result.emplace(&ControlPanel_moreContrastTrName,
                   std::make_unique<const ProportionalSliderValue>(
                       std::make_unique<const ProportionalSliderValue::Params>(
                           ControlPanel_Converter_Contrast_maxSlider,
                           ControlPanel_Converter_Contrast_maxReal)));
    result.emplace(&ControlPanel_gravityTrName,
                   std::make_unique<const ProportionalSliderValue>(
                       std::make_unique<const ProportionalSliderValue::Params>(
                           ControlPanel_Converter_Gravity_maxSlider,
                           ControlPanel_Converter_Gravity_maxReal)));
    result.emplace(&ControlPanel_directionTrName,
                   std::make_unique<const ProportionalSliderValue>(
                       std::make_unique<const ProportionalSliderValue::Params>(
                           ControlPanel_Converter_Direction_maxSlider,
                           ControlPanel_Converter_Direction_maxReal)));
    result.emplace(&ControlPanel_largerSymTrName,
                   std::make_unique<const ProportionalSliderValue>(
                       std::make_unique<const ProportionalSliderValue::Params>(
                           ControlPanel_Converter_LargerSym_maxSlider,
                           ControlPanel_Converter_LargerSym_maxReal)));

    initialized = true;
  }

  return result;
}

void ControlPanel::updateMatchSettings(
    const p2s::cfg::IMatchSettings& ms) noexcept {
  int newVal{ms.isHybridResult()};
  while (hybridResult != newVal)
    setTrackbarPos(*(pLuckySliderName = &ControlPanel_hybridResultTrName),
                   winName, newVal);

  newVal = slidersConverters()
               .at(&ControlPanel_structuralSimTrName)
               ->toSlider(ms.get_kSsim());
  while (structuralSim != newVal)
    setTrackbarPos(*(pLuckySliderName = &ControlPanel_structuralSimTrName),
                   winName, newVal);

  newVal = slidersConverters()
               .at(&ControlPanel_correlationTrName)
               ->toSlider(ms.get_kCorrel());
  while (correlationCorrectness != newVal)
    setTrackbarPos(*(pLuckySliderName = &ControlPanel_correlationTrName),
                   winName, newVal);

  newVal = slidersConverters()
               .at(&ControlPanel_underGlyphCorrectnessTrName)
               ->toSlider(ms.get_kSdevFg());
  while (underGlyphCorrectness != newVal)
    setTrackbarPos(
        *(pLuckySliderName = &ControlPanel_underGlyphCorrectnessTrName),
        winName, newVal);

  newVal = slidersConverters()
               .at(&ControlPanel_glyphEdgeCorrectnessTrName)
               ->toSlider(ms.get_kSdevEdge());
  while (glyphEdgeCorrectness != newVal)
    setTrackbarPos(
        *(pLuckySliderName = &ControlPanel_glyphEdgeCorrectnessTrName), winName,
        newVal);

  newVal = slidersConverters()
               .at(&ControlPanel_asideGlyphCorrectnessTrName)
               ->toSlider(ms.get_kSdevBg());
  while (asideGlyphCorrectness != newVal)
    setTrackbarPos(
        *(pLuckySliderName = &ControlPanel_asideGlyphCorrectnessTrName),
        winName, newVal);

  newVal = slidersConverters()
               .at(&ControlPanel_moreContrastTrName)
               ->toSlider(ms.get_kContrast());
  while (moreContrast != newVal)
    setTrackbarPos(*(pLuckySliderName = &ControlPanel_moreContrastTrName),
                   winName, newVal);

  newVal = slidersConverters()
               .at(&ControlPanel_gravityTrName)
               ->toSlider(ms.get_kMCsOffset());
  while (gravity != newVal)
    setTrackbarPos(*(pLuckySliderName = &ControlPanel_gravityTrName), winName,
                   newVal);

  newVal = slidersConverters()
               .at(&ControlPanel_directionTrName)
               ->toSlider(ms.get_kCosAngleMCs());
  while (direction != newVal)
    setTrackbarPos(*(pLuckySliderName = &ControlPanel_directionTrName), winName,
                   newVal);

  newVal = slidersConverters()
               .at(&ControlPanel_largerSymTrName)
               ->toSlider(ms.get_kSymDensity());
  while (largerSym != newVal)
    setTrackbarPos(*(pLuckySliderName = &ControlPanel_largerSymTrName), winName,
                   newVal);

  newVal = (int)ms.getBlankThreshold();
  while (thresh4Blanks != newVal)
    setTrackbarPos(*(pLuckySliderName = &ControlPanel_thresh4BlanksTrName),
                   winName, newVal);

  pLuckySliderName = nullptr;
}

void ControlPanel::updateImgSettings(
    const p2s::cfg::IfImgSettings& is) noexcept {
  int newVal{(int)is.getMaxHSyms()};
  while (maxHSyms != newVal)
    setTrackbarPos(*(pLuckySliderName = &ControlPanel_outWTrName), winName,
                   newVal);

  newVal = (int)is.getMaxVSyms();
  while (maxVSyms != newVal)
    setTrackbarPos(*(pLuckySliderName = &ControlPanel_outHTrName), winName,
                   newVal);

  pLuckySliderName = nullptr;
}

void ControlPanel::updateSymSettings(unsigned encIdx,
                                     unsigned fontSz_) noexcept {
  while (encoding != (int)encIdx)
    setTrackbarPos(*(pLuckySliderName = &ControlPanel_encodingTrName), winName,
                   (int)encIdx);

  while (fontSz != (int)fontSz_)
    setTrackbarPos(*(pLuckySliderName = &ControlPanel_fontSzTrName), winName,
                   (int)fontSz_);

  pLuckySliderName = nullptr;
}

void ControlPanel::updateEncodingsCount(unsigned uniqueEncodings) noexcept {
  updatingEncMax = true;
  setTrackbarMax(ControlPanel_encodingTrName, winName,
                 max(1, int(uniqueEncodings - 1U)));

  // Sequence from below is required to really update the trackbar max & pos
  // The controller should prevent them to trigger Controller::newFontEncoding
  setTrackbarPos(*(pLuckySliderName = &ControlPanel_encodingTrName), winName,
                 1);
  updatingEncMax = false;
  setTrackbarPos(ControlPanel_encodingTrName, winName, 0);
  pLuckySliderName = nullptr;
}

void ControlPanel::restoreSliderValue(const String& trName,
                                      string_view errText) noexcept {
  // Determine previous value
  int prevVal{};
  if (&trName == &ControlPanel_outWTrName) {
    prevVal = (int)cfg->getIS().getMaxHSyms();
  } else if (&trName == &ControlPanel_outHTrName) {
    prevVal = (int)cfg->getIS().getMaxVSyms();
  } else if (&trName == &ControlPanel_encodingTrName) {
    prevVal = (int)performer->getFontEncodingIdx();
  } else if (&trName == &ControlPanel_fontSzTrName) {
    prevVal = (int)cfg->getSS().getFontSz();
  } else if (&trName == &ControlPanel_symsBatchSzTrName) {
    prevVal = symsBatchSz;  // no change needed for Symbols Batch Size!
  } else if (&trName == &ControlPanel_hybridResultTrName) {
    prevVal = cfg->getMS().isHybridResult() ? 1 : 0;
  } else if (&trName == &ControlPanel_structuralSimTrName) {
    prevVal = slidersConverters()
                  .at(&ControlPanel_structuralSimTrName)
                  ->toSlider(cfg->getMS().get_kSsim());
  } else if (&trName == &ControlPanel_correlationTrName) {
    prevVal = slidersConverters()
                  .at(&ControlPanel_correlationTrName)
                  ->toSlider(cfg->getMS().get_kCorrel());
  } else if (&trName == &ControlPanel_underGlyphCorrectnessTrName) {
    prevVal = slidersConverters()
                  .at(&ControlPanel_underGlyphCorrectnessTrName)
                  ->toSlider(cfg->getMS().get_kSdevFg());
  } else if (&trName == &ControlPanel_glyphEdgeCorrectnessTrName) {
    prevVal = slidersConverters()
                  .at(&ControlPanel_glyphEdgeCorrectnessTrName)
                  ->toSlider(cfg->getMS().get_kSdevEdge());
  } else if (&trName == &ControlPanel_asideGlyphCorrectnessTrName) {
    prevVal = slidersConverters()
                  .at(&ControlPanel_asideGlyphCorrectnessTrName)
                  ->toSlider(cfg->getMS().get_kSdevBg());
  } else if (&trName == &ControlPanel_moreContrastTrName) {
    prevVal = slidersConverters()
                  .at(&ControlPanel_moreContrastTrName)
                  ->toSlider(cfg->getMS().get_kContrast());
  } else if (&trName == &ControlPanel_gravityTrName) {
    prevVal = slidersConverters()
                  .at(&ControlPanel_gravityTrName)
                  ->toSlider(cfg->getMS().get_kMCsOffset());
  } else if (&trName == &ControlPanel_directionTrName) {
    prevVal = slidersConverters()
                  .at(&ControlPanel_directionTrName)
                  ->toSlider(cfg->getMS().get_kCosAngleMCs());
  } else if (&trName == &ControlPanel_largerSymTrName) {
    prevVal = slidersConverters()
                  .at(&ControlPanel_largerSymTrName)
                  ->toSlider(cfg->getMS().get_kSymDensity());
  } else if (&trName == &ControlPanel_thresh4BlanksTrName) {
    prevVal = (int)cfg->getMS().getBlankThreshold();
  } else {
    cerr << "Either the tracker name: `" << trName
         << "` is invalid or code for it must be added within "
         << HERE.function_name() << endl;
    return;
  }

  // Deals with the case when the value was already restored/not modified at all
  if (getTrackbarPos(trName, winName) == prevVal)
    return;

  slidersRestoringValue.insert(trName);

  sliderErrorsHandler =
      jthread{[&](const String sliderName, int previousVal,
                  string&& errorText) noexcept {
                errMsg(errorText);

                // Set the previous value
                while (getTrackbarPos(sliderName, winName) != previousVal) {
                  setTrackbarPos(sliderName, winName, previousVal);
                  this_thread::yield();
                }

                slidersRestoringValue.erase(sliderName);
              },
              trName, prevVal, string{errText}};
}

ControlPanel::ControlPanel(IControlPanelActions& performer_,
                           const p2s::cfg::ISettings& cfg_) noexcept
    : performer(&performer_),
      cfg(&cfg_),
      maxHSyms((int)cfg_.getIS().getMaxHSyms()),
      maxVSyms((int)cfg_.getIS().getMaxVSyms()),
      encoding(0),
      fontSz((int)cfg_.getSS().getFontSz()),
      symsBatchSz((int)SymsBatch_defaultSz),
      hybridResult(cfg_.getMS().isHybridResult() ? 1 : 0),
      structuralSim(slidersConverters()
                        .at(&ControlPanel_structuralSimTrName)
                        ->toSlider(cfg_.getMS().get_kSsim())),
      correlationCorrectness(slidersConverters()
                                 .at(&ControlPanel_correlationTrName)
                                 ->toSlider(cfg_.getMS().get_kCorrel())),
      underGlyphCorrectness(slidersConverters()
                                .at(&ControlPanel_underGlyphCorrectnessTrName)
                                ->toSlider(cfg_.getMS().get_kSdevFg())),
      glyphEdgeCorrectness(slidersConverters()
                               .at(&ControlPanel_glyphEdgeCorrectnessTrName)
                               ->toSlider(cfg_.getMS().get_kSdevEdge())),
      asideGlyphCorrectness(slidersConverters()
                                .at(&ControlPanel_asideGlyphCorrectnessTrName)
                                ->toSlider(cfg_.getMS().get_kSdevBg())),
      moreContrast(slidersConverters()
                       .at(&ControlPanel_moreContrastTrName)
                       ->toSlider(cfg_.getMS().get_kContrast())),
      gravity(slidersConverters()
                  .at(&ControlPanel_gravityTrName)
                  ->toSlider(cfg_.getMS().get_kMCsOffset())),
      direction(slidersConverters()
                    .at(&ControlPanel_directionTrName)
                    ->toSlider(cfg_.getMS().get_kCosAngleMCs())),
      largerSym(slidersConverters()
                    .at(&ControlPanel_largerSymTrName)
                    ->toSlider(cfg_.getMS().get_kSymDensity())),
      thresh4Blanks((int)cfg_.getMS().getBlankThreshold()) {
  createButton(
      ControlPanel_selectImgLabel,
      [](int, void* userdata) noexcept {
        not_null<IControlPanelActions*> pActions =
            static_cast<IControlPanelActions*>(userdata);
        static ImgSelector is;

        if (is.promptForUserChoice())
          ignore = pActions->newImage(is.selection());
      },
      static_cast<void*>(&*performer));
  createButton(
      ControlPanel_transformImgLabel,
      [](int, void* userdata) noexcept {
        not_null<IControlPanelActions*> pActions =
            static_cast<IControlPanelActions*>(userdata);
        ignore = pActions->performTransformation();
      },
      static_cast<void*>(&*performer));

  createTrackbar(
      ControlPanel_outWTrName, winName, &maxHSyms, (int)Settings_MAX_H_SYMS,
      [](int val, void* userdata) noexcept {
        not_null<IControlPanelActions*> pActions =
            static_cast<IControlPanelActions*>(userdata);
        pActions->newHmaxSyms(val);
      },
      static_cast<void*>(&*performer));
  createTrackbar(
      ControlPanel_outHTrName, winName, &maxVSyms, (int)Settings_MAX_V_SYMS,
      [](int val, void* userdata) noexcept {
        not_null<IControlPanelActions*> pActions =
            static_cast<IControlPanelActions*>(userdata);
        pActions->newVmaxSyms(val);
      },
      static_cast<void*>(&*performer));

  createButton(
      ControlPanel_selectFontLabel,
      [](int, void* userdata) {
        not_null<IControlPanelActions*> pActions =
            static_cast<IControlPanelActions*>(userdata);

        static SelectFont sf;

        if (sf.promptForUserChoice()) {
          const string& selection = sf.selection();
          if (!selection.empty())
            pActions->newFontFamily(selection, sf.size());

          else {  // caught exception while computing selection
            pActions->invalidateFont();
            infoMsg(
                "Couldn't locate the selected font!" + CannotLoadFontErrSuffix,
                "Manageable Error");
          }
        }
      },
      static_cast<void*>(&*performer));

  createTrackbar(
      ControlPanel_encodingTrName, winName, &encoding, 1,
      [](int val, void* userdata) {
        not_null<IControlPanelActions*> pActions =
            static_cast<IControlPanelActions*>(userdata);
        pActions->newFontEncoding(val);
      },
      static_cast<void*>(&*performer));
  createTrackbar(
      ControlPanel_fontSzTrName, winName, &fontSz, (int)Settings_MAX_FONT_SIZE,
      [](int val, void* userdata) {
        not_null<IControlPanelActions*> pActions =
            static_cast<IControlPanelActions*>(userdata);
        pActions->newFontSize(val);
      },
      static_cast<void*>(&*performer));

  createTrackbar(
      ControlPanel_symsBatchSzTrName, winName, &symsBatchSz,
      (int)SymsBatch_trackMax,
      [](int val, void* userdata) noexcept {
        not_null<IControlPanelActions*> pActions =
            static_cast<IControlPanelActions*>(userdata);
        pActions->newSymsBatchSize(val);
      },
      static_cast<void*>(&*performer));

  createButton(
      ControlPanel_restoreDefaultsLabel,
      [](int, void* userdata) noexcept {
        not_null<IControlPanelActions*> pActions =
            static_cast<IControlPanelActions*>(userdata);
        pActions->restoreUserDefaultMatchSettings();
      },
      static_cast<void*>(&*performer));
  createButton(
      ControlPanel_saveAsDefaultsLabel,
      [](int, void* userdata) noexcept {
        not_null<const IControlPanelActions*> pActions =
            static_cast<IControlPanelActions*>(userdata);
        pActions->setUserDefaultMatchSettings();
      },
      static_cast<void*>(&*performer));

  createTrackbar(
      ControlPanel_hybridResultTrName, winName, &hybridResult, 1,
      [](int state, void* userdata) noexcept {
        not_null<IControlPanelActions*> pActions =
            static_cast<IControlPanelActions*>(userdata);
        pActions->setResultMode(state);
      },
      static_cast<void*>(&*performer));

  createTrackbar(
      ControlPanel_structuralSimTrName, winName, &structuralSim,
      ControlPanel_Converter_StructuralSim_maxSlider,
      [](int val, void* userdata) noexcept {
        // Redeclared within lambda, since no capture is allowed
        not_null<IControlPanelActions*> pActions =
            static_cast<IControlPanelActions*>(userdata);
        pActions->newStructuralSimilarityFactor(
            slidersConverters()
                .at(&ControlPanel_structuralSimTrName)
                ->fromSlider(val));
      },
      static_cast<void*>(&*performer));
  createTrackbar(
      ControlPanel_correlationTrName, winName, &correlationCorrectness,
      ControlPanel_Converter_Correlation_maxSlider,
      [](int val, void* userdata) noexcept {
        // Redeclared within lambda, since no capture is allowed
        not_null<IControlPanelActions*> pActions =
            static_cast<IControlPanelActions*>(userdata);
        pActions->newCorrelationFactor(slidersConverters()
                                           .at(&ControlPanel_correlationTrName)
                                           ->fromSlider(val));
      },
      static_cast<void*>(&*performer));
  createTrackbar(
      ControlPanel_underGlyphCorrectnessTrName, winName, &underGlyphCorrectness,
      ControlPanel_Converter_Correctness_maxSlider,
      [](int val, void* userdata) noexcept {
        // Redeclared within lambda, since no capture is allowed
        not_null<IControlPanelActions*> pActions =
            static_cast<IControlPanelActions*>(userdata);
        pActions->newUnderGlyphCorrectnessFactor(
            slidersConverters()
                .at(&ControlPanel_underGlyphCorrectnessTrName)
                ->fromSlider(val));
      },
      static_cast<void*>(&*performer));
  createTrackbar(
      ControlPanel_glyphEdgeCorrectnessTrName, winName, &glyphEdgeCorrectness,
      ControlPanel_Converter_Correctness_maxSlider,
      [](int val, void* userdata) noexcept {
        // Redeclared within lambda, since no capture is allowed
        not_null<IControlPanelActions*> pActions =
            static_cast<IControlPanelActions*>(userdata);
        pActions->newGlyphEdgeCorrectnessFactor(
            slidersConverters()
                .at(&ControlPanel_glyphEdgeCorrectnessTrName)
                ->fromSlider(val));
      },
      static_cast<void*>(&*performer));
  createTrackbar(
      ControlPanel_asideGlyphCorrectnessTrName, winName, &asideGlyphCorrectness,
      ControlPanel_Converter_Correctness_maxSlider,
      [](int val, void* userdata) noexcept {
        // Redeclared within lambda, since no capture is allowed
        not_null<IControlPanelActions*> pActions =
            static_cast<IControlPanelActions*>(userdata);
        pActions->newAsideGlyphCorrectnessFactor(
            slidersConverters()
                .at(&ControlPanel_asideGlyphCorrectnessTrName)
                ->fromSlider(val));
      },
      static_cast<void*>(&*performer));

  createTrackbar(
      ControlPanel_moreContrastTrName, winName, &moreContrast,
      ControlPanel_Converter_Contrast_maxSlider,
      [](int val, void* userdata) noexcept {
        // Redeclared within lambda, since no capture is allowed
        not_null<IControlPanelActions*> pActions =
            static_cast<IControlPanelActions*>(userdata);
        pActions->newContrastFactor(slidersConverters()
                                        .at(&ControlPanel_moreContrastTrName)
                                        ->fromSlider(val));
      },
      static_cast<void*>(&*performer));

  createTrackbar(
      ControlPanel_gravityTrName, winName, &gravity,
      ControlPanel_Converter_Gravity_maxSlider,
      [](int val, void* userdata) noexcept {
        // Redeclared within lambda, since no capture is allowed
        not_null<IControlPanelActions*> pActions =
            static_cast<IControlPanelActions*>(userdata);
        pActions->newGravitationalSmoothnessFactor(
            slidersConverters()
                .at(&ControlPanel_gravityTrName)
                ->fromSlider(val));
      },
      static_cast<void*>(&*performer));
  createTrackbar(
      ControlPanel_directionTrName, winName, &direction,
      ControlPanel_Converter_Direction_maxSlider,
      [](int val, void* userdata) noexcept {
        // Redeclared within lambda, since no capture is allowed
        not_null<IControlPanelActions*> pActions =
            static_cast<IControlPanelActions*>(userdata);
        pActions->newDirectionalSmoothnessFactor(
            slidersConverters()
                .at(&ControlPanel_directionTrName)
                ->fromSlider(val));
      },
      static_cast<void*>(&*performer));

  createTrackbar(
      ControlPanel_largerSymTrName, winName, &largerSym,
      ControlPanel_Converter_LargerSym_maxSlider,
      [](int val, void* userdata) noexcept {
        // Redeclared within lambda, since no capture is allowed
        not_null<IControlPanelActions*> pActions =
            static_cast<IControlPanelActions*>(userdata);
        pActions->newGlyphWeightFactor(slidersConverters()
                                           .at(&ControlPanel_largerSymTrName)
                                           ->fromSlider(val));
      },
      static_cast<void*>(&*performer));

  createTrackbar(
      ControlPanel_thresh4BlanksTrName, winName, &thresh4Blanks,
      (int)Settings_MAX_THRESHOLD_FOR_BLANKS,
      [](int val, void* userdata) noexcept {
        not_null<IControlPanelActions*> pActions =
            static_cast<IControlPanelActions*>(userdata);
        pActions->newThreshold4BlanksFactor((unsigned)val);
      },
      static_cast<void*>(&*performer));

  createButton(
      ControlPanel_aboutLabel,
      [](int, void* userdata) noexcept {
        not_null<IControlPanelActions*> pActions =
            static_cast<IControlPanelActions*>(userdata);
        pActions->showAboutDlg(ControlPanel_aboutLabel, ControlPanel_aboutText);
      },
      static_cast<void*>(&*performer));
  createButton(
      ControlPanel_instructionsLabel,
      [](int, void* userdata) noexcept {
        not_null<IControlPanelActions*> pActions =
            static_cast<IControlPanelActions*>(userdata);
        pActions->showInstructionsDlg(ControlPanel_instructionsLabel,
                                      ControlPanel_instructionsText);
      },
      static_cast<void*>(&*performer));
  createButton(
      ControlPanel_loadSettingsLabel,
      [](int, void* userdata) {
        not_null<IControlPanelActions*> pActions =
            static_cast<IControlPanelActions*>(userdata);
        ignore = pActions->loadSettings();
      },
      static_cast<void*>(&*performer));
  createButton(
      ControlPanel_saveSettingsLabel,
      [](int, void* userdata) noexcept {
        not_null<const IControlPanelActions*> pActions =
            static_cast<IControlPanelActions*>(userdata);
        pActions->saveSettings();
      },
      static_cast<void*>(&*performer));
}

}  // namespace ui
}  // namespace pic2sym

#endif  // UNIT_TESTING not defined
