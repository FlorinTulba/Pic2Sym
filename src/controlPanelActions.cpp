/******************************************************************************
 The application Pic2Sym approximates images by a
 grid of colored symbols with colored backgrounds.

 Copyrights from the libraries used by the program:
 - (c) 2003 Boost (www.boost.org)
     License: doc/licenses/Boost.lic
     http://www.boost.org/LICENSE_1_0.txt
 - (c) 2015-2016 OpenCV (www.opencv.org)
     License: doc/licenses/OpenCV.lic
     http://opencv.org/license/
 - (c) 1996-2002, 2006 The FreeType Project (www.freetype.org)
     License: doc/licenses/FTL.txt
     http://git.savannah.gnu.org/cgit/freetype/freetype2.git/plain/docs/FTL.TXT
 - (c) 1997-2002 OpenMP Architecture Review Board (www.openmp.org)
   (c) Microsoft Corporation (implementation for OpenMP C/C++ v2.0 March 2002)
     See: https://msdn.microsoft.com/en-us/library/8y6825x5.aspx
 - (c) 1995-2017 zlib software (Jean-loup Gailly and Mark Adler - www.zlib.net)
     License: doc/licenses/zlib.lic
     http://www.zlib.net/zlib_license.html


 (c) 2016-2019 Florin Tulba <florintulba@yahoo.com>

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

#include "cmapInspectBase.h"
#include "comparatorBase.h"
#include "controlPanel.h"
#include "controlPanelActions.h"
#include "controllerBase.h"
#include "dlgs.h"
#include "fontEngineBase.h"
#include "img.h"
#include "imgSettings.h"
#include "matchAssessment.h"
#include "matchSettings.h"
#include "misc.h"
#include "settings.h"
#include "symSettings.h"
#include "symsLoadingFailure.h"
#include "tinySymsProvider.h"
#include "transformBase.h"

#pragma warning(push, 0)

#define WIN32_LEAN_AND_MEAN
#include <Windows.h>

#include <fstream>
#include <iostream>
#include <sstream>

#include <opencv2/core/core.hpp>

#include <filesystem>

#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>

#pragma warning(pop)

using namespace std;
using namespace cv;
using namespace std::filesystem;
using namespace boost::archive;

#pragma warning(disable : WARN_REF_TO_CONST_UNIQUE_PTR)
ControlPanelActions::ControlPanelActions(
    IController& ctrler_,
    ISettingsRW& cfg_,
    IFontEngine& fe_,
    MatchAssessor& ma_,
    ITransformer& t_,
    IComparator& comp_,
    const std::unique_ptr<ICmapInspect>& pCmi_) noexcept
    : ctrler(ctrler_),
      cfg(cfg_),
      fe(fe_),
      ma(ma_),
      t(t_),
      img(getImg()),
      comp(comp_),
      cp(getControlPanel(cfg_)),
      pCmi(pCmi_) {
  assert(!pCmi);  // comes nullptr; changes after _newFontFamily
}
#pragma warning(default : WARN_REF_TO_CONST_UNIQUE_PTR)

bool ControlPanelActions::validState(bool imageRequired /* = true*/) const
    noexcept {
  const bool noEnabledMatchAspects = (ma.enabledMatchAspectsCount() == 0ULL);
  if ((imageOk || !imageRequired) && fontFamilyOk && !noEnabledMatchAspects)
    return true;

  ostringstream oss;
  oss << "The problems are:\n\n";
  if (imageRequired && !imageOk)
    oss << "- no image to transform\n";
  if (!fontFamilyOk)
    oss << "- no font family to use during transformation\n";
  if (noEnabledMatchAspects)
    oss << "- no enabled matching aspects to consider\n";
  errMsg(oss.str(), "Please Correct these errors first!");
  return false;
}

// Methods from below have different definitions for UnitTesting project
#ifndef UNIT_TESTING

#define GET_FIELD(FieldType, ...)      \
  static FieldType field{__VA_ARGS__}; \
  return field

Img& ControlPanelActions::getImg() noexcept {
  GET_FIELD(Img);
}

IControlPanel& ControlPanelActions::getControlPanel(
    const ISettingsRW& cfg_) noexcept {
  GET_FIELD(ControlPanel, *this, cfg_);
}

#undef GET_FIELD

#endif  // UNIT_TESTING not defined

void ControlPanelActions::restoreUserDefaultMatchSettings() noexcept(!UT) {
  extern const cv::String ControlPanel_restoreDefaultsLabel;
  const unique_ptr<const ActionPermit> permit =
      cp.actionDemand(ControlPanel_restoreDefaultsLabel);
  if (nullptr == permit)
    return;

#ifndef UNIT_TESTING
  cfg.refMS().replaceByUserDefaults();
#endif  // UNIT_TESTING not defined

  cp.updateMatchSettings(cfg.getMS());
  ma.updateEnabledMatchAspectsCount();
}

void ControlPanelActions::setUserDefaultMatchSettings() const noexcept {
  extern const cv::String ControlPanel_saveAsDefaultsLabel;
  const unique_ptr<const ActionPermit> permit =
      cp.actionDemand(ControlPanel_saveAsDefaultsLabel);
  if (nullptr == permit)
    return;

#ifndef UNIT_TESTING
  cfg.refMS().saveAsUserDefaults();
#endif  // UNIT_TESTING not defined
}

bool ControlPanelActions::loadSettings(const string& from /* = ""*/) noexcept {
  extern const cv::String ControlPanel_loadSettingsLabel;
  const unique_ptr<const ActionPermit> permit =
      cp.actionDemand(ControlPanel_loadSettingsLabel);
  if (nullptr == permit)
    return false;

  string sourceFile;
  if (!from.empty()) {
    sourceFile = from;

  } else {
    // Prompting the user for the file to be loaded
    static SettingsSelector ss;  // loader

    if (!ss.promptForUserChoice())
      return false;

    sourceFile = ss.selection();
  }

  // Keep a copy of old SymSettings
  const unique_ptr<ISymSettings> prevSymSettings = cfg.getSS().clone();

  cout << "Loading settings from '" << sourceFile << '\'' << endl;

#pragma warning(disable : WARN_SEH_NOT_CAUGHT)
  try {
    ifstream ifs(sourceFile, ios::binary);
    binary_iarchive ia(ifs);
    ia >> dynamic_cast<Settings&>(cfg);
  } catch (...) {
    cerr << "Couldn't load these settings" << endl;
    return false;
  }
#pragma warning(default : WARN_SEH_NOT_CAUGHT)

  cp.updateMatchSettings(cfg.getMS());
  ma.updateEnabledMatchAspectsCount();
  cp.updateImgSettings(cfg.getIS());

  if (dynamic_cast<const SymSettings&>(*prevSymSettings) ==
      dynamic_cast<const SymSettings&>(cfg.getSS()))
    return true;

  bool fontFileChanged = false, encodingChanged = false;
  const string newEncName = cfg.getSS().getEncoding();
  if (prevSymSettings->getFontFile() != cfg.getSS().getFontFile()) {
    _newFontFamily(cfg.getSS().getFontFile(), true);
    fontFileChanged = true;
  }
  // New font file or not, _newFontFamily() was called at least once, so:
  assert(pCmi);  // holds after a call to _newFontFamily()

  if ((!fontFileChanged && prevSymSettings->getEncoding() != newEncName) ||
      (fontFileChanged && cfg.getSS().getEncoding() != newEncName)) {
    _newFontEncoding(newEncName, true);
    encodingChanged = true;
  }

  if (prevSymSettings->getFontSz() != cfg.getSS().getFontSz()) {
    if (fontFileChanged || encodingChanged) {
      pCmi->updateGrid();
    } else {
      _newFontSize((int)cfg.getSS().getFontSz(), true);
    }
  }

  unsigned currEncIdx;
  fe.getEncoding(&currEncIdx);
  cp.updateSymSettings(currEncIdx, cfg.getSS().getFontSz());

  try {
    ctrler.symbolsChanged();
  } catch (const TinySymsLoadingFailure&) {
    invalidateFont();
    SymsLoadingFailure::informUser(
        "Couldn't load the tiny versions "
        "of the font pointed "
        "by this settings file!");
    return false;
  } catch (const NormalSymsLoadingFailure&) {
    invalidateFont();
    SymsLoadingFailure::informUser(
        "Couldn't load the normal versions "
        "of the font pointed "
        "by this settings file!");
    return false;
  }

  if (Settings::olderVersionDuringLastIO())
#pragma warning(disable : WARN_SEH_NOT_CAUGHT)
    try {  // Rewriting the file. Same thread is used.
      ofstream ofs(sourceFile, ios::binary);
      binary_oarchive oa(ofs);
      oa << dynamic_cast<const Settings&>(cfg);
      cout << "Rewritten settings to `" << sourceFile
           << "` because it used older versions of some classes!" << endl;
    } catch (...) {
      cout << "Information: Unable to upgrade the file " << sourceFile << endl;
    }
#pragma warning(default : WARN_SEH_NOT_CAUGHT)

  return true;
}

void ControlPanelActions::saveSettings() const noexcept {
  extern const cv::String ControlPanel_saveSettingsLabel;
  if (!cp.actionDemand(ControlPanel_saveSettingsLabel))
    return;

  if (!cfg.getSS().initialized()) {
    warnMsg(
        "There's no Font yet.\nSave settings only after selecting a font !");
    return;
  }

  static SettingsSelector ss(false);  // saver

  if (!ss.promptForUserChoice())
    return;

  cout << "Saving settings to '" << ss.selection() << '\'' << endl;

#pragma warning(disable : WARN_SEH_NOT_CAUGHT)
  try {
    ofstream ofs(ss.selection(), ios::binary);
    binary_oarchive oa(ofs);
    oa << dynamic_cast<const Settings&>(cfg);
  } catch (...) {
    cerr << "Couldn't save current settings" << endl;
    return;
  }
#pragma warning(default : WARN_SEH_NOT_CAUGHT)
}

unsigned ControlPanelActions::getFontEncodingIdx() const noexcept {
  if (fontFamilyOk) {
    unsigned currEncIdx;
    fe.getEncoding(&currEncIdx);

    return currEncIdx;
  }

  // This method must NOT throw when fontFamilyOk is false,
  // since the requested index appears on the Control Panel, and must exist if:
  // - no font was loaded yet
  // - a font has been loaded
  // - a requested font couldn't be loaded and was discarded
  return 0U;
}

bool ControlPanelActions::newImage(const string& imgPath,
                                   bool silent /* = false*/) noexcept {
  extern const cv::String ControlPanel_selectImgLabel;
  if (!cp.actionDemand(ControlPanel_selectImgLabel))
    return false;

  if (img.absPath() == absolute(imgPath))
    return true;  // same image

  if (!img.reset(imgPath)) {
    if (!silent) {
      ostringstream oss;
      oss << "Invalid image file: '" << imgPath << '\'';
      errMsg(oss.str());
    }
    return false;
  }

  ostringstream oss;
  oss << "Pic2Sym on image: " << img.absPath();
  comp.setTitle(oss.str());

  if (!imageOk) {  // 1st image loaded
    comp.permitResize();
    imageOk = true;
  }

  const cv::Mat& orig = img.original();
  comp.setReference(orig);  // displays the image
  comp.resize();
  return true;
}

void ControlPanelActions::invalidateFont() noexcept {
  fontFamilyOk = false;

  cp.updateEncodingsCount(1U);
  if (pCmi)
    pCmi->clear();

  cfg.refSS().reset();
  fe.invalidateFont();
}

bool ControlPanelActions::_newFontFamily(
    const string& fontFile,
    bool forceUpdate /* = false*/) noexcept {
  if (fe.fontFileName() == fontFile && !forceUpdate)
    return false;  // same font

  if (!fe.newFont(fontFile)) {
    ostringstream oss;
    oss << "Invalid font file: '" << fontFile << '\'';
    errMsg(oss.str());
    return false;
  }

  cp.updateEncodingsCount(fe.uniqueEncodings());

  if (!fontFamilyOk) {
    fontFamilyOk = true;

    ctrler.ensureExistenceCmapInspect();
    assert(pCmi);  // effect of previous statement
  }

  return true;
}

void ControlPanelActions::newFontFamily(const string& fontFile) noexcept {
  extern const cv::String ControlPanel_selectFontLabel;
  if (!cp.actionDemand(ControlPanel_selectFontLabel))
    return;

  if (!_newFontFamily(fontFile))
    return;

  try {
    ctrler.symbolsChanged();
  } catch (const TinySymsLoadingFailure&) {
    invalidateFont();
    SymsLoadingFailure::informUser(
        "Couldn't load the tiny versions "
        "of the newly selected font family!");
  } catch (const NormalSymsLoadingFailure&) {
    invalidateFont();
    SymsLoadingFailure::informUser(
        "Couldn't load the normal versions "
        "of the newly selected font family!");
  }
}

void ControlPanelActions::newFontEncoding(int encodingIdx) noexcept {
  // Ignore call if no font yet, or just 1 encoding,
  // or if the required hack (mentioned in 'views.h') provoked this call
  if (!fontFamilyOk || fe.uniqueEncodings() <= 1U || cp.encMaxHack())
    return;

  unsigned currEncIdx;
  fe.getEncoding(&currEncIdx);
  if (currEncIdx == (unsigned)encodingIdx)
    return;

  extern const cv::String ControlPanel_encodingTrName;
  if (!cp.actionDemand(ControlPanel_encodingTrName))
    return;

  fe.setNthUniqueEncoding((unsigned)encodingIdx);

  try {
    ctrler.symbolsChanged();
  } catch (const TinySymsLoadingFailure&) {
    invalidateFont();
    SymsLoadingFailure::informUser(
        "Couldn't load the tiny versions of the font "
        "whose encoding has been updated!");
  } catch (const NormalSymsLoadingFailure&) {
    invalidateFont();
    SymsLoadingFailure::informUser(
        "Couldn't load the normal versions of the font "
        "whose encoding has been updated!");
  }
}

bool ControlPanelActions::_newFontEncoding(
    const string& encName,
    bool forceUpdate /* = false*/) noexcept {
  return fe.setEncoding(encName, forceUpdate);
}

#ifdef UNIT_TESTING
bool ControlPanelActions::newFontEncoding(const string& encName) noexcept {
  bool result = _newFontEncoding(encName);
  if (result) {
    try {
      ctrler.symbolsChanged();
    } catch (const TinySymsLoadingFailure&) {
      invalidateFont();
      SymsLoadingFailure::informUser(
          "Couldn't load the tiny versions of the font "
          "whose encoding has been updated!");
      return false;
    } catch (const NormalSymsLoadingFailure&) {
      invalidateFont();
      SymsLoadingFailure::informUser(
          "Couldn't load the normal versions of the font "
          "whose encoding has been updated!");
      return false;
    }
  }

  return result;
}
#endif  // UNIT_TESTING defined

bool ControlPanelActions::_newFontSize(int fontSz,
                                       bool forceUpdate /* = false*/) noexcept {
  extern const cv::String ControlPanel_fontSzTrName;
  extern const unsigned Settings_MIN_FONT_SIZE;

  if (!ISettings::isFontSizeOk((unsigned)fontSz)) {
    ostringstream oss;
    oss << "Invalid font size. Please set at least " << Settings_MIN_FONT_SIZE
        << '.';
    cp.restoreSliderValue(ControlPanel_fontSzTrName, oss.str());
    return false;
  }

  if ((unsigned)fontSz == cfg.getSS().getFontSz() && !forceUpdate)
    return false;

  cfg.refSS().setFontSz((unsigned)fontSz);

  if (!fontFamilyOk) {
    if (pCmi)
      pCmi->clear();
    return false;
  }

  pCmi->updateGrid();

  return true;
}

void ControlPanelActions::newFontSize(int fontSz) noexcept {
  extern const cv::String ControlPanel_fontSzTrName;
  if (!cp.actionDemand(ControlPanel_fontSzTrName))
    return;

  if (!_newFontSize(fontSz))
    return;

  try {
    ctrler.symbolsChanged();
  } catch (const TinySymsLoadingFailure&) {
    invalidateFont();
    SymsLoadingFailure::informUser(
        "Couldn't load the tiny versions of the font "
        "whose size has been updated!");
  } catch (const NormalSymsLoadingFailure&) {
    invalidateFont();
    SymsLoadingFailure::informUser(
        "Couldn't load the requested size versions "
        "of the fonts!");
  }
}

void ControlPanelActions::newSymsBatchSize(int symsBatchSz) noexcept {
  extern const cv::String ControlPanel_symsBatchSzTrName;
  if (!cp.actionDemand(ControlPanel_symsBatchSzTrName))
    return;

  t.setSymsBatchSize(symsBatchSz);
}

void ControlPanelActions::newHmaxSyms(int maxSymbols) noexcept {
  extern const cv::String ControlPanel_outWTrName;
  extern const unsigned Settings_MIN_H_SYMS;

  if (!cp.actionDemand(ControlPanel_outWTrName))
    return;

  // It's possible if the previous value was invalid
  if ((unsigned)maxSymbols == cfg.getIS().getMaxHSyms())
    return;

  if (!ISettings::isHmaxSymsOk((unsigned)maxSymbols)) {
    ostringstream oss;
    oss << "Invalid max number of horizontal symbols. "
           "Please set at least "
        << Settings_MIN_H_SYMS << '.';
    cp.restoreSliderValue(ControlPanel_outWTrName, oss.str());
    return;
  }

  cfg.refIS().setMaxHSyms((unsigned)maxSymbols);
}

void ControlPanelActions::newVmaxSyms(int maxSymbols) noexcept {
  extern const cv::String ControlPanel_outHTrName;
  extern const unsigned Settings_MIN_V_SYMS;

  if (!cp.actionDemand(ControlPanel_outHTrName))
    return;

  if ((unsigned)maxSymbols ==
      cfg.getIS()
          .getMaxVSyms())  // it's possible if the previous value was invalid
    return;

  if (!ISettings::isVmaxSymsOk((unsigned)maxSymbols)) {
    ostringstream oss;
    oss << "Invalid max number of vertical symbols. Please set at least "
        << Settings_MIN_V_SYMS << '.';
    cp.restoreSliderValue(ControlPanel_outHTrName, oss.str());
    return;
  }

  cfg.refIS().setMaxVSyms((unsigned)maxSymbols);
}

void ControlPanelActions::setResultMode(bool hybrid) noexcept {
  extern const cv::String ControlPanel_hybridResultTrName;
  if (!cp.actionDemand(ControlPanel_hybridResultTrName))
    return;

  cfg.refMS().setResultMode(hybrid);
}

void ControlPanelActions::newThreshold4BlanksFactor(
    unsigned threshold) noexcept {
  extern const cv::String ControlPanel_thresh4BlanksTrName;
  if (!cp.actionDemand(ControlPanel_thresh4BlanksTrName))
    return;

  cfg.refMS().setBlankThreshold(threshold);
}

#define UPDATE_MATCH_ASPECT_VALUE(AspectName, NewValue)          \
  const double PrevVal = cfg.getMS().get_k##AspectName();        \
  if (NewValue != PrevVal) {                                     \
    cfg.refMS().set_k##AspectName(NewValue);                     \
    if (PrevVal == 0.) { /* just enabled this aspect */          \
      ma.newlyEnabledMatchAspect();                              \
    } else if (NewValue == 0.) { /* just disabled this aspect */ \
      ma.newlyDisabledMatchAspect();                             \
    }                                                            \
  }

void ControlPanelActions::newContrastFactor(double k) noexcept {
  extern const cv::String ControlPanel_moreContrastTrName;
  if (!cp.actionDemand(ControlPanel_moreContrastTrName))
    return;

  UPDATE_MATCH_ASPECT_VALUE(Contrast, k);
}

void ControlPanelActions::newStructuralSimilarityFactor(double k) noexcept {
  extern const cv::String ControlPanel_structuralSimTrName;
  if (!cp.actionDemand(ControlPanel_structuralSimTrName))
    return;

  UPDATE_MATCH_ASPECT_VALUE(Ssim, k);
}

void ControlPanelActions::newCorrelationFactor(double k) noexcept {
  extern const cv::String ControlPanel_correlationTrName;
  if (!cp.actionDemand(ControlPanel_correlationTrName))
    return;

  UPDATE_MATCH_ASPECT_VALUE(Correl, k);
}

void ControlPanelActions::newUnderGlyphCorrectnessFactor(double k) noexcept {
  extern const cv::String ControlPanel_underGlyphCorrectnessTrName;
  if (!cp.actionDemand(ControlPanel_underGlyphCorrectnessTrName))
    return;

  UPDATE_MATCH_ASPECT_VALUE(SdevFg, k);
}

void ControlPanelActions::newAsideGlyphCorrectnessFactor(double k) noexcept {
  extern const cv::String ControlPanel_asideGlyphCorrectnessTrName;
  if (!cp.actionDemand(ControlPanel_asideGlyphCorrectnessTrName))
    return;

  UPDATE_MATCH_ASPECT_VALUE(SdevBg, k);
}

void ControlPanelActions::newGlyphEdgeCorrectnessFactor(double k) noexcept {
  extern const cv::String ControlPanel_glyphEdgeCorrectnessTrName;
  if (!cp.actionDemand(ControlPanel_glyphEdgeCorrectnessTrName))
    return;

  UPDATE_MATCH_ASPECT_VALUE(SdevEdge, k);
}

void ControlPanelActions::newDirectionalSmoothnessFactor(double k) noexcept {
  extern const cv::String ControlPanel_directionTrName;
  if (!cp.actionDemand(ControlPanel_directionTrName))
    return;

  UPDATE_MATCH_ASPECT_VALUE(CosAngleMCs, k);
}

void ControlPanelActions::newGravitationalSmoothnessFactor(double k) noexcept {
  extern const cv::String ControlPanel_gravityTrName;
  if (!cp.actionDemand(ControlPanel_gravityTrName))
    return;

  UPDATE_MATCH_ASPECT_VALUE(MCsOffset, k);
}

void ControlPanelActions::newGlyphWeightFactor(double k) noexcept {
  extern const cv::String ControlPanel_largerSymTrName;
  if (!cp.actionDemand(ControlPanel_largerSymTrName))
    return;

  UPDATE_MATCH_ASPECT_VALUE(SymDensity, k);
}

#undef UPDATE_MATCH_ASPECT_VALUE

bool ControlPanelActions::performTransformation(
    double* durationS /* = nullptr*/) noexcept {
  extern const cv::String ControlPanel_transformImgLabel;
  if (!cp.actionDemand(ControlPanel_transformImgLabel))
    return false;

  if (!validState())
    return false;

  t.run();

  if (nullptr != durationS)
    *durationS = t.duration();

  return true;
}

void ControlPanelActions::showAboutDlg(const string& title,
                                       const wstring& content) noexcept {
  extern const cv::String ControlPanel_aboutLabel;
  if (!cp.actionDemand(ControlPanel_aboutLabel))
    return;

  MessageBox(nullptr, content.c_str(), str2wstr(title).c_str(),
             MB_ICONINFORMATION | MB_OK | MB_TASKMODAL | MB_SETFOREGROUND);
}

void ControlPanelActions::showInstructionsDlg(const string& title,
                                              const wstring& content) noexcept {
  extern const cv::String ControlPanel_instructionsLabel;
  if (!cp.actionDemand(ControlPanel_instructionsLabel))
    return;

  MessageBox(nullptr, content.c_str(), str2wstr(title).c_str(),
             MB_ICONINFORMATION | MB_OK | MB_TASKMODAL | MB_SETFOREGROUND);
}
