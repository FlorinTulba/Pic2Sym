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

#include "bestMatchBase.h"
#include "clusterEngineBase.h"
#include "clusterSupport.h"
#include "cmapPerspective.h"
#include "controlPanel.h"
#include "controlPanelActions.h"
#include "controller.h"
#include "fontEngine.h"
#include "glyphsProgressTracker.h"
#include "jobMonitor.h"
#include "match.h"
#include "matchAssessment.h"
#include "matchEngine.h"
#include "matchParamsBase.h"
#include "matchSupport.h"
#include "misc.h"
#include "picTransformProgressTracker.h"
#include "presentCmap.h"
#include "progressNotifier.h"
#include "resizedImgBase.h"
#include "selectSymbols.h"
#include "settingsBase.h"
#include "symSettingsBase.h"
#include "transform.h"
#include "transformSupportBase.h"
#include "updateSymSettings.h"
#include "views.h"

#pragma warning(push, 0)

#include <sstream>

#pragma warning(pop)

using namespace std;

extern const double Transform_ProgressReportsIncrement;
extern const double SymbolsProcessing_ProgressReportsIncrement;
extern const string Controller_PREFIX_GLYPH_PROGRESS;

namespace {
/// Adapter from IProgressNotifier to IGlyphsProgressTracker
class SymsUpdateProgressNotifier : public IProgressNotifier {
 public:
  explicit SymsUpdateProgressNotifier(const IController& performer_) noexcept
      : performer(performer_) {}

  void notifyUser(const std::string&, double progress) noexcept override {
    performer.hourGlass(progress, Controller_PREFIX_GLYPH_PROGRESS,
                        true);  // async call
  }

 private:
  const IController& performer;
};

/// Adapter from IProgressNotifier to IPicTransformProgressTracker
class PicTransformProgressNotifier : public IProgressNotifier {
 public:
  explicit PicTransformProgressNotifier(
      const IPicTransformProgressTracker& performer_) noexcept
      : performer(performer_) {}

  void notifyUser(const std::string&, double progress) noexcept override {
    performer.reportTransformationProgress(progress);
  }

 private:
  const IPicTransformProgressTracker& performer;
};
}  // anonymous namespace

#pragma warning(disable : WARN_BASE_INIT_USING_THIS)
Controller::Controller(ISettingsRW& s) noexcept
    : updateSymSettings(make_unique<const UpdateSymSettings>(s.refSS())),
      glyphsProgressTracker(make_unique<const GlyphsProgressTracker>(*this)),
      picTransformProgressTracker(
          make_unique<PicTransformProgressTracker>(*this)),
      glyphsUpdateMonitor(make_unique<JobMonitor>(
          "Processing glyphs",
          make_unique<SymsUpdateProgressNotifier>(*this),
          SymbolsProcessing_ProgressReportsIncrement)),
      imgTransformMonitor(
          make_unique<JobMonitor>("Transforming image",
                                  make_unique<PicTransformProgressNotifier>(
                                      getPicTransformProgressTracker()),
                                  Transform_ProgressReportsIncrement)),
      cmP(make_unique<CmapPerspective>()),
      presentCmap(make_unique<const PresentCmap>(
          *this,
          *cmP,
          getMatchEngine(s).isClusteringUseful())),
      fe(getFontEngine(s.getSS()).useSymsMonitor(*glyphsUpdateMonitor)),
      cfg(s),
      me(getMatchEngine(s).useSymsMonitor(*glyphsUpdateMonitor)),
      t(getTransformer(s).useTransformMonitor(*imgTransformMonitor)),
      comp(getComparator()),
      pCmi(),
      selectSymbols(make_unique<const SelectSymbols>(*this,
                                                     getMatchEngine(s),
                                                     *cmP,
                                                     pCmi)),
      controlPanelActions(
          make_unique<ControlPanelActions>(*this,
                                           s,
                                           getFontEngine(s.getSS()),
                                           getMatchEngine(s).mutableAssessor(),
                                           getTransformer(s),
                                           getComparator(),
                                           pCmi)) {
  // The constructor ensures following asserts:
  assert(updateSymSettings);
  assert(glyphsProgressTracker);
  assert(glyphsUpdateMonitor);
  assert(picTransformProgressTracker);
  assert(selectSymbols);
  /*
   No assert(presentCmap) as this method is also called during Controller's
   construction within a cyclic dependency while the presentCmap is
   initialized
  */
  assert(controlPanelActions);
}
#pragma warning(default : WARN_BASE_INIT_USING_THIS)

const IUpdateSymSettings& Controller::getUpdateSymSettings() const noexcept {
  return *updateSymSettings;
}

const IGlyphsProgressTracker& Controller::getGlyphsProgressTracker() const
    noexcept {
  return *glyphsProgressTracker;
}

IPicTransformProgressTracker& Controller::getPicTransformProgressTracker() const
    noexcept {
  return *picTransformProgressTracker;
}

const std::unique_ptr<const IPresentCmap>& Controller::getPresentCmap() const
    noexcept {
  return presentCmap;
}

void Controller::ensureExistenceCmapInspect() noexcept {
  if (!pCmi)
    pCmi =
        make_unique<CmapInspect>(*presentCmap, *selectSymbols, getFontSize());
}

IControlPanelActions& Controller::getControlPanelActions() const noexcept {
  return *controlPanelActions;
}

const unsigned& Controller::getFontSize() const noexcept {
  return cfg.getSS().getFontSz();
}

// Methods from below have different definitions for UnitTesting project
#ifndef UNIT_TESTING

#define GET_FIELD(FieldType, ...)      \
  static FieldType field{__VA_ARGS__}; \
  return field

IComparator& Controller::getComparator() noexcept {
  GET_FIELD(Comparator);
}

IFontEngine& Controller::getFontEngine(const ISymSettings& ss_) noexcept {
  GET_FIELD(FontEngine, *this, ss_);
}

IMatchEngine& Controller::getMatchEngine(const ISettings& cfg_) noexcept {
  GET_FIELD(MatchEngine, cfg_, getFontEngine(cfg_.getSS()), *cmP);
}

ITransformer& Controller::getTransformer(const ISettings& cfg_) noexcept {
  GET_FIELD(Transformer, *this, cfg_, getMatchEngine(cfg_),
            (IBasicImgData&)ControlPanelActions::getImg());
}

#undef GET_FIELD

#endif  // UNIT_TESTING not defined
