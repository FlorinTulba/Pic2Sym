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
 
 (c) 2016, 2017 Florin Tulba <florintulba@yahoo.com>

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

#include "controller.h"
#include "settingsBase.h"
#include "symSettingsBase.h"
#include "updateSymSettings.h"
#include "fontEngine.h"
#include "match.h"
#include "matchEngine.h"
#include "matchParamsBase.h"
#include "bestMatchBase.h"
#include "matchAssessment.h"
#include "views.h"
#include "resizedImgBase.h"
#include "transform.h"
#include "transformSupportBase.h"
#include "matchSupport.h"
#include "clusterEngineBase.h"
#include "clusterSupport.h"
#include "controlPanel.h"
#include "controlPanelActions.h"
#include "presentCmap.h"
#include "cmapPerspective.h"
#include "selectSymbols.h"
#include "progressNotifier.h"
#include "glyphsProgressTracker.h"
#include "picTransformProgressTracker.h"
#include "jobMonitor.h"
#include "misc.h"

#pragma warning ( push, 0 )

#include <sstream>

#pragma warning ( pop )

using namespace std;

extern const double Transform_ProgressReportsIncrement;
extern const double SymbolsProcessing_ProgressReportsIncrement;
extern const stringType Controller_PREFIX_GLYPH_PROGRESS;

namespace {
	/// Adapter from IProgressNotifier to IGlyphsProgressTracker
	struct SymsUpdateProgressNotifier : IProgressNotifier {
		const IController &performer;

		SymsUpdateProgressNotifier(const IController &performer_) : performer(performer_) {}
		void operator=(const SymsUpdateProgressNotifier&) = delete;

		void notifyUser(const std::stringType&, double progress) override {
			performer.hourGlass(progress, Controller_PREFIX_GLYPH_PROGRESS, true); // async call
		}
	};

	/// Adapter from IProgressNotifier to IPicTransformProgressTracker
	struct PicTransformProgressNotifier : IProgressNotifier {
		const IPicTransformProgressTracker &performer;

		PicTransformProgressNotifier(const IPicTransformProgressTracker &performer_) : performer(performer_) {}
		void operator=(const PicTransformProgressNotifier&) = delete;

		void notifyUser(const std::stringType&, double progress) override {
			performer.reportTransformationProgress(progress);
		}
	};
} // anonymous namespace

#pragma warning( disable : WARN_BASE_INIT_USING_THIS )
Controller::Controller(ISettingsRW &s) :
		updateSymSettings(std::makeUnique<const UpdateSymSettings>(s.refSS())),
		glyphsProgressTracker(std::makeUnique<const GlyphsProgressTracker>(*this)),
		picTransformProgressTracker(new PicTransformProgressTracker(*this)),
		glyphsUpdateMonitor(new JobMonitor("Processing glyphs",
			std::makeUnique<SymsUpdateProgressNotifier>(*this),
			SymbolsProcessing_ProgressReportsIncrement)),
		imgTransformMonitor(new JobMonitor("Transforming image",
			std::makeUnique<PicTransformProgressNotifier>(getPicTransformProgressTracker()),
			Transform_ProgressReportsIncrement)),
		cmP(new CmapPerspective),
		presentCmap(std::makeUnique<const PresentCmap>(*this, *cmP,
			getMatchEngine(s).isClusteringUseful())),
		fe(getFontEngine(s.getSS()).useSymsMonitor(*glyphsUpdateMonitor)),
		cfg(s),
		me(getMatchEngine(s).useSymsMonitor(*glyphsUpdateMonitor)),
		t(getTransformer(s).useTransformMonitor(*imgTransformMonitor)),
		comp(getComparator()),
		pCmi(),
		selectSymbols(std::makeUnique<const SelectSymbols>(*this, getMatchEngine(s), *cmP, pCmi)),
		controlPanelActions(new ControlPanelActions(*this, s,
			getFontEngine(s.getSS()), getMatchEngine(s).assessor(),
			getTransformer(s), getComparator(), pCmi)) {}
#pragma warning( default : WARN_BASE_INIT_USING_THIS )

const IUpdateSymSettings& Controller::getUpdateSymSettings() const {
	assert(updateSymSettings);
	return *updateSymSettings;
}

const IGlyphsProgressTracker& Controller::getGlyphsProgressTracker() const {
	assert(glyphsProgressTracker);
	return *glyphsProgressTracker;
}

IPicTransformProgressTracker& Controller::getPicTransformProgressTracker() {
	assert(picTransformProgressTracker);
	return *picTransformProgressTracker;
}

const std::uniquePtr<const IPresentCmap>& Controller::getPresentCmap() const {
	// no assert(presentCmap) as this method is also called during Controller's construction within a cyclic dependency while the presentCmap is initialized
	return presentCmap;
}

void Controller::ensureExistenceCmapInspect() {
	if(!pCmi)
		pCmi.reset(new CmapInspect(*presentCmap, *selectSymbols, getFontSize()));
}

IControlPanelActions& Controller::getControlPanelActions() {
	assert(controlPanelActions);
	return *controlPanelActions;
}

const unsigned& Controller::getFontSize() const {
	return cfg.getSS().getFontSz();
}

// Methods from below have different definitions for UnitTesting project
#ifndef UNIT_TESTING

#ifndef AI_REVIEWER_CHECK

#define GET_FIELD_NO_ARGS(FieldType) \
	__pragma( warning( disable : WARN_THREAD_UNSAFE ) ) \
	static FieldType field; \
	__pragma( warning( default : WARN_THREAD_UNSAFE ) ) \
	return field

#define GET_FIELD(FieldType, ...) \
	__pragma( warning( disable : WARN_THREAD_UNSAFE ) ) \
	static FieldType field(__VA_ARGS__); \
	__pragma( warning( default : WARN_THREAD_UNSAFE ) ) \
	return field

#else // AI_REVIEWER_CHECK defined

#define GET_FIELD_NO_ARGS(FieldType) \
	static FieldType field; \
	return field

#define GET_FIELD(FieldType, ...) \
	static FieldType field(__VA_ARGS__); \
	return field
#endif // AI_REVIEWER_CHECK

IComparator& Controller::getComparator() {
	GET_FIELD_NO_ARGS(Comparator);
}

IFontEngine& Controller::getFontEngine(const ISymSettings &ss_) const {
	GET_FIELD(FontEngine, *this, ss_);
}

IMatchEngine& Controller::getMatchEngine(const ISettings &cfg_) {
	GET_FIELD(MatchEngine, cfg_, getFontEngine(cfg_.getSS()), *cmP);
}

ITransformer& Controller::getTransformer(const ISettings &cfg_) {
	GET_FIELD(Transformer, *this, cfg_, getMatchEngine(cfg_),
			  (IBasicImgData&)ControlPanelActions::getImg());
}

#undef GET_FIELD_NO_ARGS
#undef GET_FIELD

#endif // UNIT_TESTING not defined
