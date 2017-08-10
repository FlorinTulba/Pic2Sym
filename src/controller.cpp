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
#include "symSettings.h"
#include "matchEngine.h"
#include "matchParamsBase.h"
#include "bestMatchBase.h"
#include "matchAssessment.h"
#include "views.h"
#include "img.h"
#include "transform.h"
#include "preselectManager.h"
#include "controlPanel.h"
#include "controlPanelActions.h"
#include "misc.h"

#pragma warning ( push, 0 )

#include <sstream>

#pragma warning ( pop )

using namespace std;

std::shared_ptr<const IUpdateSymSettings> Controller::getUpdateSymSettings() const {
	return updateSymSettings;
}

std::shared_ptr<const IGlyphsProgressTracker> Controller::getGlyphsProgressTracker() const {
	return glyphsProgressTracker;
}

std::shared_ptr<IPicTransformProgressTracker> Controller::getPicTransformProgressTracker() {
	return picTransformProgressTracker;
}

std::shared_ptr<const IPresentCmap> Controller::getPresentCmap() const {
	return presentCmap;
}

std::shared_ptr<const ISelectSymbols> Controller::getSelectSymbols() const {
	return selectSymbols;
}

std::shared_ptr<IControlPanelActions> Controller::getControlPanelActions() {
	return controlPanelActions;
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

Comparator& Controller::getComparator() {
	GET_FIELD_NO_ARGS(Comparator);
}

FontEngine& Controller::getFontEngine(const SymSettings &ss_) const {
	GET_FIELD(FontEngine, *this, ss_);
}

MatchEngine& Controller::getMatchEngine(const ISettings &cfg_) {
	GET_FIELD(MatchEngine, cfg_, getFontEngine(cfg_.getSS()), cmP);
}

Transformer& Controller::getTransformer(const ISettings &cfg_) {
	GET_FIELD(Transformer, *this, cfg_, getMatchEngine(cfg_), ControlPanelActions::getImg());
}

PreselManager& Controller::getPreselManager(const ISettings &cfg_) {
	GET_FIELD(PreselManager, getMatchEngine(cfg_), getTransformer(cfg_));
}

#undef GET_FIELD_NO_ARGS
#undef GET_FIELD

#endif // UNIT_TESTING not defined
