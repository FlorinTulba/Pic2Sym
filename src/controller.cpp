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

#include "controller.h"
#include "preselectManager.h"
#include "matchAssessment.h"
#include "matchParams.h"
#include "misc.h"
#include "dlgs.h"
#include "settings.h"
#include "controlPanel.h"
#include "match.h"

#pragma warning ( push, 0 )

#include <Windows.h>
#include <sstream>

#include <boost/filesystem/operations.hpp>

#pragma warning ( pop )

#ifndef UNIT_TESTING
#	include "matchSettingsManip.h"
#endif

using namespace std;
using namespace boost::filesystem;
using namespace boost::archive;

extern const unsigned Settings_MAX_THRESHOLD_FOR_BLANKS;
extern const unsigned Settings_MIN_H_SYMS;
extern const unsigned Settings_MAX_H_SYMS;
extern const unsigned Settings_MIN_V_SYMS;
extern const unsigned Settings_MAX_V_SYMS;
extern const unsigned Settings_MIN_FONT_SIZE;
extern const unsigned Settings_MAX_FONT_SIZE;
extern const unsigned Settings_DEF_FONT_SIZE;

bool Settings::isBlanksThresholdOk(unsigned t) {
	return t < Settings_MAX_THRESHOLD_FOR_BLANKS;
}

bool Settings::isHmaxSymsOk(unsigned syms) {
	return syms>=Settings_MIN_H_SYMS && syms<=Settings_MAX_H_SYMS;
}

bool Settings::isVmaxSymsOk(unsigned syms) {
	return syms>=Settings_MIN_V_SYMS && syms<=Settings_MAX_V_SYMS;
}

bool Settings::isFontSizeOk(unsigned fs) {
	return fs>=Settings_MIN_FONT_SIZE && fs<=Settings_MAX_FONT_SIZE;
}

Settings::Settings(const MatchSettings &ms_) :
	ss(Settings_DEF_FONT_SIZE), is(Settings_MAX_H_SYMS, Settings_MAX_V_SYMS), ms(ms_) {}

Settings::Settings() :
	ss(Settings_DEF_FONT_SIZE), is(Settings_MAX_H_SYMS, Settings_MAX_V_SYMS), ms() {}

bool Controller::validState(bool imageRequired/* = true*/) const {
	const bool noEnabledMatchAspects = (me.assessor().enabledMatchAspectsCount() == 0ULL);
	if((imageOk || !imageRequired) && fontFamilyOk && !noEnabledMatchAspects)
		return true;

	ostringstream oss;
	oss<<"The problems are:"<<endl<<endl;
	if(imageRequired && !imageOk)
		oss<<"- no image to transform"<<endl;
	if(!fontFamilyOk)
		oss<<"- no font family to use during transformation"<<endl;
	if(noEnabledMatchAspects)
		oss<<"- no enabled matching aspects to consider"<<endl;
	errMsg(oss.str(), "Please Correct these errors first!");
	return false;
}

void Controller::newImage(const string &imgPath) {
	extern const cv::String ControlPanel_selectImgLabel;
	const auto permit = cp.actionDemand(ControlPanel_selectImgLabel);
	if(nullptr == permit)
		return;

	if(img.absPath().compare(absolute(imgPath)) == 0)
		return; // same image

	if(!img.reset(imgPath)) {
		ostringstream oss;
		oss<<"Invalid image file: '"<<imgPath<<'\'';
		errMsg(oss.str());
		return;
	}

	ostringstream oss;
	oss<<"Pic2Sym on image: "<<img.absPath();
	comp.setTitle(oss.str());

	if(!imageOk) { // 1st image loaded
		comp.permitResize();
		imageOk = true;
	}

	const cv::Mat &orig = img.original();
	comp.setReference(orig); // displays the image
	comp.resize();
}

bool Controller::_newFontFamily(const string &fontFile, bool forceUpdate/* = false*/) {
	if(fe.fontFileName().compare(fontFile) == 0 && !forceUpdate)
		return false; // same font

	if(!fe.newFont(fontFile)) {
		ostringstream oss;
		oss<<"Invalid font file: '"<<fontFile<<'\'';
		errMsg(oss.str());
		return false;
	}

	cp.updateEncodingsCount(fe.uniqueEncodings());

	if(!fontFamilyOk) {
		fontFamilyOk = true;
		pCmi = std::make_shared<CmapInspect>(*this);
	}

	return true;
}

void Controller::newFontFamily(const string &fontFile) {
	extern const cv::String ControlPanel_selectFontLabel;
	const auto permit = cp.actionDemand(ControlPanel_selectFontLabel);
	if(nullptr == permit)
		return;

	if(!_newFontFamily(fontFile))
		return;

	symbolsChanged();
}

void Controller::selectedFontFile(const string &fName) const {
	cfg.ss.setFontFile(fName);
}

unsigned Controller::getFontEncodingIdx() const {
	if(!fontFamilyOk)
		THROW_WITH_CONST_MSG("Please setup a font before calling " __FUNCTION__, logic_error);

	unsigned currEncIdx;
	fe.getEncoding(&currEncIdx);

	return currEncIdx;
}

void Controller::newFontEncoding(int encodingIdx) {
	extern const cv::String ControlPanel_encodingTrName;
	const auto permit = cp.actionDemand(ControlPanel_encodingTrName);
	if(nullptr == permit)
		return;

	// Ignore call if no font yet, or just 1 encoding,
	// or if the required hack (mentioned in 'ui.h') provoked this call
	if(!fontFamilyOk || fe.uniqueEncodings() == 1U || cp.encMaxHack())
		return;
	
	unsigned currEncIdx;
	fe.getEncoding(&currEncIdx);
	if(currEncIdx == (unsigned)encodingIdx)
		return;

	fe.setNthUniqueEncoding((unsigned)encodingIdx);

	symbolsChanged();
}

bool Controller::_newFontEncoding(const string &encName, bool forceUpdate/* = false*/) {
	return fe.setEncoding(encName, forceUpdate);
}

bool Controller::newFontEncoding(const string &encName) {
	bool result = _newFontEncoding(encName);
	if(result)
		symbolsChanged();

	return result;
}

void Controller::selectedEncoding(const string &encName) const {
	cfg.ss.setEncoding(encName);
}

bool Controller::_newFontSize(int fontSz, bool forceUpdate/* = false*/) {
	extern const cv::String ControlPanel_fontSzTrName;
	if(!Settings::isFontSizeOk((unsigned)fontSz)) {
		ostringstream oss;
		oss<<"Invalid font size. Please set at least "<<Settings_MIN_FONT_SIZE<<'.';
		cp.restoreSliderValue(ControlPanel_fontSzTrName, oss.str());
		return false;
	}

	if((unsigned)fontSz == cfg.ss.getFontSz() && !forceUpdate)
		return false;

	if(!fontFamilyOk) {
		if((unsigned)fontSz != cfg.ss.getFontSz())
			cfg.ss.setFontSz((unsigned)fontSz);
		return false;
	}
		
	cfg.ss.setFontSz((unsigned)fontSz);
	pCmi->updateGrid();

	return true;
}

void Controller::newFontSize(int fontSz) {
	extern const cv::String ControlPanel_fontSzTrName;
	const auto permit = cp.actionDemand(ControlPanel_fontSzTrName);
	if(nullptr == permit)
		return;

	if(!_newFontSize(fontSz))
		return;

	symbolsChanged();
}

void Controller::newSymsBatchSize(int symsBatchSz) {
	extern const cv::String ControlPanel_symsBatchSzTrName;
	const auto permit = cp.actionDemand(ControlPanel_symsBatchSzTrName);
	if(nullptr == permit)
		return;

	t.setSymsBatchSize(symsBatchSz);
}

void Controller::newHmaxSyms(int maxSymbols) {
	extern const cv::String ControlPanel_outWTrName;
	const auto permit = cp.actionDemand(ControlPanel_outWTrName);
	if(nullptr == permit)
		return;

	if((unsigned)maxSymbols == cfg.is.getMaxHSyms()) // it's possible if the previous value was invalid
		return;

	if(!Settings::isHmaxSymsOk((unsigned)maxSymbols)) {
		ostringstream oss;
		oss<<"Invalid max number of horizontal symbols. Please set at least "<<Settings_MIN_H_SYMS<<'.';
		cp.restoreSliderValue(ControlPanel_outWTrName, oss.str());
		return;
	}

	cfg.is.setMaxHSyms((unsigned)maxSymbols);
}

void Controller::newVmaxSyms(int maxSymbols) {
	extern const cv::String ControlPanel_outHTrName;
	const auto permit = cp.actionDemand(ControlPanel_outHTrName);
	if(nullptr == permit)
		return;

	if((unsigned)maxSymbols == cfg.is.getMaxVSyms()) // it's possible if the previous value was invalid
		return;

	if(!Settings::isVmaxSymsOk((unsigned)maxSymbols)) {
		ostringstream oss;
		oss<<"Invalid max number of vertical symbols. Please set at least "<<Settings_MIN_V_SYMS<<'.';
		cp.restoreSliderValue(ControlPanel_outHTrName, oss.str());
		return;
	}

	cfg.is.setMaxVSyms((unsigned)maxSymbols);
}

void Controller::setResultMode(bool hybrid) {
	extern const cv::String ControlPanel_hybridResultTrName;
	const auto permit = cp.actionDemand(ControlPanel_hybridResultTrName);
	if(nullptr == permit)
		return;

	if(hybrid != cfg.ms.isHybridResult())
		cfg.ms.setResultMode(hybrid);
}

unsigned Controller::getFontSize() const {
	return cfg.ss.getFontSz();
}

void Controller::newThreshold4BlanksFactor(unsigned threshold) {
	extern const cv::String ControlPanel_thresh4BlanksTrName;
	const auto permit = cp.actionDemand(ControlPanel_thresh4BlanksTrName);
	if(nullptr == permit)
		return;

	if((unsigned)threshold != cfg.ms.getBlankThreshold())
		cfg.ms.setBlankThreshold(threshold);
}

#define UPDATE_MATCH_ASPECT_VALUE(AspectName, NewValue) \
	const double PrevVal = cfg.ms.get_k##AspectName(); \
	if(NewValue != PrevVal) { \
		cfg.ms.set_k##AspectName(NewValue); \
		if(PrevVal == 0.) { /* just enabled this aspect */ \
			me.matchAssessor.newlyEnabledMatchAspect(); \
		} else if(NewValue == 0.) { /* just disabled this aspect */ \
			me.matchAssessor.newlyDisabledMatchAspect(); \
		} \
	}

void Controller::newContrastFactor(double k) {
	extern const cv::String ControlPanel_moreContrastTrName;
	const auto permit = cp.actionDemand(ControlPanel_moreContrastTrName);
	if(nullptr == permit)
		return;

	UPDATE_MATCH_ASPECT_VALUE(Contrast, k);
}

void Controller::newStructuralSimilarityFactor(double k) {
	extern const cv::String ControlPanel_structuralSimTrName;
	const auto permit = cp.actionDemand(ControlPanel_structuralSimTrName);
	if(nullptr == permit)
		return;

	UPDATE_MATCH_ASPECT_VALUE(Ssim, k);
}

void Controller::newUnderGlyphCorrectnessFactor(double k) {
	extern const cv::String ControlPanel_underGlyphCorrectnessTrName;
	const auto permit = cp.actionDemand(ControlPanel_underGlyphCorrectnessTrName);
	if(nullptr == permit)
		return;

	UPDATE_MATCH_ASPECT_VALUE(SdevFg, k);
}

void Controller::newAsideGlyphCorrectnessFactor(double k) {
	extern const cv::String ControlPanel_asideGlyphCorrectnessTrName;
	const auto permit = cp.actionDemand(ControlPanel_asideGlyphCorrectnessTrName);
	if(nullptr == permit)
		return;

	UPDATE_MATCH_ASPECT_VALUE(SdevBg, k);
}

void Controller::newGlyphEdgeCorrectnessFactor(double k) {
	extern const cv::String ControlPanel_glyphEdgeCorrectnessTrName;
	const auto permit = cp.actionDemand(ControlPanel_glyphEdgeCorrectnessTrName);
	if(nullptr == permit)
		return;

	UPDATE_MATCH_ASPECT_VALUE(SdevEdge, k);
}

void Controller::newDirectionalSmoothnessFactor(double k) {
	extern const cv::String ControlPanel_directionTrName;
	const auto permit = cp.actionDemand(ControlPanel_directionTrName);
	if(nullptr == permit)
		return;

	UPDATE_MATCH_ASPECT_VALUE(CosAngleMCs, k);
}

void Controller::newGravitationalSmoothnessFactor(double k) {
	extern const cv::String ControlPanel_gravityTrName;
	const auto permit = cp.actionDemand(ControlPanel_gravityTrName);
	if(nullptr == permit)
		return;

	UPDATE_MATCH_ASPECT_VALUE(MCsOffset, k);
}

void Controller::newGlyphWeightFactor(double k) {
	extern const cv::String ControlPanel_largerSymTrName;
	const auto permit = cp.actionDemand(ControlPanel_largerSymTrName);
	if(nullptr == permit)
		return;

	UPDATE_MATCH_ASPECT_VALUE(SymDensity, k);
}

#undef UPDATE_MATCH_ASPECT_VALUE

bool Controller::updateResizedImg(std::shared_ptr<const ResizedImg> resizedImg_) {
	if(!resizedImg_)
		THROW_WITH_CONST_MSG("Provided nullptr param to " __FUNCTION__, invalid_argument);

	const bool result = !resizedImg || (*resizedImg != *resizedImg_);

	if(result)
		resizedImg = resizedImg_;
	comp.setReference(resizedImg->get());
	if(result)
		comp.resize();

	return result;
}

bool Controller::performTransformation() {
	extern const cv::String ControlPanel_transformImgLabel;
	const auto permit = cp.actionDemand(ControlPanel_transformImgLabel);
	if(nullptr == permit)
		return false;

	if(!validState())
		return false;

	t.run();
	return true;
}

void Controller::restoreUserDefaultMatchSettings() {
	extern const cv::String ControlPanel_restoreDefaultsLabel;
	const auto permit = cp.actionDemand(ControlPanel_restoreDefaultsLabel);
	if(nullptr == permit)
		return;

#ifndef UNIT_TESTING
	MatchSettingsManip::instance().loadUserDefaults(cfg.ms);
#endif
	cp.updateMatchSettings(cfg.ms);
	me.matchAssessor.updateEnabledMatchAspectsCount();
}

void Controller::setUserDefaultMatchSettings() const {
	extern const cv::String ControlPanel_saveAsDefaultsLabel;
	const auto permit = cp.actionDemand(ControlPanel_saveAsDefaultsLabel);
	if(nullptr == permit)
		return;

#ifndef UNIT_TESTING
	MatchSettingsManip::instance().saveUserDefaults(cfg.ms);
#endif
}

void Controller::loadSettings() {
	extern const cv::String ControlPanel_loadSettingsLabel;
	const auto permit = cp.actionDemand(ControlPanel_loadSettingsLabel);
	if(nullptr == permit)
		return;

#pragma warning ( disable : WARN_THREAD_UNSAFE )
	static SettingsSelector ss; // loader
#pragma warning ( default : WARN_THREAD_UNSAFE )

	if(!ss.promptForUserChoice())
		return;
	
	const SymSettings prevSymSettings(cfg.ss); // keep a copy of old SymSettings
	cout<<"Loading settings from '"<<ss.selection()<<'\''<<endl;

#pragma warning ( disable : WARN_SEH_NOT_CAUGHT )
	try {
		ifstream ifs(ss.selection(), ios::binary);
		binary_iarchive ia(ifs);
		ia>>cfg;
	} catch(...) {
		cerr<<"Couldn't load these settings"<<endl;
		return;
	}
#pragma warning ( default : WARN_SEH_NOT_CAUGHT )

	cp.updateMatchSettings(cfg.ms);
	me.matchAssessor.updateEnabledMatchAspectsCount();
	cp.updateImgSettings(cfg.is);

	if(prevSymSettings==cfg.ss)
		return;

	bool fontFileChanged = false, encodingChanged = false;
	const auto newEncName = cfg.ss.getEncoding();
	if(prevSymSettings.getFontFile().compare(cfg.ss.getFontFile()) != 0) {
		_newFontFamily(cfg.ss.getFontFile(), true);
		fontFileChanged = true;
	}
	if((!fontFileChanged && prevSymSettings.getEncoding().compare(newEncName) != 0) ||
			(fontFileChanged && cfg.ss.getEncoding().compare(newEncName) != 0)) {
		_newFontEncoding(newEncName, true);
		encodingChanged = true;
	}

	if(prevSymSettings.getFontSz() != cfg.ss.getFontSz()) {
		if(fontFileChanged || encodingChanged) {
			pCmi->updateGrid();
		} else {
			_newFontSize((int)cfg.ss.getFontSz(), true);
		}
	}

	unsigned currEncIdx;
	fe.getEncoding(&currEncIdx);
	cp.updateSymSettings(currEncIdx, cfg.ss.getFontSz());
	
	symbolsChanged();
}

void Controller::saveSettings() const {
	extern const cv::String ControlPanel_saveSettingsLabel;
	const auto permit = cp.actionDemand(ControlPanel_saveSettingsLabel);
	if(nullptr == permit)
		return;

	if(!cfg.ss.ready()) {
		warnMsg("There's no Font yet.\nSave settings only after selecting a font !");
		return;
	}

#pragma warning ( disable : WARN_THREAD_UNSAFE )
	static SettingsSelector ss(false); // saver
#pragma warning ( default : WARN_THREAD_UNSAFE )

	if(!ss.promptForUserChoice())
		return;

	cout<<"Saving settings to '"<<ss.selection()<<'\''<<endl;
	
#pragma warning ( disable : WARN_SEH_NOT_CAUGHT )
	try {
		ofstream ofs(ss.selection(), ios::binary);
		binary_oarchive oa(ofs);
		oa<<cfg;
	} catch(...) {
		cerr<<"Couldn't save current settings"<<endl;
		return;
	}
#pragma warning ( default : WARN_SEH_NOT_CAUGHT )
}

// Methods from below have different definitions for UnitTesting project
#ifndef UNIT_TESTING

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

Img& Controller::getImg() {
	GET_FIELD_NO_ARGS(Img);
}

Comparator& Controller::getComparator() {
	GET_FIELD_NO_ARGS(Comparator);
}

FontEngine& Controller::getFontEngine(const SymSettings &ss_) const {
	GET_FIELD(FontEngine, *this, ss_);
}

MatchEngine& Controller::getMatchEngine(const Settings &cfg_) const {
	GET_FIELD(MatchEngine, cfg_, getFontEngine(cfg_.ss));
}

Transformer& Controller::getTransformer(const Settings &cfg_) const {
	GET_FIELD(Transformer, *this, cfg_, getMatchEngine(cfg_), getImg());
}

PreselManager& Controller::getPreselManager(const Settings &cfg_) const {
	GET_FIELD(PreselManager, getMatchEngine(cfg_), getTransformer(cfg_));
}

ControlPanel& Controller::getControlPanel(Settings &cfg_) {
	GET_FIELD(ControlPanel, *this, cfg_);
}

#undef GET_FIELD_NO_ARGS
#undef GET_FIELD

#endif // UNIT_TESTING not defined
