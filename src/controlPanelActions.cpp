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

#include "controlPanelActions.h"
#include "controlPanel.h"
#include "matchAssessment.h"
#include "transformBase.h"
#include "fontEngineBase.h"
#include "dlgs.h"
#include "misc.h"
#include "settings.h"
#include "symSettings.h"
#include "imgSettings.h"
#include "matchSettings.h"
#include "tinySymsProvider.h"
#include "symsLoadingFailure.h"
#include "comparatorBase.h"
#include "cmapInspectBase.h"
#include "img.h"

#pragma warning ( push, 0 )

#include <Windows.h>

#include <iostream>
#include <sstream>
#include <fstream>

#include <opencv2/core/core.hpp>

#include "boost_filesystem_operations.h"

#ifndef AI_REVIEWER_CHECK
#	include <boost/archive/binary_oarchive.hpp>
#	include <boost/archive/binary_iarchive.hpp>
#endif // AI_REVIEWER_CHECK not defined

#pragma warning ( pop )

using namespace std;
using namespace cv;
using namespace boost::filesystem;
using namespace boost::archive;

ControlPanelActions::ControlPanelActions(IController &ctrler_, ISettingsRW &cfg_,
										 IFontEngine &fe_, const MatchAssessor &ma_, ITransformer &t_,
										 IComparator &comp_, const std::uniquePtr<ICmapInspect> &pCmi_) :
	ctrler(ctrler_), cfg(cfg_), fe(fe_),
	ma(const_cast<MatchAssessor&>(ma_)), // match aspects might get enabled/disabled by the corresponding sliders 
	t(t_), img(getImg()),
	comp(comp_), cp(getControlPanel(cfg_)), pCmi(pCmi_) {}

bool ControlPanelActions::validState(bool imageRequired/* = true*/) const {
	const bool noEnabledMatchAspects = (ma.enabledMatchAspectsCount() == 0ULL);
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

Img& ControlPanelActions::getImg() {
	GET_FIELD_NO_ARGS(Img);
}

IControlPanel& ControlPanelActions::getControlPanel(ISettingsRW &cfg_) {
	GET_FIELD(ControlPanel, *this, cfg_);
}

#undef GET_FIELD_NO_ARGS
#undef GET_FIELD

#endif // UNIT_TESTING not defined

void ControlPanelActions::restoreUserDefaultMatchSettings() {
	extern const cv::String ControlPanel_restoreDefaultsLabel;
	const uniquePtr<const ActionPermit> permit = cp.actionDemand(ControlPanel_restoreDefaultsLabel);
	if(nullptr == permit)
		return;

#ifndef UNIT_TESTING
	cfg.refMS().replaceByUserDefaults();
#endif // UNIT_TESTING not defined

	cp.updateMatchSettings(cfg.getMS());
	ma.updateEnabledMatchAspectsCount();
}

void ControlPanelActions::setUserDefaultMatchSettings() const {
	extern const cv::String ControlPanel_saveAsDefaultsLabel;
	const uniquePtr<const ActionPermit> permit = cp.actionDemand(ControlPanel_saveAsDefaultsLabel);
	if(nullptr == permit)
		return;

#ifndef UNIT_TESTING
	cfg.refMS().saveAsUserDefaults();
#endif // UNIT_TESTING not defined
}

bool ControlPanelActions::loadSettings(const stringType &from/* = ""*/) {
	extern const cv::String ControlPanel_loadSettingsLabel;
	const uniquePtr<const ActionPermit> permit = cp.actionDemand(ControlPanel_loadSettingsLabel);
	if(nullptr == permit)
		return false;

	stringType sourceFile;
	if(!from.empty()) {
		sourceFile = from;

	} else { // prompting the user for the file to be loaded
#pragma warning ( disable : WARN_THREAD_UNSAFE )
		static SettingsSelector ss; // loader
#pragma warning ( default : WARN_THREAD_UNSAFE )

		if(!ss.promptForUserChoice())
			return false;

		sourceFile = ss.selection();
	}

	const uniquePtr<ISymSettings> prevSymSettings = cfg.getSS().clone(); // keep a copy of old SymSettings
	cout<<"Loading settings from '"<<sourceFile<<'\''<<endl;

#pragma warning ( disable : WARN_SEH_NOT_CAUGHT )
	try {
		ifstream ifs(sourceFile, ios::binary);
		binary_iarchive ia(ifs);
		ia >> dynamic_cast<Settings&>(cfg);
	} catch(...) {
		cerr<<"Couldn't load these settings"<<endl;
		return false;
	}
#pragma warning ( default : WARN_SEH_NOT_CAUGHT )

	cp.updateMatchSettings(cfg.getMS());
	ma.updateEnabledMatchAspectsCount();
	cp.updateImgSettings(cfg.getIS());

	if(dynamic_cast<const SymSettings&>(*prevSymSettings)
			== dynamic_cast<const SymSettings&>(cfg.getSS()))
		return true;

	bool fontFileChanged = false, encodingChanged = false;
	const stringType newEncName = cfg.getSS().getEncoding();
	if(prevSymSettings->getFontFile().compare(cfg.getSS().getFontFile()) != 0) {
		_newFontFamily(cfg.getSS().getFontFile(), true);
		fontFileChanged = true;
	}
	if((!fontFileChanged && prevSymSettings->getEncoding().compare(newEncName) != 0) ||
	   (fontFileChanged && cfg.getSS().getEncoding().compare(newEncName) != 0)) {
		_newFontEncoding(newEncName, true);
		encodingChanged = true;
	}

	if(prevSymSettings->getFontSz() != cfg.getSS().getFontSz()) {
		if(fontFileChanged || encodingChanged) {
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
	} catch(TinySymsLoadingFailure &tslf) {
		invalidateFont();
		tslf.informUser("Couldn't load the tiny versions of the font pointed by this settings file!");
		return false;
	} catch(NormalSymsLoadingFailure &nslf) {
		invalidateFont();
		nslf.informUser("Couldn't load the normal versions of the font pointed by this settings file!");
		return false;
	}

	return true;
}

void ControlPanelActions::saveSettings() const {
	extern const cv::String ControlPanel_saveSettingsLabel;
	const uniquePtr<const ActionPermit> permit = cp.actionDemand(ControlPanel_saveSettingsLabel);
	if(nullptr == permit)
		return;

	if(!cfg.getSS().initialized()) {
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
		oa << dynamic_cast<const Settings&>(cfg);
	} catch(...) {
		cerr<<"Couldn't save current settings"<<endl;
		return;
	}
#pragma warning ( default : WARN_SEH_NOT_CAUGHT )
}

unsigned ControlPanelActions::getFontEncodingIdx() const {
	if(fontFamilyOk) {
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

bool ControlPanelActions::newImage(const stringType &imgPath, bool silent/* = false*/) {
	extern const cv::String ControlPanel_selectImgLabel;
	const uniquePtr<const ActionPermit> permit = cp.actionDemand(ControlPanel_selectImgLabel);
	if(nullptr == permit)
		return false;

	if(img.absPath().compare(absolute(imgPath)) == 0)
		return true; // same image

	if(!img.reset(imgPath)) {
		if(!silent) {
			ostringstream oss;
			oss<<"Invalid image file: '"<<imgPath<<'\'';
			errMsg(oss.str());
		}
		return false;
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
	return true;
}

void ControlPanelActions::invalidateFont() {
	fontFamilyOk = false;

	cp.updateEncodingsCount(1U);
	if(pCmi)
		pCmi->clear();

	cfg.refSS().reset();
	fe.invalidateFont();
}

bool ControlPanelActions::_newFontFamily(const stringType &fontFile, bool forceUpdate/* = false*/) {
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

		ctrler.ensureExistenceCmapInspect();
		assert(pCmi);
	}

	return true;
}

void ControlPanelActions::newFontFamily(const stringType &fontFile) {
	extern const cv::String ControlPanel_selectFontLabel;
	const uniquePtr<const ActionPermit> permit = cp.actionDemand(ControlPanel_selectFontLabel);
	if(nullptr == permit)
		return;

	if(!_newFontFamily(fontFile))
		return;

	try {
		ctrler.symbolsChanged();
	} catch(TinySymsLoadingFailure &tslf) {
		invalidateFont();
		tslf.informUser("Couldn't load the tiny versions of the newly selected font family!");
	} catch(NormalSymsLoadingFailure &nslf) {
		invalidateFont();
		nslf.informUser("Couldn't load the normal versions of the newly selected font family!");
	}
}

void ControlPanelActions::newFontEncoding(int encodingIdx) {
	// Ignore call if no font yet, or just 1 encoding,
	// or if the required hack (mentioned in 'ui.h') provoked this call
	if(!fontFamilyOk || fe.uniqueEncodings() <= 1U || cp.encMaxHack())
		return;

	unsigned currEncIdx;
	fe.getEncoding(&currEncIdx);
	if(currEncIdx == (unsigned)encodingIdx)
		return;

	extern const cv::String ControlPanel_encodingTrName;
	const uniquePtr<const ActionPermit> permit = cp.actionDemand(ControlPanel_encodingTrName);
	if(nullptr == permit)
		return;

	fe.setNthUniqueEncoding((unsigned)encodingIdx);

	try {
		ctrler.symbolsChanged();
	} catch(TinySymsLoadingFailure &tslf) {
		invalidateFont();
		tslf.informUser("Couldn't load the tiny versions of the font whose encoding has been updated!");
	} catch(NormalSymsLoadingFailure &nslf) {
		invalidateFont();
		nslf.informUser("Couldn't load the normal versions of the font whose encoding has been updated!");
	}
}

bool ControlPanelActions::_newFontEncoding(const stringType &encName, bool forceUpdate/* = false*/) {
	return fe.setEncoding(encName, forceUpdate);
}

#ifdef UNIT_TESTING
bool ControlPanelActions::newFontEncoding(const stringType &encName) {
	bool result = _newFontEncoding(encName);
	if(result) {
		try {
			ctrler.symbolsChanged();
		} catch(TinySymsLoadingFailure &tslf) {
			invalidateFont();
			tslf.informUser("Couldn't load the tiny versions of the font whose encoding has been updated!");
			return false;
		} catch(NormalSymsLoadingFailure &nslf) {
			invalidateFont();
			nslf.informUser("Couldn't load the normal versions of the font whose encoding has been updated!");
			return false;
		}
	}

	return result;
}
#endif // UNIT_TESTING defined

bool ControlPanelActions::_newFontSize(int fontSz, bool forceUpdate/* = false*/) {
	extern const cv::String ControlPanel_fontSzTrName;
	extern const unsigned Settings_MIN_FONT_SIZE;

	if(!ISettings::isFontSizeOk((unsigned)fontSz)) {
		ostringstream oss;
		oss<<"Invalid font size. Please set at least "<<Settings_MIN_FONT_SIZE<<'.';
		cp.restoreSliderValue(ControlPanel_fontSzTrName, oss.str());
		return false;
	}

	if((unsigned)fontSz == cfg.getSS().getFontSz() && !forceUpdate)
		return false;

	cfg.refSS().setFontSz((unsigned)fontSz);

	if(!fontFamilyOk) {
		if(pCmi)
			pCmi->clear();
		return false;
	}

	pCmi->updateGrid();

	return true;
}

void ControlPanelActions::newFontSize(int fontSz) {
	extern const cv::String ControlPanel_fontSzTrName;
	const uniquePtr<const ActionPermit> permit = cp.actionDemand(ControlPanel_fontSzTrName);
	if(nullptr == permit)
		return;

	if(!_newFontSize(fontSz))
		return;

	try {
		ctrler.symbolsChanged();
	} catch(TinySymsLoadingFailure &tslf) {
		invalidateFont();
		tslf.informUser("Couldn't load the tiny versions of the font whose size has been updated!");
	} catch(NormalSymsLoadingFailure &nslf) {
		invalidateFont();
		nslf.informUser("Couldn't load the requested size versions of the fonts!");
	}
}

void ControlPanelActions::newSymsBatchSize(int symsBatchSz) {
	extern const cv::String ControlPanel_symsBatchSzTrName;
	const uniquePtr<const ActionPermit> permit = cp.actionDemand(ControlPanel_symsBatchSzTrName);
	if(nullptr == permit)
		return;

	t.setSymsBatchSize(symsBatchSz);
}

void ControlPanelActions::newHmaxSyms(int maxSymbols) {
	extern const cv::String ControlPanel_outWTrName;
	extern const unsigned Settings_MIN_H_SYMS;

	const uniquePtr<const ActionPermit> permit = cp.actionDemand(ControlPanel_outWTrName);
	if(nullptr == permit)
		return;

	if((unsigned)maxSymbols == cfg.getIS().getMaxHSyms()) // it's possible if the previous value was invalid
		return;

	if(!ISettings::isHmaxSymsOk((unsigned)maxSymbols)) {
		ostringstream oss;
		oss<<"Invalid max number of horizontal symbols. Please set at least "<<Settings_MIN_H_SYMS<<'.';
		cp.restoreSliderValue(ControlPanel_outWTrName, oss.str());
		return;
	}

	cfg.refIS().setMaxHSyms((unsigned)maxSymbols);
}

void ControlPanelActions::newVmaxSyms(int maxSymbols) {
	extern const cv::String ControlPanel_outHTrName;
	extern const unsigned Settings_MIN_V_SYMS;

	const uniquePtr<const ActionPermit> permit = cp.actionDemand(ControlPanel_outHTrName);
	if(nullptr == permit)
		return;

	if((unsigned)maxSymbols == cfg.getIS().getMaxVSyms()) // it's possible if the previous value was invalid
		return;

	if(!ISettings::isVmaxSymsOk((unsigned)maxSymbols)) {
		ostringstream oss;
		oss<<"Invalid max number of vertical symbols. Please set at least "<<Settings_MIN_V_SYMS<<'.';
		cp.restoreSliderValue(ControlPanel_outHTrName, oss.str());
		return;
	}

	cfg.refIS().setMaxVSyms((unsigned)maxSymbols);
}

void ControlPanelActions::setResultMode(bool hybrid) {
	extern const cv::String ControlPanel_hybridResultTrName;
	const uniquePtr<const ActionPermit> permit = cp.actionDemand(ControlPanel_hybridResultTrName);
	if(nullptr == permit)
		return;

	cfg.refMS().setResultMode(hybrid);
}

void ControlPanelActions::newThreshold4BlanksFactor(unsigned threshold) {
	extern const cv::String ControlPanel_thresh4BlanksTrName;
	const uniquePtr<const ActionPermit> permit = cp.actionDemand(ControlPanel_thresh4BlanksTrName);
	if(nullptr == permit)
		return;

	cfg.refMS().setBlankThreshold(threshold);
}

#define UPDATE_MATCH_ASPECT_VALUE(AspectName, NewValue) \
	const double PrevVal = cfg.getMS().get_k##AspectName(); \
	if(NewValue != PrevVal) { \
		cfg.refMS().set_k##AspectName(NewValue); \
		if(PrevVal == 0.) { /* just enabled this aspect */ \
			ma.newlyEnabledMatchAspect(); \
		} else if(NewValue == 0.) { /* just disabled this aspect */ \
			ma.newlyDisabledMatchAspect(); \
		} \
	}

void ControlPanelActions::newContrastFactor(double k) {
	extern const cv::String ControlPanel_moreContrastTrName;
	const uniquePtr<const ActionPermit> permit = cp.actionDemand(ControlPanel_moreContrastTrName);
	if(nullptr == permit)
		return;

	UPDATE_MATCH_ASPECT_VALUE(Contrast, k);
}

void ControlPanelActions::newStructuralSimilarityFactor(double k) {
	extern const cv::String ControlPanel_structuralSimTrName;
	const uniquePtr<const ActionPermit> permit = cp.actionDemand(ControlPanel_structuralSimTrName);
	if(nullptr == permit)
		return;

	UPDATE_MATCH_ASPECT_VALUE(Ssim, k);
}

void ControlPanelActions::newUnderGlyphCorrectnessFactor(double k) {
	extern const cv::String ControlPanel_underGlyphCorrectnessTrName;
	const uniquePtr<const ActionPermit> permit = cp.actionDemand(ControlPanel_underGlyphCorrectnessTrName);
	if(nullptr == permit)
		return;

	UPDATE_MATCH_ASPECT_VALUE(SdevFg, k);
}

void ControlPanelActions::newAsideGlyphCorrectnessFactor(double k) {
	extern const cv::String ControlPanel_asideGlyphCorrectnessTrName;
	const uniquePtr<const ActionPermit> permit = cp.actionDemand(ControlPanel_asideGlyphCorrectnessTrName);
	if(nullptr == permit)
		return;

	UPDATE_MATCH_ASPECT_VALUE(SdevBg, k);
}

void ControlPanelActions::newGlyphEdgeCorrectnessFactor(double k) {
	extern const cv::String ControlPanel_glyphEdgeCorrectnessTrName;
	const uniquePtr<const ActionPermit> permit = cp.actionDemand(ControlPanel_glyphEdgeCorrectnessTrName);
	if(nullptr == permit)
		return;

	UPDATE_MATCH_ASPECT_VALUE(SdevEdge, k);
}

void ControlPanelActions::newDirectionalSmoothnessFactor(double k) {
	extern const cv::String ControlPanel_directionTrName;
	const uniquePtr<const ActionPermit> permit = cp.actionDemand(ControlPanel_directionTrName);
	if(nullptr == permit)
		return;

	UPDATE_MATCH_ASPECT_VALUE(CosAngleMCs, k);
}

void ControlPanelActions::newGravitationalSmoothnessFactor(double k) {
	extern const cv::String ControlPanel_gravityTrName;
	const uniquePtr<const ActionPermit> permit = cp.actionDemand(ControlPanel_gravityTrName);
	if(nullptr == permit)
		return;

	UPDATE_MATCH_ASPECT_VALUE(MCsOffset, k);
}

void ControlPanelActions::newGlyphWeightFactor(double k) {
	extern const cv::String ControlPanel_largerSymTrName;
	const uniquePtr<const ActionPermit> permit = cp.actionDemand(ControlPanel_largerSymTrName);
	if(nullptr == permit)
		return;

	UPDATE_MATCH_ASPECT_VALUE(SymDensity, k);
}

#undef UPDATE_MATCH_ASPECT_VALUE

bool ControlPanelActions::performTransformation(double *durationS/* = nullptr*/) {
	extern const cv::String ControlPanel_transformImgLabel;
	const uniquePtr<const ActionPermit> permit = cp.actionDemand(ControlPanel_transformImgLabel);
	if(nullptr == permit)
		return false;

	if(!validState())
		return false;

	t.run();

	if(nullptr != durationS)
		*durationS = t.duration();

	return true;
}

void ControlPanelActions::showAboutDlg(const stringType &title, const wstringType &content) {
	extern const String ControlPanel_aboutLabel;
	const uniquePtr<const ActionPermit> permit = cp.actionDemand(ControlPanel_aboutLabel);
	if(nullptr == permit)
		return;

	MessageBox(nullptr, content.c_str(),
			   str2wstr(title).c_str(), MB_ICONINFORMATION | MB_OK | MB_TASKMODAL | MB_SETFOREGROUND);
}

void ControlPanelActions::showInstructionsDlg(const stringType &title, const wstringType &content) {
	extern const String ControlPanel_instructionsLabel;
	const uniquePtr<const ActionPermit> permit = cp.actionDemand(ControlPanel_instructionsLabel);
	if(nullptr == permit)
		return;

	MessageBox(nullptr, content.c_str(),
			   str2wstr(title).c_str(), MB_ICONINFORMATION | MB_OK | MB_TASKMODAL | MB_SETFOREGROUND);
}
