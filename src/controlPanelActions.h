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

#ifndef H_CONTROL_PANEL_ACTIONS
#define H_CONTROL_PANEL_ACTIONS

#include "controlPanelActionsBase.h"

// Forward declarations
struct ISettingsRW;
struct IFontEngine;
class MatchAssessor;
struct ITransformer;
class Img;
struct ICmapInspect;
struct IComparator;
struct IControlPanel;

/// Implementation for the actions triggered by the controls from Control Panel
class ControlPanelActions : public IControlPanelActions {
protected:
	IController &ctrler;
	ISettingsRW &cfg;
	IFontEngine &fe;
	MatchAssessor &ma;
	ITransformer &t;
	Img &img;			///< original image to process after resizing
	IComparator &comp;	///< view for comparing original & result
	IControlPanel &cp;	///< the configuration view
	const std::uniquePtr<ICmapInspect> &pCmi;	///< viewer of the Cmap

	// Validation flags
	bool imageOk = false;		///< is there an image to be transformed (not set yet, so false)
	bool fontFamilyOk = false;	///< is there a symbol set available (not set yet, so false)

	/// Reports uncorrected settings when visualizing the cmap or while executing transform command.
	/// Cmap visualization can ignore image-related errors by setting 'imageRequired' to false.
	bool validState(bool imageRequired = true) const;

	// Next 3 protected methods do the ground work for their public correspondent methods
	bool _newFontFamily(const std::stringType &fontFile, bool forceUpdate = false);
	bool _newFontEncoding(const std::stringType &encName, bool forceUpdate = false);
	bool _newFontSize(int fontSz, bool forceUpdate = false);

public:
	static Img& getImg();
	IControlPanel& getControlPanel(ISettingsRW &cfg_);

	ControlPanelActions(IController &ctrler_, ISettingsRW &cfg_,
						IFontEngine &fe_, const MatchAssessor &ma_, ITransformer &t_,
						IComparator &comp_, const std::uniquePtr<ICmapInspect> &pCmi_);

	void operator=(const ControlPanelActions&) = delete;

	/// overwriting IMatchSettings with the content of 'initMatchSettings.cfg'
	void restoreUserDefaultMatchSettings() override;
	void setUserDefaultMatchSettings() const override; ///< saving current IMatchSettings to 'initMatchSettings.cfg'

	bool loadSettings(const std::stringType &from = "") override;	///< updating the ISettingsRW object
	void saveSettings() const override;	///< saving the ISettingsRW object

	unsigned getFontEncodingIdx() const override; ///< needed to restore encoding index

	/**
	Sets an image to be transformed.
	@param imgPath the image to be set
	@param silent when true, it doesn't show popup windows if the image is not valid

	@return false if the image cannot be set
	*/
	bool newImage(const std::stringType &imgPath, bool silent = false) override;

#ifdef UNIT_TESTING
	// Method available only in Unit Testing mode
	bool newImage(const cv::Mat &imgMat) override;	///< Provide directly a matrix instead of an image
#endif // UNIT_TESTING defined

	void invalidateFont() override;	///< When unable to process a font type, invalidate it completely
	void newFontFamily(const std::stringType &fontFile) override;
	void newFontEncoding(int encodingIdx) override;
#ifdef UNIT_TESTING
	bool newFontEncoding(const std::stringType &encName) override;
#endif // UNIT_TESTING defined
	void newFontSize(int fontSz) override;
	void newSymsBatchSize(int symsBatchSz) override;
	void newStructuralSimilarityFactor(double k) override;
	void newUnderGlyphCorrectnessFactor(double k) override;
	void newGlyphEdgeCorrectnessFactor(double k) override;
	void newAsideGlyphCorrectnessFactor(double k) override;
	void newContrastFactor(double k) override;
	void newGravitationalSmoothnessFactor(double k) override;
	void newDirectionalSmoothnessFactor(double k) override;
	void newGlyphWeightFactor(double k) override;
	void newThreshold4BlanksFactor(unsigned t) override;
	void newHmaxSyms(int maxSyms) override;
	void newVmaxSyms(int maxSyms) override;

	/**
	Sets the result mode:
	- approximations only (actual result) - patches become symbols, with no cosmeticizing.
	- hybrid (cosmeticized result) - for displaying approximations blended with a blurred version of the original. The better an approximation, the fainter the hint background

	@param hybrid boolean: when true, establishes the cosmeticized mode; otherwise leaves the actual result as it is
	*/
	void setResultMode(bool hybrid) override;

	/**
	Approximates an image based on current settings

	@param durationS if not nullptr, it will return the duration of the transformation (when successful)

	@return false if the transformation cannot be started; true otherwise (even when the transformation is canceled and the result is just a draft)
	*/
	bool performTransformation(double *durationS = nullptr) override;

	void showAboutDlg(const std::stringType &title, const std::wstringType &content) override;
	void showInstructionsDlg(const std::stringType &title, const std::wstringType &content) override;
};

#endif // H_CONTROL_PANEL_ACTIONS
