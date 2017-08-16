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

#ifndef H_CONTROL_PANEL_ACTIONS_BASE
#define H_CONTROL_PANEL_ACTIONS_BASE

#include "controllerBase.h"

#pragma warning ( push, 0 )

#include <string>

#ifdef UNIT_TESTING
#include <opencv2/core/core.hpp>
#endif // UNIT_TESTING defined

#pragma warning ( pop )

/// Interface defining the actions triggered by the controls from Control Panel
struct IControlPanelActions /*abstract*/ {
	/// overwriting IMatchSettings with the content of 'initMatchSettings.cfg'
	virtual void restoreUserDefaultMatchSettings() = 0;
	virtual void setUserDefaultMatchSettings() const = 0; ///< saving current IMatchSettings to 'initMatchSettings.cfg'

	virtual bool loadSettings(const std::string &from = "") = 0;	///< updating the Settings object
	virtual void saveSettings() const = 0;	///< saving the Settings object

	virtual unsigned getFontEncodingIdx() const = 0; ///< needed to restore encoding index

	/**
	Sets an image to be transformed.
	@param imgPath the image to be set
	@param silent when true, it doesn't show popup windows if the image is not valid

	@return false if the image cannot be set
	*/
	virtual bool newImage(const std::string &imgPath, bool silent = false) = 0;

#ifdef UNIT_TESTING
	// Method available only in Unit Testing mode
	virtual bool newImage(const cv::Mat &imgMat) = 0;	///< Provide directly a matrix instead of an image
#endif // UNIT_TESTING defined

	virtual void invalidateFont() = 0;	///< When unable to process a font type, invalidate it completely
	virtual void newFontFamily(const std::string &fontFile) = 0;
	virtual void newFontEncoding(int encodingIdx) = 0;
	virtual bool newFontEncoding(const std::string &encName) = 0;
	virtual void newFontSize(int fontSz) = 0;
	virtual void newSymsBatchSize(int symsBatchSz) = 0;
	virtual void newStructuralSimilarityFactor(double k) = 0;
	virtual void newUnderGlyphCorrectnessFactor(double k) = 0;
	virtual void newGlyphEdgeCorrectnessFactor(double k) = 0;
	virtual void newAsideGlyphCorrectnessFactor(double k) = 0;
	virtual void newContrastFactor(double k) = 0;
	virtual void newGravitationalSmoothnessFactor(double k) = 0;
	virtual void newDirectionalSmoothnessFactor(double k) = 0;
	virtual void newGlyphWeightFactor(double k) = 0;
	virtual void newThreshold4BlanksFactor(unsigned t) = 0;
	virtual void newHmaxSyms(int maxSyms) = 0;
	virtual void newVmaxSyms(int maxSyms) = 0;

	/**
	Sets the result mode:
	- approximations only (actual result) - patches become symbols, with no cosmeticizing.
	- hybrid (cosmeticized result) - for displaying approximations blended with a blurred version of the original. The better an approximation, the fainter the hint background

	@param hybrid boolean: when true, establishes the cosmeticized mode; otherwise leaves the actual result as it is
	*/
	virtual void setResultMode(bool hybrid) = 0;

	/**
	Approximates an image based on current settings

	@param durationS if not nullptr, it will return the duration of the transformation (when successful)

	@return false if the transformation cannot be started; true otherwise (even when the transformation is canceled and the result is just a draft)
	*/
	virtual bool performTransformation(double *durationS = nullptr) = 0;

	virtual void showAboutDlg(const std::string &title, const std::wstring &content) = 0;
	virtual void showInstructionsDlg(const std::string &title, const std::wstring &content) = 0;

	virtual ~IControlPanelActions() = 0 {}
};

#endif // H_CONTROL_PANEL_ACTIONS_BASE
