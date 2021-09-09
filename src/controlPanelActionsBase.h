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

#ifndef H_CONTROL_PANEL_ACTIONS_BASE
#define H_CONTROL_PANEL_ACTIONS_BASE

#include "misc.h"

#pragma warning(push, 0)

#include <string>

#ifdef UNIT_TESTING
#include <opencv2/core/core.hpp>
#endif  // UNIT_TESTING defined

#pragma warning(pop)

namespace pic2sym {

/// Interface defining the actions triggered by the controls from Control Panel
class IControlPanelActions /*abstract*/ {
 public:
  /**
  Overwriting IMatchSettings with the content of 'initMatchSettings.cfg'
  @throw domain_error for obsolete / invalid file from replaceByUserDefaults()

  Exception to be only reported, not handled
  */
  virtual void restoreUserDefaultMatchSettings() noexcept(!UT) = 0;

  /// Saving current IMatchSettings to 'initMatchSettings.cfg'
  virtual void setUserDefaultMatchSettings() const noexcept = 0;

  /**
  Updating the Settings object. Rewriting the source file if it contains
  older versions of some classes.

  @throw SymsChangeInterrupted when the user aborts the operation
  */
  virtual bool loadSettings(const std::string& from = "") = 0;

  /// Saving the Settings object
  virtual void saveSettings() const noexcept = 0;

  /// Needed to restore encoding index
  virtual unsigned getFontEncodingIdx() const noexcept = 0;

  /**
  Sets an image to be transformed.
  @param imgPath the image to be set
  @param silent when true, it doesn't show popup windows if the image is not
  valid

  @return false if the image cannot be set
  */
  virtual bool newImage(const std::string& imgPath,
                        bool silent = false) noexcept = 0;

#ifdef UNIT_TESTING  // Method available only in Unit Testing mode
  /// Provide directly a matrix instead of an image
  virtual bool newImage(const cv::Mat& imgMat) noexcept = 0;
#endif  // UNIT_TESTING defined

  /// When unable to process a font type, invalidate it completely
  virtual void invalidateFont() noexcept = 0;

  /// @throw SymsChangeInterrupted when the user aborts the operation
  virtual void newFontFamily(const std::string& fontFile, unsigned fontSz) = 0;

  /// @throw SymsChangeInterrupted when the user aborts the operation
  virtual void newFontEncoding(int encodingIdx) = 0;

#ifdef UNIT_TESTING  // Method available only in Unit Testing mode
  /// @throw SymsChangeInterrupted when the user aborts the operation
  virtual bool newFontEncoding(const std::string& encName) = 0;
#endif  // UNIT_TESTING defined

  /// @throw SymsChangeInterrupted when the user aborts the operation
  virtual void newFontSize(int fontSz) = 0;
  virtual void newSymsBatchSize(int symsBatchSz) noexcept = 0;
  virtual void newStructuralSimilarityFactor(double k) noexcept = 0;
  virtual void newCorrelationFactor(double k) noexcept = 0;
  virtual void newUnderGlyphCorrectnessFactor(double k) noexcept = 0;
  virtual void newGlyphEdgeCorrectnessFactor(double k) noexcept = 0;
  virtual void newAsideGlyphCorrectnessFactor(double k) noexcept = 0;
  virtual void newContrastFactor(double k) noexcept = 0;
  virtual void newGravitationalSmoothnessFactor(double k) noexcept = 0;
  virtual void newDirectionalSmoothnessFactor(double k) noexcept = 0;
  virtual void newGlyphWeightFactor(double k) noexcept = 0;
  virtual void newThreshold4BlanksFactor(unsigned t) noexcept = 0;
  virtual void newHmaxSyms(int maxSyms) noexcept = 0;
  virtual void newVmaxSyms(int maxSyms) noexcept = 0;

  /**
  Sets the result mode:
  - approximations only (actual result) - patches become symbols, with no
  cosmeticizing.
  - hybrid (cosmeticized result) - for displaying approximations blended with a
  blurred version of the original. The better an approximation, the fainter the
  hint background

  @param hybrid boolean: when true, establishes the cosmeticized mode; otherwise
  leaves the actual result as it is
  */
  virtual void setResultMode(bool hybrid) noexcept = 0;

  /**
  Approximates an image based on current settings

  @param durationS if not nullptr, it will return the duration of the
  transformation (when successful)

  @return false if the transformation cannot be started; true otherwise (even
  when the transformation is canceled and the result is just a draft)
  */
  virtual bool performTransformation(double* durationS = nullptr) noexcept = 0;

  virtual void showAboutDlg(const std::string& title,
                            const std::wstring& content) noexcept = 0;

  virtual void showInstructionsDlg(const std::string& title,
                                   const std::wstring& content) noexcept = 0;

  virtual ~IControlPanelActions() noexcept = 0 {}
};

}  // namespace pic2sym

#endif  // H_CONTROL_PANEL_ACTIONS_BASE
