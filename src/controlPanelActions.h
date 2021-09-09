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

#ifndef H_CONTROL_PANEL_ACTIONS
#define H_CONTROL_PANEL_ACTIONS

#include "controlPanelActionsBase.h"

#include "cmapInspectBase.h"
#include "comparatorBase.h"
#include "controlPanel.h"
#include "controllerBase.h"
#include "fontEngineBase.h"
#include "img.h"
#include "matchAssessment.h"
#include "settingsBase.h"
#include "transformBase.h"

#pragma warning(push, 0)

#include <functional>
#include <memory>

#include <gsl/gsl>

#pragma warning(pop)

namespace pic2sym {

/// Implementation for the actions triggered by the controls from Control Panel
class ControlPanelActions : public IControlPanelActions {
 public:
  static input::Img& getImg() noexcept;

  ui::IControlPanel& getControlPanel(const cfg::ISettingsRW& cfg_) noexcept;

  ControlPanelActions(IController& ctrler_,
                      cfg::ISettingsRW& cfg_,
                      syms::IFontEngine& fe_,
                      match::MatchAssessor& ma_,
                      transform::ITransformer& t_,
                      ui::IComparator& comp_,
                      std::function<ui::ICmapInspect*()>&& cmiFn_) noexcept;

  // Slicing prevention
  ControlPanelActions(const ControlPanelActions&) = delete;
  ControlPanelActions(ControlPanelActions&&) = delete;
  void operator=(const ControlPanelActions&) = delete;
  void operator=(ControlPanelActions&&) = delete;

  /**
  Overwriting IMatchSettings with the content of 'initMatchSettings.cfg'
  @throw domain_error for obsolete / invalid file from replaceByUserDefaults()

  Exception to be only reported, not handled
  */
  void restoreUserDefaultMatchSettings() noexcept(!UT) override;

  /// Saving current IMatchSettings to 'initMatchSettings.cfg'
  void setUserDefaultMatchSettings() const noexcept override;

  /// Updating the ISettingsRW object.
  /// @throw SymsChangeInterrupted when the user aborts the operation
  bool loadSettings(const std::string& from = "") override;

  /// Saving the ISettingsRW object
  void saveSettings() const noexcept override;

  /// Needed to restore encoding index
  unsigned getFontEncodingIdx() const noexcept override;

  /**
  Sets an image to be transformed.
  @param imgPath the image to be set
  @param silent when true, it doesn't show popup windows if the image is not
  valid

  @return false if the image cannot be set
  */
  bool newImage(const std::string& imgPath,
                bool silent = false) noexcept override;

#ifdef UNIT_TESTING  // Method available only in Unit Testing mode
  /// Provide directly a matrix instead of an image
  bool newImage(const cv::Mat& imgMat) noexcept override;
#endif  // UNIT_TESTING defined

  /// When unable to process a font type, invalidate it completely
  void invalidateFont() noexcept override;

  /// @throw SymsChangeInterrupted when the user aborts the operation
  void newFontFamily(const std::string& fontFile, unsigned fontSz) override;

  /// @throw SymsChangeInterrupted when the user aborts the operation
  void newFontEncoding(int encodingIdx) override;

#ifdef UNIT_TESTING  // Method available only in Unit Testing mode
  /// @throw SymsChangeInterrupted when the user aborts the operation
  bool newFontEncoding(const std::string& encName) override;
#endif  // UNIT_TESTING defined

  /// @throw SymsChangeInterrupted when the user aborts the operation
  void newFontSize(int fontSz) override;
  void newSymsBatchSize(int symsBatchSz) noexcept override;
  void newStructuralSimilarityFactor(double k) noexcept override;
  void newCorrelationFactor(double k) noexcept override;
  void newUnderGlyphCorrectnessFactor(double k) noexcept override;
  void newGlyphEdgeCorrectnessFactor(double k) noexcept override;
  void newAsideGlyphCorrectnessFactor(double k) noexcept override;
  void newContrastFactor(double k) noexcept override;
  void newGravitationalSmoothnessFactor(double k) noexcept override;
  void newDirectionalSmoothnessFactor(double k) noexcept override;
  void newGlyphWeightFactor(double k) noexcept override;
  void newThreshold4BlanksFactor(unsigned t) noexcept override;
  void newHmaxSyms(int maxSyms) noexcept override;
  void newVmaxSyms(int maxSyms) noexcept override;

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
  void setResultMode(bool hybrid) noexcept override;

  /**
  Approximates an image based on current settings

  @param durationS if not nullptr, it will return the duration of the
  transformation (when successful)

  @return false if the transformation cannot be started; true otherwise (even
  when the transformation is canceled and the result is just a draft)
  */
  bool performTransformation(double* durationS = nullptr) noexcept override;

  void showAboutDlg(const std::string& title,
                    const std::wstring& content) noexcept override;

  void showInstructionsDlg(const std::string& title,
                           const std::wstring& content) noexcept override;

 protected:
  /// Reports uncorrected settings when visualizing the cmap or while executing
  /// transform command. Cmap visualization can ignore image-related errors by
  /// setting 'imageRequired' to false.
  bool validState(bool imageRequired = true) const noexcept;

  // Next 3 protected methods do the ground work for their public correspondent
  // methods

  bool _newFontFamily(const std::string& fontFile,
                      bool forceUpdate = false) noexcept;
  bool _newFontEncoding(const std::string& encName,
                        bool forceUpdate = false) noexcept;
  bool _newFontSize(int fontSz, bool forceUpdate = false) noexcept;

 private:
  gsl::not_null<IController*> ctrler;
  gsl::not_null<cfg::ISettingsRW*> cfg;
  gsl::not_null<syms::IFontEngine*> fe;
  gsl::not_null<match::MatchAssessor*> ma;
  gsl::not_null<transform::ITransformer*> t;

  gsl::not_null<input::Img*> img;  ///< original image to process after resizing

  /// View for comparing original & result
  gsl::not_null<ui::IComparator*> comp;
  gsl::not_null<ui::IControlPanel*> cp;  ///< the configuration view

  /// viewer of the Cmap as lazy function
  std::function<ui::ICmapInspect*()> cmiFn;

  // Validation flags

  /// Is there an image to be transformed (not set yet, so false)
  bool imageOk{false};

  /// Is there a symbol set available (not set yet, so false)
  bool fontFamilyOk{false};
};

}  // namespace pic2sym

#endif  // H_CONTROL_PANEL_ACTIONS
