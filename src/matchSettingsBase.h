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

#ifndef H_MATCH_SETTINGS_BASE
#define H_MATCH_SETTINGS_BASE

#pragma warning(push, 0)

#include <iostream>
#include <memory>
#include <string>

#pragma warning(pop)

namespace pic2sym::cfg {

/// IMatchSettings class controls the matching parameters for transforming one
/// or more images
class IMatchSettings /*abstract*/ {
 public:
  /// 'normal' means actual result; 'hybrid' cosmeticizes the result
  virtual const bool& isHybridResult() const noexcept = 0;

  /// Displays the update, if any
  virtual IMatchSettings& setResultMode(bool hybridResultMode_) noexcept = 0;

  /// Power of factor controlling structural similarity
  virtual const double& get_kSsim() const noexcept = 0;
  /// Displays the update, if any
  virtual IMatchSettings& set_kSsim(double kSsim_) noexcept = 0;

  /// Power of factor controlling correlation aspect
  virtual const double& get_kCorrel() const noexcept = 0;
  /// Displays the update, if any
  virtual IMatchSettings& set_kCorrel(double kCorrel_) noexcept = 0;

  /// Power of factor for foreground glyph-patch correlation
  virtual const double& get_kSdevFg() const noexcept = 0;
  /// Displays the update, if any
  virtual IMatchSettings& set_kSdevFg(double kSdevFg_) noexcept = 0;

  /// Power of factor for contour glyph-patch correlation
  virtual const double& get_kSdevEdge() const noexcept = 0;
  /// Displays the update, if any
  virtual IMatchSettings& set_kSdevEdge(double kSdevEdge_) noexcept = 0;

  /// Power of factor for background glyph-patch correlation
  virtual const double& get_kSdevBg() const noexcept = 0;
  /// Displays the update, if any
  virtual IMatchSettings& set_kSdevBg(double kSdevBg_) noexcept = 0;

  /// Power of factor for the resulted glyph contrast
  virtual const double& get_kContrast() const noexcept = 0;
  /// Displays the update, if any
  virtual IMatchSettings& set_kContrast(double kContrast_) noexcept = 0;

  /// Power of factor targeting smoothness (mass-centers angle)
  virtual const double& get_kCosAngleMCs() const noexcept = 0;
  /// Displays the update, if any
  virtual IMatchSettings& set_kCosAngleMCs(double kCosAngleMCs_) noexcept = 0;

  /// Power of factor targeting smoothness (mass-center offset)
  virtual const double& get_kMCsOffset() const noexcept = 0;
  /// Displays the update, if any
  virtual IMatchSettings& set_kMCsOffset(double kMCsOffset_) noexcept = 0;

  /// Power of factor aiming fanciness, not correctness
  virtual const double& get_kSymDensity() const noexcept = 0;
  /// Displays the update, if any
  virtual IMatchSettings& set_kSymDensity(double kSymDensity_) noexcept = 0;

  /// Using Blank character replacement under this threshold
  virtual unsigned getBlankThreshold() const noexcept = 0;
  /// Displays the update, if any
  virtual IMatchSettings& setBlankThreshold(
      unsigned threshold4Blank_) noexcept = 0;

#ifndef UNIT_TESTING
  /**
  Loads user defaults
  @throw domain_error for obsolete / invalid file

  Exception is handled, so no noexcept
  */
  virtual void replaceByUserDefaults() = 0;

  /// Save these as user defaults
  virtual void saveAsUserDefaults() const noexcept = 0;
#endif  // UNIT_TESTING not defined

  /// Provides a representation of the settings in a verbose manner or not
  virtual std::string toString(bool verbose) const noexcept = 0;

  /// @return a clone of current settings
  virtual std::unique_ptr<IMatchSettings> clone() const noexcept = 0;

  virtual ~IMatchSettings() noexcept = 0 {}
};

std::ostream& operator<<(std::ostream& os, const IMatchSettings& ms) noexcept;

}  // namespace pic2sym::cfg

#endif  // H_MATCH_SETTINGS_BASE
