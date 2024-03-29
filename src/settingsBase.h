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

#ifndef H_SETTINGS_BASE
#define H_SETTINGS_BASE

#include "imgSettingsBase.h"
#include "matchSettingsBase.h"
#include "symSettingsBase.h"

#pragma warning(push, 0)

#include <iostream>

#pragma warning(pop)

namespace pic2sym::cfg {

/**
Interface providing read-only access to all parameters required for transforming
images. Allows serialization and feeding such objects to output streams.
*/
class ISettings /*abstract*/ {
 public:
  // Static validator methods
  static bool isBlanksThresholdOk(
      unsigned t,
      const std::string& reportedItem = "") noexcept;
  static bool isHmaxSymsOk(unsigned syms) noexcept;
  static bool isVmaxSymsOk(unsigned syms) noexcept;
  static bool isFontSizeOk(unsigned fs) noexcept;

  /// @return existing symbols settings
  virtual const ISymSettings& getSS() const noexcept = 0;

  /// @return existing image settings
  virtual const IfImgSettings& getIS() const noexcept = 0;

  /// @return existing match settings
  virtual const IMatchSettings& getMS() const noexcept = 0;

  virtual ~ISettings() noexcept = 0 {}
};

/// The ISettings interface plus accessors for settings modification
class ISettingsRW /*abstract*/ : public ISettings {
 public:
  /// Allows current symbols settings to be changed
  virtual ISymSettings& refSS() noexcept = 0;

  /// Allows current image settings to be changed
  virtual IfImgSettings& refIS() noexcept = 0;

  /// Allows current match settings to be changed
  virtual IMatchSettings& refMS() noexcept = 0;
};

std::ostream& operator<<(std::ostream& os, const ISettings& s) noexcept;

}  // namespace pic2sym::cfg

#endif  // H_SETTINGS_BASE
