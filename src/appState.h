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

#ifdef UNIT_TESTING
#error Should not include this file when UNIT_TESTING is defined

#else  // UNIT_TESTING not defined

#ifndef H_APP_STATE
#define H_APP_STATE

namespace pic2sym::ui {

using AppStateType = size_t;

/**
Application distinct states

The actual state of the application might be a combination of them.
*/
enum struct AppState : AppStateType {
  Idle = 0ULL,       ///< Application is Idle
  UpdateImg = 1ULL,  ///< Updating the image to be transformed

  /// Updating settings related to the symbols to be used
  UpdateSymSettings = UpdateImg << 1,

  /// Updating settings related to the image to be transformed
  UpdateImgSettings = UpdateImg << 2,

  /// Updating settings related to the match aspects
  UpdateMatchSettings = UpdateImg << 3,

  /// Saving only the settings about match aspects
  SaveMatchSettings = UpdateImg << 4,

  SaveAllSettings = UpdateImg << 5,  ///< Saving all 3 categories of settings

  LoadAllSettings = UpdateImg << 6,  ///< Loading all 3 categories of settings

  /// Loading only the settings about match aspects
  LoadMatchSettings = UpdateImg << 7,

  ImgTransform = UpdateImg << 8  ///< Performing an approximation of an image
};

/// More readable states
#define ST(State) (AppStateType) AppState::State

/**
Every operation from the Control Panel must be authorized.
They must acquire such a permit when they start.
If they don't receive one, they must return.

In realization classes:
- the constructor will update application state to reflect the new [parallel]
action.
- the destructor should be used to revert the application state change
*/
class ActionPermit /*abstract*/ {
 public:
  virtual ~ActionPermit() noexcept = 0 {}
};

}  // namespace pic2sym::ui

#endif  // H_APP_STATE

#endif  // UNIT_TESTING
