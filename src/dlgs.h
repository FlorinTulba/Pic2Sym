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
#include "../test/mockDlgs.h"

#else  // UNIT_TESTING not defined

#ifndef H_DLGS
#define H_DLGS

#pragma warning(push, 0)

#define NOMINMAX
#include <Windows.h>
#include <tchar.h>

#include <array>
#include <memory>
#include <stdexcept>
#include <string>

#include <gsl/gsl>

#pragma warning(pop)

namespace pic2sym {

extern const unsigned Settings_DEF_FONT_SIZE;

namespace ui {

/// Distinct exception class for easier catching and handling font location
/// failures
class FontLocationFailure : public std::runtime_error {
 public:
  using runtime_error::runtime_error;
};

/// Dlg is the base class for the standard Windows dialogs from below
class Dlg /*abstract*/ {
 public:
  virtual ~Dlg() noexcept = 0 {}

  /**
  Displays the dialog and stores the selection or returns false when canceled.

  Getting user's option should be noexcept. Refinig the selection, however,
  might trigger an exception. If there's an exception while computing the
  selection, the method won't throw, but instead it will return true and the
  selection will be empty.
  */
  virtual bool promptForUserChoice() noexcept = 0;

  const std::string& selection() const noexcept { return _result; }

  virtual void reset() noexcept { _result.clear(); }

 protected:
  /// The result to be returned - const version
  const std::string& result() const noexcept { return _result; }

  /// The result to be returned - reference version
  std::string& result() noexcept { return _result; }

  /// The result to be returned - setter version
  void result(const std::string& result) noexcept { _result = result; }

 private:
  std::string _result;  ///< the result to be returned
};

/// OpenSave class controls a FileOpenDialog / FileSaveDialog.
class OpenSave /*abstract*/ : public Dlg {
 public:
  /**
  Displays the dialog and stores the selection or returns false when canceled.

  Getting user's option should be noexcept. Refinig the selection, however,
  might trigger an exception. If there's an exception while computing the
  selection, the method won't throw, but instead it will return true and the
  selection will be empty.
  */
  bool promptForUserChoice() noexcept override;

 protected:
  /// Prepares the dialog
  OpenSave(gsl::not_null<gsl::basic_zstring<const TCHAR> >
               title,  ///< displayed title of the dialog
           gsl::basic_zstring<const TCHAR> filter,  ///< expected extensions
           gsl::basic_zstring<const TCHAR> defExtension =
               nullptr,         ///< default extension
           bool toOpen_ = true  ///< open or save dialog
           ) noexcept;

  // Slicing prevention
  OpenSave(const OpenSave&) = delete;
  OpenSave(OpenSave&&) = delete;
  void operator=(const OpenSave&) = delete;
  void operator=(OpenSave&&) = delete;

 private:
  OPENFILENAME ofn;  ///< structure used by the FileOpenDialog
  std::array<TCHAR, 1024ULL> fNameBuf;  ///< buffer for the selected image file

  /// Most derived classes want Open File Dialog (not Save)
  bool toOpen{true};
};

/// Selecting an image to transform
class ImgSelector : public OpenSave {
 public:
  ImgSelector() noexcept;
};

/// Selecting a settings file to load / be saved
class SettingsSelector : public OpenSave {
 public:
  explicit SettingsSelector(bool toOpen_ = true) noexcept;
};

/// SelectFont class controls a ChooseFont Dialog.
class SelectFont : public Dlg {
 public:
  SelectFont() noexcept;  ///< Prepares the dialog

  // Explicit definition for releasing unique_ptr of undefined type: FontFinder
  ~SelectFont() noexcept override;

  // Slicing prevention
  SelectFont(const SelectFont&) = delete;
  SelectFont(SelectFont&&) = delete;
  void operator=(const SelectFont&) = delete;
  void operator=(SelectFont&&) = delete;

  /**
  Displays the dialog and stores the selection or returns false when canceled.

  Getting user's option should be noexcept. Refinig the selection, however,
  might trigger an exception. If there's an exception while computing the
  selection, the method won't throw, but instead it will return true and the
  selection will be empty.
  */
  bool promptForUserChoice() noexcept override;

  void reset() noexcept override {
    Dlg::reset();
    cf.iPointSize = (INT)Settings_DEF_FONT_SIZE;
    isBold = isItalic = false;
  }

  bool bold() const noexcept { return isBold; }
  bool italic() const noexcept { return isItalic; }
  unsigned size() const noexcept { return unsigned(cf.iPointSize / 10); }

 private:
  class FontFinder;
  std::unique_ptr<FontFinder> fontFinder;

  CHOOSEFONT cf;  ///< structure used by the ChooseFont Dialog
  LOGFONT lf;     ///< structure filled with Font information
  bool isBold{false};
  bool isItalic{false};
};

}  // namespace ui
}  // namespace pic2sym

#endif  // H_DLGS

#endif  // UNIT_TESTING not defined
