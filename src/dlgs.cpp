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

#include "precompiled.h"
// This keeps precompiled.h first; Otherwise header sorting might move it

#ifndef UNIT_TESTING

#include "dlgs.h"

#include "misc.h"
#include "warnings.h"

#pragma warning(push, 0)

#include <algorithm>
#include <cassert>
#include <filesystem>
#include <ranges>
#include <regex>
#include <sstream>
#include <unordered_map>
#include <vector>

#include <gsl/gsl>

#pragma warning(pop)

using namespace std;
using namespace filesystem;
using namespace gsl;

extern template class unordered_map<string, string>;

namespace pic2sym {

extern const unsigned Settings_MIN_FONT_SIZE;
extern const unsigned Settings_MAX_FONT_SIZE;

namespace ui {

/**
FontFinder encapsulates the logic to obtain the file path for a given
font name. It has a single public and static method: pathFor.
*/
class SelectFont::FontFinder {
 public:
  /**
  Fills accessibleFonts with all global and user fonts to be found
  within Windows registries.

  Possible exceptions invalid_argument, runtime_error and
  out_of_range. All these shouldn't happen and their handling is not
  desired.
  */
  FontFinder() noexcept {
    RegistryHelper rh;
    FontData fd;
    while (rh.extractNextFont(fd))
      accessibleFonts.push_back(fd);
  }

  /**
  pathFor static method finds the path for a provided fontName.
  Unfortunately, the provided fontName isn't decorated with Bold and/or
  Italic at all, so isBold and isItalic parameters were necessary, too.
  @throw FontLocationFailure for empty choices

  Exception handled, so no rapid termination via noexcept
  */
  string pathFor(const string& fontName, bool isBold, bool isItalic) {
    unordered_map<string, string> choices;
    wstring wFontName{CBOUNDS(fontName)};

    for (FontData& fd : accessibleFonts)
      if (relevantFontName(fd.name, wFontName, isBold, isItalic))
        choices[wstr2str(fd.name)] = refineFontFileName(fd.location);

    return extractResult(choices);
  }

 private:
  /// Details about a font - its name and the file location for its data
  struct FontData {
    wstring name;
    wstring location;
  };

  /**
  RegistryHelper isolates Registry API from the business logic within
  FontFinder. It provides an iterator-like method: extractNextFont.
  */
  class RegistryHelper final {
   public:
#pragma warning(disable : WARN_THROWS_ALTHOUGH_NOEXCEPT)
    /**
    Looks in the Windows registries for the installed & accessible
    fonts.
    @throw domain_error for wrong registry keys or any failed query

    Exception to be only reported, not handled
    */
    RegistryHelper() noexcept {
      // The mapping between the font name and the corresponding font
      // file can be found in the registry in:
      // HKEY_LOCAL_MACHINE>SOFTWARE>Microsoft>Windows
      // NT>CurrentVersion>Fonts
      // HKEY_CURRENT_USER>Software>Microsoft>Windows
      // NT>CurrentVersion>Fonts Last one might be missing if there are
      // no user fonts Registry key names are case insensitive, so
      // SOFTWARE == Software
      static constexpr LPCTSTR fontRegistryPath{
          _T("Software\\Microsoft\\Windows NT\\CurrentVersion\\Fonts")};

      // Throw only if the global font key wasn't found
      if (RegOpenKeyEx(HKEY_LOCAL_MACHINE,  // predefined key
                       fontRegistryPath,    // subkey
                       0U,                  // ulOptions - not an symbolic link
                       KEY_READ,            // rights to query, enumerate
                       &globalFontsKey      // returns the necessary key
                       ) != ERROR_SUCCESS)
        REPORT_AND_THROW_CONST_MSG(
            domain_error, "Couldn't find the Fonts mapping within Registry!");
      assert(globalFontsKey);

      const bool userFontsKeyExists{
          RegOpenKeyEx(HKEY_CURRENT_USER,  // predefined key
                       fontRegistryPath,   // subkey
                       0U,                 // ulOptions - not an symbolic link
                       KEY_READ,           // rights to query, enumerate
                       &userFontsKey       // returns the necessary key
                       ) == ERROR_SUCCESS};

      // Making sure userFontsKey is NULL if user fonts key wasn't found
      if (!userFontsKeyExists && userFontsKey)
        userFontsKey = nullptr;

      // Get the required buffer size for font names and the names of
      // the corresponding font files
      DWORD auxLongestNameLen{};
      DWORD auxLongestDataLen{};

      const bool globalFontsQueryOk{
          RegQueryInfoKey(
              globalFontsKey,
              nullptr,            // lpClass
              nullptr,            // lpcClass
              nullptr,            // lpReserved
              nullptr,            // lpcSubKeys (There are no subkeys)
              nullptr,            // lpcMaxSubKeyLen
              nullptr,            // lpcMaxClassLen
              &globalFontsCount,  // lpcValues (gloabl fonts count)
              &longestNameLen,    // returns required buffer size for font
                                  // names
              &longestDataLen,    // returns required buffer size for
                                  // corresponding names of the font files
              nullptr,            // lpcbSecurityDescriptor (Not necessary)
              nullptr             // lpftLastWriteTime (Not interested in this)
              ) == ERROR_SUCCESS};
      const bool userFontsQueryOk{
          !userFontsKeyExists ||
          RegQueryInfoKey(userFontsKey,
                          nullptr,  // lpClass
                          nullptr,  // lpcClass
                          nullptr,  // lpReserved
                          nullptr,  // lpcSubKeys (There are no subkeys)
                          nullptr,  // lpcMaxSubKeyLen
                          nullptr,  // lpcMaxClassLen
                          nullptr,  // lpcValues (user fonts count ignored)
                          &auxLongestNameLen,  // returns required buffer size
                                               // for font names
                          &auxLongestDataLen,  // returns required buffer size
                                               // for corresponding names of the
                                               // font files
                          nullptr,  // lpcbSecurityDescriptor (Not necessary)
                          nullptr  // lpftLastWriteTime (Not interested in this)
                          ) == ERROR_SUCCESS};

      // Throw if any query failed
      if (!globalFontsQueryOk || !userFontsQueryOk)
        REPORT_AND_THROW_CONST_MSG(domain_error,
                                   "Couldn't interrogate the Fonts keys!");

      longestNameLen = (std::max)(longestNameLen, auxLongestNameLen);
      longestDataLen = (std::max)(longestDataLen, auxLongestDataLen);

      fontNameBuf.resize((size_t)longestNameLen +
                         1ULL);  // reserve also for '\0'
      fontFileBuf.resize((size_t)longestDataLen +
                         2ULL);  // reserve also for '\0'(wchar_t as BYTE)
    }
#pragma warning(default : WARN_THROWS_ALTHOUGH_NOEXCEPT)

    ~RegistryHelper() noexcept {
      if (globalFontsKey)
        RegCloseKey(globalFontsKey);
      if (userFontsKey)
        RegCloseKey(userFontsKey);
    }

    RegistryHelper(const RegistryHelper&) = delete;
    RegistryHelper(RegistryHelper&&) = delete;
    void operator=(const RegistryHelper&) = delete;
    void operator=(RegistryHelper&&) = delete;

#pragma warning(disable : WARN_THROWS_ALTHOUGH_NOEXCEPT)
    /**
    extractNextFont returns true if there was another font to be
    handled. In that case, it returns the font name and font file name
    within the output parameter.

    @throw out_of_range for buffer overruns while reading fonts
    information
    @throw runtime_error if accessing the fonts data triggered some
    other error

    Such exceptions shouldn't happen, so their handling isn't desired.
    */
    bool extractNextFont(FontData& fontData) noexcept {
      DWORD lenFontName{longestNameLen + 1UL};
      DWORD lenFontFileName{longestDataLen + 2UL};
      LONG ret{RegEnumValue(
          globalFontsKey,
          idx++,               // which font index
          fontNameBuf.data(),  // storage for the font names
          &lenFontName,        // length of the returned font name
          nullptr,             // lpReserved
          nullptr,             // lpType (All are REG_SZ)
          fontFileBuf.data(),  // storage for the font file names
          &lenFontFileName)};  // length of the returned font file name

      // Start looking within the user fonts after checking all global
      // ones
      if (ERROR_NO_MORE_ITEMS == ret) {
        if (!userFontsKey)
          return false;  // no more fonts accessible

        ret = RegEnumValue(
            userFontsKey,
            idx - globalFontsCount - 1UL,  // which font index
            fontNameBuf.data(),            // storage for the font names
            &lenFontName,                  // length of the returned font name
            nullptr,                       // lpReserved
            nullptr,                       // lpType (All are REG_SZ)
            fontFileBuf.data(),            // storage for the font file names
            &lenFontFileName);  // length of the returned font file name

        if (ERROR_NO_MORE_ITEMS == ret)
          return false;  // no more global / user fonts accessible
      }

      if (ERROR_MORE_DATA == ret)
        REPORT_AND_THROW_CONST_MSG(out_of_range,
                                   HERE.function_name() +
                                       " : Allocated buffer isn't large enough "
                                       "to fit the font name"
                                       " or font file name!"s);

      if (ERROR_SUCCESS != ret)
        REPORT_AND_THROW_CONST_MSG(
            runtime_error,
            HERE.function_name() + " : Couldn't enumerate the Fonts!"s);

      fontData.name.assign(fontNameBuf.data());
      fontData.location.assign((TCHAR*)fontFileBuf.data());

      return true;
    }
#pragma warning(default : WARN_THROWS_ALTHOUGH_NOEXCEPT)

   private:
    HKEY globalFontsKey{nullptr};
    HKEY userFontsKey{nullptr};
    vector<TCHAR> fontNameBuf;
    vector<BYTE> fontFileBuf;
    DWORD globalFontsCount{};
    DWORD longestNameLen{};
    DWORD longestDataLen{};
    DWORD idx{};
  };

  /**
  The font read from registry needs to contain the required name and
  also match bold&italic style. Otherwise it will return false.
  */
  bool relevantFontName(const wstring& wCurFontName,
                        const wstring& wFontName,
                        bool isBold,
                        bool isItalic) noexcept {
    // fontName won't be necessarily a prefix of the font file name!!
    const auto at = wCurFontName.find(wFontName);
    if (at == wstring::npos)
      return false;  // current font doesn't contain the desired prefix

    // Bold and Italic fonts typically append such terms to their key
    // name.
    static const wregex rexBold{L"Bold|Heavy|Black", regex_constants::icase};
    static const wregex rexItalic{L"Italic|Oblique", regex_constants::icase};

    static match_results<wstring::const_iterator> match;

    // extract the suffix
    const wstring wSuffixCurFontName{
        (wstring)wCurFontName.substr(at + wFontName.length())};

    if (isBold != regex_search(wSuffixCurFontName, match, rexBold))
      return false;  // current font has different Bold status than
                     // expected

    if (isItalic != regex_search(wSuffixCurFontName, match, rexItalic))
      return false;  // current font has different Italic status than
                     // expected

    return true;
  }

  /**
  Ensures the obtained font file name represents a valid path
  @throw FontLocationFailure if unable to locate font file

  Exception handled, so no rapid termination via noexcept
  */
  string refineFontFileName(wstring& wCurFontFileName) {
#pragma warning(disable : WARN_DEPRECATED)
    // The fonts are typically installed within:
    // - %SystemRoot%\Fonts - global fonts
    // - %LOCALAPPDATA%\Microsoft\Windows\Fonts - user fonts
    static const path typicalGlobalFontsDir{
        path(string(getenv("SystemRoot"))).append("Fonts")};

    static const path typicalUserFontsDir{path(string(getenv("LOCALAPPDATA")))
                                              .append("Microsoft")
                                              .append("Windows")
                                              .append("Fonts")};
#pragma warning(default : WARN_DEPRECATED)

#pragma warning(disable : WARN_LOSSY_CONVERSION_CHANCE)
    path curFontFile{string{CBOUNDS(wCurFontFileName)}};
#pragma warning(default : WARN_LOSSY_CONVERSION_CHANCE)

    bool fullPathAlready{false};

    // The investigated paths for reaching the font file from the
    // parameter
    vector<string> attempts;

    if (curFontFile.has_parent_path()) {
      fullPathAlready = true;
      attempts.push_back(curFontFile.string());
    } else {
      // If the curFontFile isn't a path already, prefix it with a
      // typical fonts dir
      path temp{typicalGlobalFontsDir};
      temp /= curFontFile;
      if (!exists(temp)) {
        attempts.push_back(temp.string());
        temp = typicalUserFontsDir;
        temp /= curFontFile;
      }
      attempts.push_back(temp.string());
      curFontFile = move(temp);
    }

    if (!exists(curFontFile)) {
      ostringstream oss;
      ranges::copy(attempts, ostream_iterator<string>(oss, ", "));
      oss << "\b\b ";
      reportAndThrow<FontLocationFailure>(
          HERE.function_name() +
          " : Unable to find font file within the following location(s): "s +
          oss.str());
    }

    string result{curFontFile.string()};

    if (!fullPathAlready)  // update FontData.location to be full path
      wCurFontFileName.assign(CBOUNDS(result));

    return result;
  }

  /**
  When ambiguous results, lets the user select the correct one.
  @throw FontLocationFailure for empty choices

  Exception handled, so no rapid termination via noexcept
  */
  string extractResult(const unordered_map<string, string>& choices) {
    if (choices.empty())
      REPORT_AND_THROW_CONST_MSG(
          FontLocationFailure,
          HERE.function_name() +
              " : Couldn't find this font within registry!\n"
              "It might be there under a different name or "
              "it might appear only among the Windows Fonts as a "
              "shortcut "
              "to the actual file.\n"
              "The Font Dialog presents all corresponding Windows "
              "Fonts, "
              "unfortunately providing unreliable font name hints"s);

    const size_t choicesCount{std::size(choices)};
    if (1ULL == choicesCount)
      return cbegin(choices)->second;

    // More than 1 file suits the selected font and the user should
    // choose the appropriate one
    cout << "\nMore fonts within Windows Registry suit the selected "
            "Font type. "
            "Please select the appropriate one:\n";
    size_t idx{};
    for (const auto& [fontName, fontPath] : choices)
      cout << idx++ << " : " << fontName << " -> " << fontPath << '\n';

    // idx is here choicesCount
    while (idx >= choicesCount) {
      cout << "Enter correct index: ";
      cin >> idx;
    }

    return next(cbegin(choices), (ptrdiff_t)idx)->second;
  }

  vector<FontData> accessibleFonts;
};

OpenSave::OpenSave(not_null<basic_zstring<const TCHAR>> title,
                   basic_zstring<const TCHAR> filter,
                   basic_zstring<const TCHAR> defExtension /* = nullptr*/,
                   bool toOpen_ /* = true*/) noexcept
    : Dlg(), toOpen(toOpen_) {
  ZeroMemory(&ofn, sizeof(ofn));
  ofn.lStructSize = sizeof(ofn);
  ofn.hwndOwner = nullptr;  // no owner
  fNameBuf[0ULL] = '\0';
  ofn.lpstrFile = fNameBuf.data();
  ofn.nMaxFile = narrow_cast<DWORD>(size(fNameBuf));
  ofn.lpstrFilter = filter;
  ofn.nFilterIndex = 1;
  ofn.lpstrFileTitle = nullptr;
  ofn.nMaxFileTitle = 0;
  ofn.lpstrInitialDir = nullptr;
  ofn.lpstrTitle = title;
  if (defExtension)
    ofn.lpstrDefExt = defExtension;
  if (toOpen)
    ofn.Flags = OFN_PATHMUSTEXIST | OFN_FILEMUSTEXIST | OFN_NOCHANGEDIR;
  else
    ofn.Flags = OFN_OVERWRITEPROMPT | OFN_NOCHANGEDIR;
}

bool OpenSave::promptForUserChoice() noexcept {
  if (toOpen) {
    if (!GetOpenFileName(&ofn)) {
      reset();
      return false;
    }
  } else {
    if (!GetSaveFileName(&ofn)) {
      reset();
      return false;
    }
  }
  const wstring wResult{ofn.lpstrFile};
  result().assign(CBOUNDS(wResult));
  return true;
}

ImgSelector::ImgSelector() noexcept
    : OpenSave{_T("Please select an image to process"),
               _T("Allowed Image Files\0*.bmp;*.dib;*.png;*.tif;*.tiff;"
                  "*.jpg;*.jpe;*.jp2;*.jpeg;*.webp;*.pbm;*.pgm;*.ppm;*.sr;*."
                  "ras\0\0")} {}

SettingsSelector::SettingsSelector(bool toOpen_ /* = true*/) noexcept
    : OpenSave{toOpen_ ? _T("Please select a settings file to load")
                       : _T("Please specify where to save current settings"),
               _T("Allowed Settings Files\0*.p2s\0\0"), _T("p2s"), toOpen_} {}

SelectFont::SelectFont() noexcept
    : Dlg(), fontFinder(make_unique<FontFinder>()) {
  assert(fontFinder);

  ZeroMemory(&cf, sizeof(cf));
  cf.lStructSize = sizeof(cf);
  ZeroMemory(&lf, sizeof(lf));
  cf.lpLogFont = &lf;
  cf.Flags = CF_FORCEFONTEXIST | CF_NOVERTFONTS | CF_FIXEDPITCHONLY |
             CF_SCALABLEONLY | CF_NOSIMULATIONS | CF_NOSCRIPTSEL | CF_LIMITSIZE;
  cf.nSizeMin = (INT)Settings_MIN_FONT_SIZE;
  cf.nSizeMax = (INT)Settings_MAX_FONT_SIZE;
}

SelectFont::~SelectFont() noexcept {
  // fontFinder.release(); // implicit action
  // However, this empty d-tor is still required since FontFinder class
  // is not available to the header file, so the unique_ptr cannot use
  // delete there
}

bool SelectFont::promptForUserChoice() noexcept {
  if (!ChooseFont(&cf)) {
    reset();
    return false;
  }

  isBold = (cf.nFontType & 0x100) ||
           (lf.lfWeight > FW_MEDIUM);  // There are fonts with only a
                                       // Medium style (no Regular one)
  isItalic = (cf.nFontType & 0x200) || (lf.lfItalic != (BYTE)0);
  const wstring wResult{lf.lfFaceName};
  result().assign(CBOUNDS(wResult));

  cout << "Selected ";
  if (isBold)
    cout << "bold ";
  if (isItalic)
    cout << "italic ";
  cout << quoted(result(), '\'');

  try {
    result(fontFinder->pathFor(result(), isBold, isItalic));

    cout << " [" << result() << ']' << endl;
  } catch (const FontLocationFailure&) {
    reset();
  }

  return true;
}

}  // namespace ui
}  // namespace pic2sym

#endif  // UNIT_TESTING not defined
