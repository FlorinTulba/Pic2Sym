/******************************************************************************
 The application Pic2Sym approximates images by a
 grid of colored symbols with colored backgrounds.

 Copyrights from the libraries used by the program:
 - (c) 2003 Boost (www.boost.org)
     License: doc/licenses/Boost.lic
     http://www.boost.org/LICENSE_1_0.txt
 - (c) 2015-2016 OpenCV (www.opencv.org)
     License: doc/licenses/OpenCV.lic
     http://opencv.org/license/
 - (c) 1996-2002, 2006 The FreeType Project (www.freetype.org)
     License: doc/licenses/FTL.txt
     http://git.savannah.gnu.org/cgit/freetype/freetype2.git/plain/docs/FTL.TXT
 - (c) 1997-2002 OpenMP Architecture Review Board (www.openmp.org)
   (c) Microsoft Corporation (implementation for OpenMP C/C++ v2.0 March 2002)
     See: https://msdn.microsoft.com/en-us/library/8y6825x5.aspx
 - (c) 1995-2017 zlib software (Jean-loup Gailly and Mark Adler - www.zlib.net)
     License: doc/licenses/zlib.lic
     http://www.zlib.net/zlib_license.html


 (c) 2016-2019 Florin Tulba <florintulba@yahoo.com>

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

#ifndef H_SYM_SETTINGS
#define H_SYM_SETTINGS

#include "symSettingsBase.h"

#pragma warning(push, 0)

#include <compare>
#include <string>

#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/serialization/split_member.hpp>
#include <boost/serialization/version.hpp>

#pragma warning(pop)

/// Parameters concerning the symbols set used for approximating patches.
class SymSettings : public ISymSettings {
 public:
  /// Constructor takes an initial fontSz, just to present a valid slider value
  /// in Control Panel
  explicit SymSettings(unsigned fontSz_) noexcept : fontSz(fontSz_) {}

  /// Reset font settings apart from the font size
  /// which should remain on its value from the Control Panel
  void reset() noexcept override;

  /// Report if these settings are initialized or not
  bool initialized() const noexcept override;

  const std::string& getFontFile() const noexcept final { return fontFile; }
  void setFontFile(const std::string& fontFile_) noexcept override;

  const std::string& getEncoding() const noexcept final { return encoding; }
  void setEncoding(const std::string& encoding_) noexcept override;

  const unsigned& getFontSz() const noexcept final { return fontSz; }
  void setFontSz(unsigned fontSz_) noexcept override;

  /// @return a copy of these settings
  std::unique_ptr<ISymSettings> clone() const noexcept override;

#ifdef __cpp_lib_three_way_comparison
  std::strong_equality operator<=>(const SymSettings& other) const noexcept;

#else   // __cpp_lib_three_way_comparison not defined
  bool operator==(const SymSettings& other) const noexcept;
  bool operator!=(const SymSettings& other) const noexcept {
    return !(*this == other);
  }
#endif  // __cpp_lib_three_way_comparison

  /**
  The classes with SymSettings might need to aggregate more information.
  Thus, these classes could have several versions while some of them have
  serialized instances.

  When loading such older classes, the extra information needs to be deduced.
  It makes sense to resave the file with the additional data to avoid
  recomputing it when reloading the same file.

  The method below helps checking if the loaded classes are the newest ones or
  not. Saved classes always use the newest class version.

  Before serializing the first object of this class, the method should return
  false.
  */
  static bool olderVersionDuringLastIO() noexcept;
  // There are no concurrent I/O operations on SymSettings

 private:
  friend class boost::serialization::access;

  /**
  Loads a SymSettings object from ar overwriting *this and reporting the
  changes.

  @param ar source of the SymSettings to load
  @param version the version of the loaded object

  @throw domain_error if loading from an archive with an unsupported version
  (more recent)

  Exception to be only reported, not handled
  */
  template <class Archive>
  void load(Archive& ar, const unsigned version) noexcept(!UT) {
    if (version > VERSION)
      THROW_WITH_VAR_MSG("Cannot serialize(load) future version (" +
                             std::to_string(version) +
                             ") of "
                             "SymSettings class (now at version " +
                             std::to_string(VERSION) + ")!",
                         std::domain_error);

    // It is useful to see which settings changed when loading
    SymSettings defSettings(*this);  // create as copy of previous values

    // read user default match settings
    string newFontFile, newEncoding;
    ar >> newFontFile >> newEncoding >> defSettings.fontSz;

    defSettings.fontFile = newFontFile;
    defSettings.encoding = newEncoding;

    // these show message when there are changes
    setFontFile(defSettings.fontFile);
    setEncoding(defSettings.encoding);
    setFontSz(defSettings.fontSz);

    if (version != VERSION_FROM_LAST_IO_OP)
      VERSION_FROM_LAST_IO_OP = version;
  }

  /// Saves *this to ar
  template <class Archive>
  void save(Archive& ar, const unsigned version) const noexcept {
    ar << (string)fontFile << (string)encoding << fontSz;

    if (version != VERSION_FROM_LAST_IO_OP)
      VERSION_FROM_LAST_IO_OP = version;
  }

  BOOST_SERIALIZATION_SPLIT_MEMBER();

  /// UINT_MAX or the class version of the last loaded/saved object
  static unsigned VERSION_FROM_LAST_IO_OP;
  // There are no concurrent I/O operations on SymSettings

  /// The file containing the used font family with the desired style
  std::string fontFile;

  std::string encoding;  ///< the particular encoding of the used cmap
  unsigned fontSz;       ///< size of the symbols

 public:
  // BUILD CLEAN WHEN THIS CHANGES!
  static const unsigned VERSION = 0U;  ///< version of SymSettings class
};

BOOST_CLASS_VERSION(SymSettings, SymSettings::VERSION)

#endif  // H_SYM_SETTINGS
