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

#ifndef H_SETTINGS
#define H_SETTINGS

#include "settingsBase.h"

#include "imgSettings.h"
#include "matchSettings.h"
#include "symSettings.h"

#pragma warning(push, 0)

#include <memory>

#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/serialization/split_member.hpp>
#include <boost/serialization/version.hpp>

#include <gsl/gsl>

#pragma warning(pop)

namespace pic2sym::cfg {

/// Envelopes all parameters required for transforming images
class Settings : public ISettingsRW {
 public:
  // BUILD CLEAN WHEN THIS CHANGES!
  /// Version of Settings class
  static constexpr unsigned Version{};

  /**
  Creates a complete set of settings required during image transformations.

  @param ms_ incoming parameter copied to ms field.
  */
  explicit Settings(const IMatchSettings& ms_) noexcept;
  Settings() noexcept;  ///< Creates Settings with empty MatchSettings

  // Slicing prevention
  Settings(const Settings&) = delete;
  Settings(Settings&&) = delete;
  void operator=(const Settings&) = delete;
  void operator=(Settings&&) = delete;

  // Read-only accessors
  const ISymSettings& getSS() const noexcept final;
  const IfImgSettings& getIS() const noexcept final;
  const IMatchSettings& getMS() const noexcept final;

  // Accessors for changing the settings
  ISymSettings& refSS() noexcept final;
  IfImgSettings& refIS() noexcept final;
  IMatchSettings& refMS() noexcept final;

  /**
  The classes with Settings might need to aggregate more information.
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
  // There are no concurrent I/O operations on Settings

 private:
  friend class boost::serialization::access;

#pragma warning(disable : WARN_THROWS_ALTHOUGH_NOEXCEPT)
  /**
  Overwrites *this with the Settings object read from ar.

  @param ar source of the object to load
  @param version the version of the loaded Settings

  @throw domain_error if loading from an archive with an unsupported version
  (more recent)

  Exception to be only reported, not handled
  */
  template <class Archive>
  void load(Archive& ar, const unsigned version) noexcept(!UT) {
    EXPECTS_OR_REPORT_AND_THROW(version <= Version, std::domain_error,
                                "Cannot serialize(load) future version ("s +
                                    std::to_string(version) +
                                    ") of Settings class (now at version "s +
                                    std::to_string(Version) + ")!"s);

    // read user default match settings
    ar >> dynamic_cast<SymSettings&>(*ss) >> dynamic_cast<ImgSettings&>(*is) >>
        dynamic_cast<MatchSettings&>(*ms);

    if (version != VersionFromLast_IO_op)
      VersionFromLast_IO_op = version;
  }
#pragma warning(default : WARN_THROWS_ALTHOUGH_NOEXCEPT)

  /// Saves *this to ar
  template <class Archive>
  void save(Archive& ar, const unsigned version) const noexcept {
    ar << dynamic_cast<const SymSettings&>(*ss)
       << dynamic_cast<const ImgSettings&>(*is)
       << dynamic_cast<const MatchSettings&>(*ms);

    if (version != VersionFromLast_IO_op)
      VersionFromLast_IO_op = version;
  }

  BOOST_SERIALIZATION_SPLIT_MEMBER();

  /// Parameters concerning the symbols set used for approximating patches
  std::unique_ptr<ISymSettings> ss;

  /// Contains max count of horizontal & vertical patches to process
  std::unique_ptr<IfImgSettings> is;

  /// Settings used during approximation process
  std::unique_ptr<IMatchSettings> ms;

  /// UINT_MAX or the class version of the last loaded/saved object
  static inline unsigned VersionFromLast_IO_op{UINT_MAX};
  // There are no concurrent I/O operations on Settings
};

}  // namespace pic2sym::cfg

BOOST_CLASS_VERSION(pic2sym::cfg::Settings, pic2sym::cfg::Settings::Version)

#endif  // H_SETTINGS
