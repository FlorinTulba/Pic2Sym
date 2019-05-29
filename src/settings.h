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

#ifndef H_SETTINGS
#define H_SETTINGS

#include "settingsBase.h"

#pragma warning(push, 0)

#include <memory>

#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/serialization/split_member.hpp>
#include <boost/serialization/version.hpp>

// Forward declarations
class SymSettings;
class ImgSettings;
class MatchSettings;

#pragma warning(pop)

// Forward declarations
class ISymSettings;
class IfImgSettings;
class IMatchSettings;

/// Envelopes all parameters required for transforming images
class Settings : public ISettingsRW {
 public:
  /**
  Creates a complete set of settings required during image transformations.

  @param ms_ incoming parameter copied to ms field.
  */
  explicit Settings(const IMatchSettings& ms_) noexcept;
  Settings() noexcept;  ///< Creates Settings with empty MatchSettings

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
    if (version > VERSION)
      THROW_WITH_VAR_MSG("Cannot serialize(load) future version (" +
                             std::to_string(version) +
                             ") of "
                             "Settings class (now at version " +
                             std::to_string(VERSION) + ")!",
                         std::domain_error);

    // read user default match settings
    ar >> dynamic_cast<SymSettings&>(*ss) >> dynamic_cast<ImgSettings&>(*is) >>
        dynamic_cast<MatchSettings&>(*ms);

    if (version != VERSION_FROM_LAST_IO_OP)
      VERSION_FROM_LAST_IO_OP = version;
  }

  /// Saves *this to ar
  template <class Archive>
  void save(Archive& ar, const unsigned version) const noexcept {
    ar << dynamic_cast<const SymSettings&>(*ss)
       << dynamic_cast<const ImgSettings&>(*is)
       << dynamic_cast<const MatchSettings&>(*ms);

    if (version != VERSION_FROM_LAST_IO_OP)
      VERSION_FROM_LAST_IO_OP = version;
  }

  BOOST_SERIALIZATION_SPLIT_MEMBER();

  /// Parameters concerning the symbols set used for approximating patches
  const std::unique_ptr<ISymSettings> ss;

  /// Contains max count of horizontal & vertical patches to process
  const std::unique_ptr<IfImgSettings> is;

  /// Settings used during approximation process
  const std::unique_ptr<IMatchSettings> ms;

  /// UINT_MAX or the class version of the last loaded/saved object
  static unsigned VERSION_FROM_LAST_IO_OP;
  // There are no concurrent I/O operations on Settings

 public:
  // BUILD CLEAN WHEN THIS CHANGES!
  static const unsigned VERSION = 0U;  ///< version of Settings class
};

BOOST_CLASS_VERSION(Settings, Settings::VERSION)

#endif  // H_SETTINGS
