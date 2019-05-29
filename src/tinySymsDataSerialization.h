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

#ifdef UNIT_TESTING
#include "../test/mockTinySymsDataSerialization.h"

#else  // UNIT_TESTING not defined

#ifndef H_TINY_SYMS_DATA_SERIALIZATION
#define H_TINY_SYMS_DATA_SERIALIZATION

#include "tinySym.h"

#pragma warning(push, 0)

#include <string>

#include <boost/serialization/vector.hpp>

#pragma warning(pop)

/// Clusters data that needs to be serialized
class VTinySymsIO {
 public:
  explicit VTinySymsIO(VTinySyms& tinySyms_) noexcept;
  virtual ~VTinySymsIO() noexcept = default;

  VTinySymsIO(const VTinySymsIO&) noexcept = default;
  VTinySymsIO(VTinySymsIO&&) noexcept = default;

  // 'tinySyms' is not supposed to change
  void operator=(const VTinySymsIO&) = delete;
  void operator=(VTinySymsIO&&) = delete;

  /**
  Overwrites current content with the items read from file located at path.
  Calls saveTo to overwrite the file if it contains older class version.

  @return false when loading fails.
  */
  bool loadFrom(const std::string& path) noexcept;

  /// Writes current content to file located at path. Returns false when saving
  /// fails.
  bool saveTo(const std::string& path) const noexcept;

  /**
  The classes with tiny symbols data might need to aggregate more information.
  Thus, these classes could have several versions while some of them have
  serialized instances.

  When loading such older classes, the extra information needs to be deduced.
  It makes sense to resave the file with the additional data to avoid
  recomputing it when reloading the same file.

  The method below helps checking if the loaded classes are the newest ones or
  not. Saved classes always use the newest class version.

  Before serializing the first object of this class, the method should return
  false.

  There are no concurrent I/O operations on VTinySymsIO
  */
  static bool olderVersionDuringLastIO() noexcept;

 private:
  friend class boost::serialization::access;

#pragma warning(disable : WARN_THROWS_ALTHOUGH_NOEXCEPT)
  /**
  Serializes this VTinySymsIO object to ar.
  @throw domain_error when loading VTinySymsIO in a format known only by newer
  versions of this class (in a newer version of Pic2Sym)

  The exception to be only reported, not handled.
  The dummy implementation from UnitTesting doesn't throw.
  */
  template <class Archive>
  void serialize(Archive& ar, const unsigned version) noexcept {
    if (version > VERSION)
      THROW_WITH_VAR_MSG("Cannot serialize(load) future version (" +
                             to_string(version) +
                             ") of "
                             "VTinySymsIO class (now at version " +
                             to_string(VERSION) + ")!",
                         std::domain_error);

    ar& tinySyms;

    if (version != VERSION_FROM_LAST_IO_OP)
      VERSION_FROM_LAST_IO_OP = version;
  }
#pragma warning(default : WARN_THROWS_ALTHOUGH_NOEXCEPT)

  /// reference to the tiny symbols to be serialized
  VTinySyms& tinySyms;

  /// UINT_MAX or the class version of the last loaded/saved object
  static unsigned VERSION_FROM_LAST_IO_OP;
  // There are no concurrent I/O operations on VTinySymsIO

 public:
  // BUILD CLEAN WHEN THIS CHANGES!
  static const unsigned VERSION = 0U;  ///< version of VTinySymsIO class
};

BOOST_CLASS_VERSION(VTinySymsIO, VTinySymsIO::VERSION);

#endif  // H_TINY_SYMS_DATA_SERIALIZATION

#endif  // UNIT_TESTING not defined
