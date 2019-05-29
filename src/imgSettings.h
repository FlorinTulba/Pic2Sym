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

#ifndef H_IMG_SETTINGS
#define H_IMG_SETTINGS

#include "imgSettingsBase.h"
#include "misc.h"

#pragma warning(push, 0)

#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/serialization/split_member.hpp>
#include <boost/serialization/version.hpp>

#pragma warning(pop)

/**
Contains max count of horizontal & vertical patches to process.

The image is resized appropriately before processing.
*/
class ImgSettings : public IfImgSettings {
 public:
  /// Constructor takes initial values just to present valid sliders positions
  /// in Control Panel
  ImgSettings(unsigned hMaxSyms_, unsigned vMaxSyms_) noexcept
      : hMaxSyms(hMaxSyms_), vMaxSyms(vMaxSyms_) {}

  unsigned getMaxHSyms() const noexcept final { return hMaxSyms; }
  void setMaxHSyms(unsigned syms) noexcept override;

  unsigned getMaxVSyms() const noexcept final { return vMaxSyms; }
  void setMaxVSyms(unsigned syms) noexcept override;

  std::unique_ptr<IfImgSettings> clone() const noexcept override;

  /**
  The classes with ImgSettings might need to aggregate more information.
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
  // There are no concurrent I/O operations on ImgSettings

 private:
  friend class boost::serialization::access;

  /**
  Overwrites *this with the ImgSettings object read from ar.

  @param ar source of the object to load
  @param version the version of the loaded ImgSettings

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
                             "ImgSettings class (now at version " +
                             std::to_string(VERSION) + ")!",
                         std::domain_error);

    // It is useful to see which settings changed when loading
    ImgSettings defSettings(*this);  // create as copy of previous values

    // read user default match settings
    ar >> defSettings.hMaxSyms >> defSettings.vMaxSyms;

    // these show message when there are changes
    setMaxHSyms(defSettings.hMaxSyms);
    setMaxVSyms(defSettings.vMaxSyms);

    if (version != VERSION_FROM_LAST_IO_OP)
      VERSION_FROM_LAST_IO_OP = version;
  }

  /// Saves *this to ar
  template <class Archive>
  void save(Archive& ar, const unsigned version) const noexcept {
    ar << hMaxSyms << vMaxSyms;

    if (version != VERSION_FROM_LAST_IO_OP)
      VERSION_FROM_LAST_IO_OP = version;
  }
  BOOST_SERIALIZATION_SPLIT_MEMBER();

  /// UINT_MAX or the class version of the last loaded/saved object
  static unsigned VERSION_FROM_LAST_IO_OP;
  // There are no concurrent I/O operations on ImgSettings

  unsigned hMaxSyms;  ///< Count of resulted horizontal symbols
  unsigned vMaxSyms;  ///< Count of resulted vertical symbols

 public:
  // BUILD CLEAN WHEN THIS CHANGES!
  static const unsigned VERSION = 0U;  ///< version of ImgSettings class
};

BOOST_CLASS_VERSION(ImgSettings, ImgSettings::VERSION)

#endif  // H_IMG_SETTINGS
