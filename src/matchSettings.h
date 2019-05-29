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

#ifndef H_MATCH_SETTINGS
#define H_MATCH_SETTINGS

#include "matchSettingsBase.h"
#include "misc.h"

#pragma warning(push, 0)

#ifndef UNIT_TESTING
#include <filesystem>
#endif  // UNIT_TESTING not defined

#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/serialization/split_member.hpp>
#include <boost/serialization/version.hpp>

#pragma warning(pop)

/// MatchSettings class controls the matching parameters for transforming one or
/// more images.
class MatchSettings : public IMatchSettings {
 public:
  /**
  Initializes the object.

  When unit testing, it leaves it empty.
  Otherwise it will load its fields from disk.
  */
  MatchSettings() noexcept(!UT);

  const bool& isHybridResult() const noexcept final { return hybridResultMode; }
  MatchSettings& setResultMode(bool hybridResultMode_) noexcept override;

  const double& get_kSsim() const noexcept final { return kSsim; }
  MatchSettings& set_kSsim(double kSsim_) noexcept override;

  const double& get_kCorrel() const noexcept final { return kCorrel; }
  MatchSettings& set_kCorrel(double kCorrel_) noexcept override;

  const double& get_kSdevFg() const noexcept final { return kSdevFg; }
  MatchSettings& set_kSdevFg(double kSdevFg_) noexcept override;

  const double& get_kSdevEdge() const noexcept final { return kSdevEdge; }
  MatchSettings& set_kSdevEdge(double kSdevEdge_) noexcept override;

  const double& get_kSdevBg() const noexcept final { return kSdevBg; }
  MatchSettings& set_kSdevBg(double kSdevBg_) noexcept override;

  const double& get_kContrast() const noexcept final { return kContrast; }
  MatchSettings& set_kContrast(double kContrast_) noexcept override;

  const double& get_kCosAngleMCs() const noexcept final { return kCosAngleMCs; }
  MatchSettings& set_kCosAngleMCs(double kCosAngleMCs_) noexcept override;

  const double& get_kMCsOffset() const noexcept final { return kMCsOffset; }
  MatchSettings& set_kMCsOffset(double kMCsOffset_) noexcept override;

  const double& get_kSymDensity() const noexcept final { return kSymDensity; }
  MatchSettings& set_kSymDensity(double kSymDensity_) noexcept override;

  unsigned getBlankThreshold() const noexcept final { return threshold4Blank; }
  MatchSettings& setBlankThreshold(unsigned threshold4Blank_) noexcept override;

#ifndef UNIT_TESTING
  /**
  Loads user defaults
  @throw domain_error for obsolete / invalid file

  Exception is handled, so no noexcept
  */
  void replaceByUserDefaults() override;

  /// Save these as user defaults
  void saveAsUserDefaults() const noexcept override;

#endif  // UNIT_TESTING not defined

  /// Provides a representation of these settings in a verbose manner or not
  const std::string toString(bool verbose) const noexcept override;

  /// @return a clone of current settings
  std::unique_ptr<IMatchSettings> clone() const noexcept override;

  /**
  The classes with MatchSettings might need to aggregate more information.
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
  // There are no concurrent I/O operations on MatchSettings

 private:
  friend class boost::serialization::access;

  /**
  Loading a MatchSettings object of a given version.
  It overwrites *this, reporting any changes

  @param ar the source of the object
  @param version what version is the loaded object

  @throw domain_error if loading from an archive with an unsupported version
  (more recent) or for an obsolete 'initMatchSettings.cfg'

  Exception handled, so no noexcept
  */
  template <class Archive>
  void load(Archive& ar, const unsigned version) {
#ifndef UNIT_TESTING
    if (version > VERSION)
      THROW_WITH_VAR_MSG("Cannot serialize future version (" +
                             std::to_string(version) +
                             ") of "
                             "MatchSettings class (now at version " +
                             std::to_string(VERSION) + ")!",
                         std::domain_error);

    if (version < VERSION) {
      /*
      MatchSettings is considered correctly initialized if its data is read from
      'res/defaultMatchSettings.txt'(most up-to-date file, which always exists)
      or 'initMatchSettings.cfg'(if it exists and is newer than
      'res/defaultMatchSettings.txt').

      Each launch of the application will either create / update
      'initMatchSettings.cfg' if this doesn't exist / is older than
      'res/defaultMatchSettings.txt'.

      Besides, anytime MatchSettings::VERSION is increased,
      'initMatchSettings.cfg' becomes obsolete, so it must be overwritten with
      the fresh data from 'res/defaultMatchSettings.txt'.

      Initialized is set to true at the end of the construction.
      */
      if (!initialized)
        // can happen only when loading an obsolete 'initMatchSettings.cfg'
        THROW_WITH_CONST_MSG("Obsolete version of 'initMatchSettings.cfg'!",
                             std::domain_error);

      // Point reachable while reading Settings with an older version of
      // MatchSettings field
    }
#endif  // UNIT_TESTING not defined

    // It is useful to see which settings changed when loading =>
    // Loading data in a temporary object and comparing with existing values.
    MatchSettings defSettings(*this);  // create as copy of previous values

    // read user default match settings
    if (version >= 2U) {  // versions >= 2 use hybridResultMode
      ar >> defSettings.hybridResultMode;
    } else {
      defSettings.hybridResultMode = false;
    }
    if (version >= 1U) {  // versions >= 1 use kSsim
      ar >> defSettings.kSsim;
    } else {
      defSettings.kSsim = 0.;
    }
    if (version >= 3U) {  // versions >= 3 use kCorrel
      ar >> defSettings.kCorrel;
    } else {
      defSettings.kCorrel = 0.;
    }
    ar >> defSettings.kSdevFg >> defSettings.kSdevEdge >> defSettings.kSdevBg >>
        defSettings.kContrast >> defSettings.kMCsOffset >>
        defSettings.kCosAngleMCs >> defSettings.kSymDensity >>
        defSettings.threshold4Blank;

    // these show message when there are changes
    setResultMode(defSettings.hybridResultMode);
    set_kSsim(defSettings.kSsim);
    set_kCorrel(defSettings.kCorrel);
    set_kSdevFg(defSettings.kSdevFg);
    set_kSdevEdge(defSettings.kSdevEdge);
    set_kSdevBg(defSettings.kSdevBg);
    set_kContrast(defSettings.kContrast);
    set_kMCsOffset(defSettings.kMCsOffset);
    set_kCosAngleMCs(defSettings.kCosAngleMCs);
    set_kSymDensity(defSettings.kSymDensity);
    setBlankThreshold(defSettings.threshold4Blank);

    if (version != VERSION_FROM_LAST_IO_OP)
      VERSION_FROM_LAST_IO_OP = version;
  }

  /// Saves *this to archive ar using current version of MatchSettings.
  template <class Archive>
  void save(Archive& ar, const unsigned version) const noexcept(!UT) {
    ar << hybridResultMode << kSsim << kCorrel << kSdevFg << kSdevEdge
       << kSdevBg << kContrast << kMCsOffset << kCosAngleMCs << kSymDensity
       << threshold4Blank;

    if (version != VERSION_FROM_LAST_IO_OP)
      VERSION_FROM_LAST_IO_OP = version;
  }

  BOOST_SERIALIZATION_SPLIT_MEMBER();

#ifndef UNIT_TESTING
  /**
  Initializes defCfgPath & cfgPath
  @throw runtim_error if couldn't find res/defaultMatchSettings.txt

  Exception to be only reported, not handled
  */
  static void configurePaths() noexcept(!UT);

  /**
  Loads the settings provided in cfgFile
  @throw info_parser_error only in UnitTesting if unable to find/parse the file

  The exception can be caught in UnitTesting and terminates the program
  otherwise
  */
  bool parseCfg() noexcept(!UT);

  /**
  Creates 'initMatchSettings.cfg' with data from 'res/defaultMatchSettings.txt'
  @throw runtime_error if unable to parse the configuration file

  Exception to be only reported, not handled
  */
  void createUserDefaults() noexcept(!UT);

  /// Path of the original configuration file
  static inline std::filesystem::path defCfgPath;

  /// Path of the user configuration file
  static inline std::filesystem::path cfgPath;

#endif  // UNIT_TESTING not defined

  double kSsim = 0.;    ///< power of factor controlling structural similarity
  double kCorrel = 0.;  ///< power of factor controlling correlation aspect

  /// power of factor for foreground glyph-patch correlation
  double kSdevFg = 0.;

  /// power of factor for contour glyph-patch correlation
  double kSdevEdge = 0.;

  /// power of factor for background glyph-patch correlation
  double kSdevBg = 0.;

  double kContrast = 0.;  ///< power of factor for the resulted glyph contrast

  /// power of factor targeting smoothness (mass-center offset)
  double kMCsOffset = 0.;

  /// power of factor targeting smoothness (mass-centers angle)
  double kCosAngleMCs = 0.;

  /// power of factor aiming fanciness, not correctness
  double kSymDensity = 0.;

  /// Using Blank character replacement under this threshold
  unsigned threshold4Blank = 0U;

  /// 'normal' means actual result; 'hybrid' cosmeticizes the result
  bool hybridResultMode = false;

#ifndef UNIT_TESTING
  bool initialized = false;  ///< true after FIRST completed initialization

#endif  // UNIT_TESTING not defined

  /// UINT_MAX or the class version of the last loaded/saved object
  static unsigned VERSION_FROM_LAST_IO_OP;
  // There are no concurrent I/O operations on MatchSettings

 public:
  // BUILD CLEAN WHEN THIS CHANGES!
  static const unsigned VERSION = 3U;  ///< version of MatchSettings class
};

BOOST_CLASS_VERSION(MatchSettings, MatchSettings::VERSION);

#endif  // H_MATCH_SETTINGS
