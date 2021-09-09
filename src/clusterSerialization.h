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
#include "../test/mockClusterSerialization.h"

#else  // UNIT_TESTING not defined

#ifndef H_CLUSTER_SERIALIZATION
#define H_CLUSTER_SERIALIZATION

#include "misc.h"
#include "warnings.h"

#pragma warning(push, 0)

#include <vector>

#include <gsl/gsl>

#include <boost/serialization/vector.hpp>
#include <boost/serialization/version.hpp>

#pragma warning(pop)

extern template class std::vector<int>;

namespace pic2sym::syms::inline cluster {

/// Clusters data that needs to be serialized
class ClusterIO {
 public:
  /**
  Overwrites current content with the items read from file located at path.
  @return false when loading fails, even for internal exceptions
  */
  bool loadFrom(const std::string& path) noexcept;

  /**
  Writes current content to file located at path.
  @return false when saving fails, even for internal exceptions
  */
  bool saveTo(const std::string& path) const noexcept;

  void reset(unsigned clustersCount_,
             std::vector<int>&& clusterLabels_) noexcept;

  const std::vector<int>& getClusterLabels() const noexcept {
    return clusterLabels;
  }

  unsigned getClustersCount() const noexcept { return clustersCount; }

  /**
  The classes with cluster data might need to aggregate more information.
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
  // There are no concurrent I/O operations on ClusterIO

  // BUILD CLEAN WHEN THIS CHANGES!
  /// Version of ClusterIO class
  static constexpr unsigned Version{};

 private:
  friend class boost::serialization::access;

#pragma warning(disable : WARN_THROWS_ALTHOUGH_NOEXCEPT)
  /**
  Serializes this ClusterIO object to ar
  @throw domain_error when loading ClusterIO in a format known only by newer
  versions of this class (in a newer version of Pic2Sym)
  Throwing such exceptions makes sense to be unit tested
  */
  template <class Archive>
  void serialize(Archive& ar, const unsigned version) noexcept(!UT) {
    EXPECTS_OR_REPORT_AND_THROW(version <= Version, std::domain_error,
                                "Cannot serialize(load) future version ("s +
                                    std::to_string(version) +
                                    ") of ClusterIO class (now at version "s +
                                    std::to_string(Version) + ")!"s);

    ar& clustersCount;
    ar& clusterLabels;

    if (version != VersionFromLast_IO_op)
      VersionFromLast_IO_op = version;
  }
#pragma warning(default : WARN_THROWS_ALTHOUGH_NOEXCEPT)

  /// UINT_MAX or the class version of the last loaded/saved object
  static inline unsigned VersionFromLast_IO_op{UINT_MAX};
  // There are no concurrent I/O operations on ClusterIO

  /// assigned cluster for each symbol when sorted as within the cmap (by
  /// symIdx)
  std::vector<int> clusterLabels;

  unsigned clustersCount{};  ///< total number of clusters
};

}  // namespace pic2sym::syms::inline cluster

BOOST_CLASS_VERSION(pic2sym::syms::cluster::ClusterIO,
                    pic2sym::syms::cluster::ClusterIO::Version);

#endif  // H_CLUSTER_SERIALIZATION

#endif  // UNIT_TESTING not defined
