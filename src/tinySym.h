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

#ifndef H_TINY_SYM
#define H_TINY_SYM

#include "symData.h"
#include "tinySymBase.h"

#pragma warning(push, 0)

#include <boost/serialization/array.hpp>
#include <boost/serialization/base_object.hpp>
#include <boost/serialization/version.hpp>

#pragma warning(pop)

class IPixMapSym;  // Forward declaration

#pragma warning(disable : WARN_INHERITED_VIA_DOMINANCE)

/**
Data for tiny symbols.

ITinySym is virtual inherited to allow further diamond inheritance - see
ICentroid and Centroid.
*/
class TinySym : public SymData, public virtual ITinySym {
 public:
  /// Empty symbols with the code & index from cmap
  TinySym(unsigned long code_ = ULONG_MAX, size_t symIdx_ = 0ULL) noexcept;

  /// Creates tiny symbol based on a much larger reference symbol
  explicit TinySym(const IPixMapSym& refSym) noexcept;

#ifdef UNIT_TESTING
  /// Generate the tiny symbol based on a negative
  TinySym(const cv::Mat& negSym_,
          const cv::Point2d& mc_ = cv::Point2d(.5, .5),
          double avgPixVal_ = 0.) noexcept;
#endif  // UNIT_TESTING defined

  /// Grounded version of the small symbol divided by TinySymArea
  const cv::Mat& getMat() const noexcept final;

  /// Horizontal projection divided by TinySymSz
  const cv::Mat& getHAvgProj() const noexcept final;

  /// Vertical projection divided by TinySymSz
  const cv::Mat& getVAvgProj() const noexcept final;

  /// Normal diagonal projection divided by TinySymDiagsCount
  const cv::Mat& getBackslashDiagAvgProj() const noexcept final;

  /// Inverse diagonal projection divided by TinySymDiagsCount
  const cv::Mat& getSlashDiagAvgProj() const noexcept final;

  /**
  Grounded version of the small symbol divided by TinySymArea
  @throw invalid_argument if parameter is invalid

  Exception to be only reported, not handled
  */
  void setMat(const cv::Mat& mat_) noexcept;

  /**
  Horizontal projection divided by TinySymSz
  @throw invalid_argument if parameter is invalid

  Exception to be only reported, not handled
  */
  void setHAvgProj(const cv::Mat& hAvgProj_) noexcept;

  /**
  Vertical projection divided by TinySymSz
  @throw invalid_argument if parameter is invalid

  Exception to be only reported, not handled
  */
  void setVAvgProj(const cv::Mat& vAvgProj_) noexcept;

  /**
  Inverse diagonal projection divided by TinySymDiagsCount
  @throw invalid_argument if parameter is invalid

  Exception to be only reported, not handled
  */
  void setSlashDiagAvgProj(const cv::Mat& slashDiagAvgProj_) noexcept;

  /**
  Normal diagonal projection divided by TinySymDiagsCount
  @throw invalid_argument if parameter is invalid

  Exception to be only reported, not handled
  */
  void setBackslashDiagAvgProj(const cv::Mat& backslashDiagAvgProj_) noexcept;

  /**
  The classes with symbol data might need to aggregate more information.
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
  // There are no concurrent I/O operations on TinySym

  PRIVATE :

      friend class boost::serialization::access;

#pragma warning(disable : WARN_THROWS_ALTHOUGH_NOEXCEPT)
  /**
  Serializes this TinySym object to ar
  @throw domain_error when loading tiny symbols in a format known only by newer
  versions of this class (in a newer version of Pic2Sym)

  Throwing such exceptions makes sense to be unit tested
  */
  template <class Archive>
  void serialize(Archive& ar, const unsigned version) noexcept(!UT) {
    if (version > VERSION)
      THROW_WITH_VAR_MSG("Cannot serialize(load) future version (" +
                             to_string(version) +
                             ") of "
                             "TinySym class (now at version " +
                             to_string(VERSION) + ")!",
                         std::domain_error);

    ar& boost::serialization::base_object<SymData>(*this);

    ar& mat& hAvgProj& vAvgProj& backslashDiagAvgProj& slashDiagAvgProj;

    if (version != VERSION_FROM_LAST_IO_OP)
      VERSION_FROM_LAST_IO_OP = version;
  }
#pragma warning(default : WARN_THROWS_ALTHOUGH_NOEXCEPT)

  /// UINT_MAX or the class version of the last loaded/saved object
  static unsigned VERSION_FROM_LAST_IO_OP;
  // There are no concurrent I/O operations on TinySym

  /*
  Next 5 matrices from below are for the grounded version, not the original.
  Each would normally contain elements in range 0..1, but all of them were
  divided by the number of elements of the corresponding matrix.
  The reason behind this division is that the matrices are norm() compared
  and when the matrices are normalized (by the above mentioned division),
  the norm() result ranges for all of them in 0..1. So, despite they have,
  respectively: n^2, n, n, 2*n-1 and 2*n-1 elements, comparing 2 of the same
  kind with norm() produces values within a UNIQUE range 0..1. Knowing that,
  we can define a SINGLE threshold for all 5 matrices that establishes when
  2 matrices of the same kind are similar.

  The alternative was to define/derive a threshold for each individual
  category (n^2, n, 2*n-1), but this requires adapting these new thresholds
  to every n - configurable size of tiny symbols.

  So, the normalization allows setting a single threshold for comparing tiny
  symbols of any configured size:
  - MaxAvgProjErrForPartitionClustering for partition clustering
  - TTSAS_Threshold_Member for TTSAS clustering
  */

  cv::Mat mat;  ///< grounded version of the small symbol divided by TinySymArea
  cv::Mat hAvgProj;  ///< horizontal projection divided by TinySymSz
  cv::Mat vAvgProj;  ///< vertical projection divided by TinySymSz

  /// Normal diagonal projection divided by TinySymDiagsCount
  cv::Mat backslashDiagAvgProj;

  /// Inverse diagonal projection divided by TinySymDiagsCount
  cv::Mat slashDiagAvgProj;

 public:
  // BUILD CLEAN WHEN THIS CHANGES!
  static const unsigned VERSION = 0U;  ///< version of ITinySym class
};

#pragma warning(default : WARN_INHERITED_VIA_DOMINANCE)

BOOST_CLASS_VERSION(TinySym, TinySym::VERSION);

/// container with TinySym-s
typedef std::vector<TinySym> VTinySyms;

#endif  // H_TINY_SYM
