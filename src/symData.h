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

#ifndef H_SYM_DATA
#define H_SYM_DATA

#include "symDataBase.h"

#include "matSerialization.h"
#include "misc.h"
#include "pixMapSymBase.h"
#include "warnings.h"

#pragma warning(push, 0)

#ifdef UNIT_TESTING
#include <unordered_map>
#endif  // UNIT_TESTING defined

#include <boost/serialization/array.hpp>
#include <boost/serialization/version.hpp>

#include <gsl/gsl>

#pragma warning(pop)

namespace pic2sym::syms {

/// Most symbol information
class SymData : public virtual ISymData {
 public:
  /// Fast constructor with the fields precomputed by computeFields - suitable
  /// for critical sections
  SymData(const cv::Mat& negSym_,
          const cv::Mat& symMiu0_,
          unsigned long code_,
          size_t symIdx_,
          double minVal_,
          double diffMinMax_,
          double avgPixVal_,
          double normSymMiu0_,
          const cv::Point2d& mc_,
          const MatArray& masks_,
          bool removable_ = false) noexcept;
  SymData(const IPixMapSym& pms, unsigned sz, bool forTinySym) noexcept;

  ~SymData() noexcept override = default;

  SymData(const SymData& other) noexcept;

  /// Moves the matrices from other (instead of just copying them)
  SymData(SymData&& other) noexcept;

  SymData& operator=(const SymData& other) noexcept;

  /// Moves the matrices from other (instead of just copying them)
  SymData& operator=(SymData&& other) noexcept;

  /// mass center of the symbol given original fg & bg (coordinates are within a
  /// unit-square: 0..1 x 0..1)
  const cv::Point2d& getMc() const noexcept final;

  /// negative of the symbol (0..255 byte for normal symbols; double for tiny)
  const cv::Mat& getNegSym() const noexcept final;

  /// The pixel values (double) are shifted so that the average pixel value
  /// (miu) is 0
  const cv::Mat& getSymMiu0() const noexcept final;

  /// norm L2 of (symbol - average pixel value)
  double getNormSymMiu0() const noexcept final;

  /// various masks
  const MatArray& getMasks() const noexcept final;

  /// symbol index within cmap
  size_t getSymIdx() const noexcept final;

#ifdef UNIT_TESTING
  /// the value of darkest pixel, range 0..1
  double getMinVal() const noexcept final;
#endif  // UNIT_TESTING defined

  /// difference between brightest and darkest pixels, each in 0..1
  double getDiffMinMax() const noexcept final;

  /// average pixel value, each pixel in 0..1
  double getAvgPixVal() const noexcept final;

  /// the code of the symbol
  unsigned long getCode() const noexcept final;

  /**
  Enabled symbol filters might mark this symbol as removable,
  but PreserveRemovableSymbolsForExamination from configuration might allow it
  to remain in the active symbol set used during image transformation.

  However, when removable == true && PreserveRemovableSymbolsForExamination,
  the symbol will appear as marked (inversed) in the cmap viewer

  This field doesn't need to be serialized, as filtering options might be
  different for distinct run sessions.
  */
  bool isRemovable() const noexcept final;

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

  There are no concurrent I/O operations on SymData
  */
  static bool olderVersionDuringLastIO() noexcept;

#ifdef UNIT_TESTING

  /// Used in the SymData constructor from below
  using IdxMatMap = std::unordered_map<int, const cv::Mat>;

  /// Constructor that allows filling only the relevant matrices from MatArray
  SymData(unsigned long code_,
          size_t symIdx_,
          double minVal_,
          double diffMinMax_,
          double avgPixVal_,
          double normSymMiu0_,
          const cv::Point2d& mc_,
          const IdxMatMap& relevantMats,
          const cv::Mat& negSym_ = cv::Mat(),
          const cv::Mat& symMiu0_ = cv::Mat()) noexcept;

  /// A clone with different symIdx
  std::unique_ptr<const SymData> clone(size_t symIdx_) const noexcept;

#endif  // UNIT_TESTING defined

  // BUILD CLEAN WHEN THIS CHANGES!
  /// Version of ISymData class
  static constexpr unsigned Version{1U};

  PROTECTED :

      /* Constructors callable from derived classes only */

      /// Used for creation of TinySym and ClusterData
      SymData(unsigned long code_ = ULONG_MAX,
              size_t symIdx_ = 0ULL,
              double avgPixVal_ = 0.,
              const cv::Point2d& mc_ = cv::Point2d(.5, .5)) noexcept;

  /// Used to create the TinySym centroid of a cluster
  SymData(const cv::Point2d& mc_, double avgPixVal_) noexcept;

  /**
  Updates negSym field
  @throw invalid_argument if negSym_ cannot be a valid value

  Exception to be only reported, not handled
  */
  void setNegSym(const cv::Mat& negSym_) noexcept;

  /**
  Updates mc field
  @throw invalid_argument if mc_ cannot be a valid value

  Exception to be only reported, not handled
  */
  void setMc(const cv::Point2d& mc_) noexcept;

  /**
  Updates double avgPixVal field
  @throw invalid_argument if avgPixVal_ cannot be a valid value

  Exception to be only reported, not handled
  */
  void setAvgPixVal(double avgPixVal_) noexcept;

  /// Computes symMiu0 and normSymMiu0 from sd based on glyph and its miu
  static void computeSymMiu0Related(const cv::Mat& glyph,
                                    double miu,
                                    SymData& sd) noexcept;

  /**
  Computes most information about a symbol based on glyph parameter.
  It's also used to spare the constructor of SymData from performing
  computeFields' job. That can be useful when multiple threads add SymData items
  to a vector within a critical section, so each new SymData's fields are
  prepared outside the critical section to minimize blocking of other threads.
  */
  static void computeFields(const cv::Mat& glyph,
                            SymData& sd,
                            bool forTinySym) noexcept;

  PRIVATE :

      friend class boost::serialization::access;

#pragma warning(disable : WARN_THROWS_ALTHOUGH_NOEXCEPT)
  /**
  Serializes this SymData object (apart from 'removable' field) to ar.

  'removable' field doesn't need to be serialized,
  as filtering options might be different for distinct run sessions.

  @throw domain_error when loading SymData in a format known only by newer
  versions of this class (in a newer version of Pic2Sym)

  Throwing such exceptions makes sense to be unit tested
  */
  template <class Archive>
  void serialize(Archive& ar, const unsigned version) noexcept(!UT) {
    EXPECTS_OR_REPORT_AND_THROW(version <= Version, std::domain_error,
                                "Cannot serialize(load) future version ("s +
                                    std::to_string(version) +
                                    ") of SymData class (now at version "s +
                                    std::to_string(Version) + ")!"s);

    ar& code& symIdx& minVal& diffMinMax& avgPixVal;
    ar& mc.x& mc.y;
    ar& negSym;
    ar& masks;

    if constexpr (Archive::is_loading::value) {
      if (version > 0U) {
        ar& symMiu0& normSymMiu0;
      } else {
        cv::Mat sym = negSym.clone();
        sym.convertTo(sym, CV_64FC1, Inv255);
        computeSymMiu0Related(sym, avgPixVal, *this);
      }

      if (removable)
        // Make sure a loaded symbol is initially not removable
        removable = false;

    } else {  // Saving
      ar& symMiu0& normSymMiu0;
    }

    if (version != VersionFromLast_IO_op)
      VersionFromLast_IO_op = version;
  }
#pragma warning(default : WARN_THROWS_ALTHOUGH_NOEXCEPT)

  /// mass center of the symbol given original fg & bg (coordinates are
  /// within a unit-square: 0..1 x 0..1)
  cv::Point2d mc{.5, .5};

  /// Negative of the symbol (0..255 byte for normal symbols; double for tiny)
  cv::Mat negSym;

  /// The pixel values (double) are shifted so that the average pixel value
  /// (miu) is 0
  cv::Mat symMiu0;

  MatArray masks;  ///< various masks

  size_t symIdx{};  ///< symbol index within cmap
  double minVal{};  ///< the value of darkest pixel, range 0..1

  /// Difference between brightest and darkest pixels, each in 0..1
  double diffMinMax{1.};

  double avgPixVal{};    ///< average pixel value, each pixel in 0..1
  double normSymMiu0{};  ///< norm L2 of (symbol - average pixel value)

  unsigned long code{ULONG_MAX};  ///< the code of the symbol

  /**
  Enabled symbol filters might mark this symbol as removable,
  but PreserveRemovableSymbolsForExamination from configuration might allow it
  to remain in the active symbol set used during image transformation.

  However, when removable == true && PreserveRemovableSymbolsForExamination,
  the symbol will appear as marked (inversed) in the cmap viewer

  This field doesn't need to be serialized, as filtering options might be
  different for distinct run sessions.
  */
  bool removable{false};

  /// UINT_MAX or the class version of the last loaded/saved object
  static inline unsigned VersionFromLast_IO_op{UINT_MAX};
  // There are no concurrent I/O operations on SymData
};

}  // namespace pic2sym::syms

BOOST_CLASS_VERSION(pic2sym::syms::SymData, pic2sym::syms::SymData::Version);

#endif  // H_SYM_DATA
