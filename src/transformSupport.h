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

#ifndef H_TRANSFORM_SUPPORT
#define H_TRANSFORM_SUPPORT

#include "transformSupportBase.h"

#include "bestMatchBase.h"
#include "matchEngineBase.h"
#include "matchSettingsBase.h"

#pragma warning(push, 0)

#include <memory>
#include <vector>

#include <gsl/gsl>

#pragma warning(pop)

namespace pic2sym::transform {

/**
Initializes and updates draft matches.
When PreselectionByTinySyms == true, it perform those tasks also for tiny
symbols.
*/
class TransformSupport : public ITransformSupport {
 public:
  /// Base constructor
  TransformSupport(match::IMatchEngine& me_,
                   const cfg::IMatchSettings& matchSettings_,
                   cv::Mat& resized_,
                   cv::Mat& resizedBlurred_,
                   std::vector<std::vector<std::unique_ptr<match::IBestMatch>>>&
                       draftMatches_) noexcept;

  // Slicing prevention
  TransformSupport(const TransformSupport&) = delete;
  TransformSupport(TransformSupport&&) = delete;
  void operator=(const TransformSupport&) = delete;
  void operator=(TransformSupport&&) = delete;

  /// Initializes the drafts when a new image needs to be approximated
  void initDrafts(bool isColor,
                  unsigned patchSz,
                  unsigned patchesPerCol,
                  unsigned patchesPerRow) noexcept override;

  /// Resets the drafts when current image needs to be approximated in a
  /// different context
  void resetDrafts(unsigned patchesPerCol) noexcept override;

  /**
  Approximates row r of patches of size patchSz from an image with given width.
  It checks only the symbols with indices in range [fromSymIdx, upperSymIdx).
  */
  void approxRow(int r,
                 int width,
                 unsigned patchSz,
                 unsigned fromSymIdx,
                 unsigned upperSymIdx,
                 cv::Mat& result) noexcept override;

 protected:
  /// Initializes a row of a draft when a new image needs to be approximated
  static void initDraftRow(
      std::vector<std::vector<std::unique_ptr<match::IBestMatch>>>& draft,
      int r,
      unsigned patchesPerRow,
      const cv::Mat& res,
      const cv::Mat& resBlurred,
      int patchSz,
      bool isColor) noexcept;

  /// Resets a row of a draft when current image needs to be approximated in a
  /// different context
  static void resetDraftRow(
      std::vector<std::vector<std::unique_ptr<match::IBestMatch>>>& draft,
      int r) noexcept;

  /// Update the visualized draft
  static void patchImproved(cv::Mat& result,
                            unsigned sz,
                            const match::IBestMatch& draftMatch,
                            const cv::Range& rowRange,
                            int startCol) noexcept;

  /// Update PatchApprox for uniform Patch only during the compare with 1st sym
  /// (from 1st batch)
  static void manageUnifPatch(const cfg::IMatchSettings& ms,
                              cv::Mat& result,
                              unsigned sz,
                              match::IBestMatch& draftMatch,
                              const cv::Range& rowRange,
                              int startCol) noexcept;

  /// Determines if a given patch is worth approximating (Uniform patches don't
  /// make sense approximating)
  static bool checkUnifPatch(const match::IBestMatch& draftMatch) noexcept;

  gsl::not_null<match::IMatchEngine*> me;                   ///< match engine
  gsl::not_null<const cfg::IMatchSettings*> matchSettings;  ///< match settings
  gsl::not_null<cv::Mat*> resized;  ///< resized version of the original
  gsl::not_null<cv::Mat*>
      resizedBlurred;  ///< blurred version of the resized original

  /// temporary best matches
  gsl::not_null<std::vector<std::vector<std::unique_ptr<match::IBestMatch>>>*>
      draftMatches;
};

}  // namespace pic2sym::transform

#endif  // H_TRANSFORM_SUPPORT
