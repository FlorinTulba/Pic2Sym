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

#ifndef H_TRANSFORM_SUPPORT_WITH_PRESELECTION
#define H_TRANSFORM_SUPPORT_WITH_PRESELECTION

#include "transformSupport.h"

#include "matchSupportBase.h"
#include "matchSupportWithPreselection.h"

#pragma warning(push, 0)

#include <vector>

#pragma warning(pop)

namespace pic2sym::transform {

/**
Initializes and updates draft matches.
It perform those tasks also for tiny symbols.
*/
class TransformSupportWithPreselection : public TransformSupport {
 public:
  /// Requires an additional IMatchSupport parameter compared to the base
  /// constructor
  TransformSupportWithPreselection(
      match::IMatchEngine& me_,
      const cfg::IMatchSettings& matchSettings_,
      cv::Mat& resized_,
      cv::Mat& resizedBlurred_,
      std::vector<std::vector<std::unique_ptr<match::IBestMatch>>>&
          draftMatches_,
      match::IMatchSupport& matchSupport_) noexcept;

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

 private:
  /// Resized version of the original used by tiny symbols preselection
  cv::Mat resizedForTinySyms;

  /// Blurred version of the resized used by tiny symbols preselection
  cv::Mat resBlForTinySyms;

  /// temporary best matches used by tiny symbols preselection
  std::vector<std::vector<std::unique_ptr<match::IBestMatch>>>
      draftMatchesForTinySyms;

  /// Match support
  gsl::not_null<match::MatchSupportWithPreselection*> matchSupport;
};

}  // namespace pic2sym::transform

#endif  // H_TRANSFORM_SUPPORT_WITH_PRESELECTION
