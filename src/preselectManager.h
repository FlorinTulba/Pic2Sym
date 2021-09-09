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

#ifndef H_PRESELECT_MANAGER
#define H_PRESELECT_MANAGER

#include "bestMatchBase.h"
#include "cachedData.h"
#include "clusterProcessingBase.h"
#include "clusterSupportBase.h"
#include "matchAssessment.h"
#include "matchEngineBase.h"
#include "matchSettingsBase.h"
#include "matchSupportBase.h"
#include "symDataBase.h"
#include "tinySymsProvider.h"
#include "transformSupportBase.h"

#pragma warning(push, 0)

#include <memory>
#include <vector>

#include <opencv2/core/core.hpp>

#pragma warning(pop)

namespace pic2sym::transform {

/// Abstract factory controlled by PreselectionByTinySyms
class IPreselManager /*abstract*/ {
 public:
  static const IPreselManager& concrete() noexcept;

  virtual std::unique_ptr<syms::cluster::IClustersSupport> createClusterSupport(
      syms::ITinySymsProvider& tsp,
      syms::cluster::IClusterProcessing& ce,
      syms::VSymData& symsSet) const noexcept = 0;

  virtual std::unique_ptr<match::IMatchSupport> createMatchSupport(
      CachedDataRW& cd,
      syms::VSymData& symsSet,
      match::MatchAssessor& matchAssessor,
      const cfg::IMatchSettings& matchSettings) const noexcept = 0;

  virtual std::unique_ptr<ITransformSupport> createTransformSupport(
      match::IMatchEngine& me,
      const cfg::IMatchSettings& matchSettings,
      cv::Mat& resized,
      cv::Mat& resizedBlurred,
      std::vector<std::vector<std::unique_ptr<match::IBestMatch>>>&
          draftMatches,
      match::IMatchSupport& matchSupport) const noexcept = 0;

  virtual ~IPreselManager() noexcept = 0 {}
};

}  // namespace pic2sym::transform

#endif  // H_PRESELECT_MANAGER
