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

#include "precompiled.h"

#include "bestMatchBase.h"
#include "clusterSupportWithPreselection.h"
#include "matchSupportWithPreselection.h"
#include "preselectManager.h"
#include "symbolsSupportWithPreselection.h"
#include "transformSupportWithPreselection.h"

using namespace std;
using namespace cv;

extern const bool PreselectionByTinySyms;

namespace {
/// PreselectionByTinySyms is true
class PreselectionOn : public IPreselManager {
 public:
  constexpr PreselectionOn() noexcept {}

  unique_ptr<IClustersSupport> createClusterSupport(ITinySymsProvider& tsp,
                                                    IClusterProcessing& ce,
                                                    VSymData& symsSet) const
      noexcept override {
    return make_unique<ClustersSupportWithPreselection>(
        tsp, ce, make_unique<SymsSupportWithPreselection>(), symsSet);
  }

  unique_ptr<IMatchSupport> createMatchSupport(
      CachedDataRW& cd,
      VSymData& symsSet,
      MatchAssessor& matchAssessor,
      const IMatchSettings& matchSettings) const noexcept override {
    return make_unique<MatchSupportWithPreselection>(cd, symsSet, matchAssessor,
                                                     matchSettings);
  }

  unique_ptr<ITransformSupport> createTransformSupport(
      IMatchEngine& me,
      const IMatchSettings& matchSettings,
      Mat& resized,
      Mat& resizedBlurred,
      vector<vector<unique_ptr<IBestMatch>>>& draftMatches,
      IMatchSupport& matchSupport) const noexcept override {
    return make_unique<TransformSupportWithPreselection>(
        me, matchSettings, resized, resizedBlurred, draftMatches, matchSupport);
  }
};

PreselectionOn preselectionOn;

/// PreselectionByTinySyms is false
class PreselectionOff : public IPreselManager {
 public:
  constexpr PreselectionOff() noexcept {}

  unique_ptr<IClustersSupport> createClusterSupport(ITinySymsProvider&,
                                                    IClusterProcessing& ce,
                                                    VSymData& symsSet) const
      noexcept override {
    return make_unique<ClustersSupport>(ce, make_unique<SymsSupport>(),
                                        symsSet);
  }

  unique_ptr<IMatchSupport> createMatchSupport(CachedDataRW& cd,
                                               VSymData&,
                                               MatchAssessor&,
                                               const IMatchSettings&) const
      noexcept override {
    return make_unique<MatchSupport>(cd);
  }

  unique_ptr<ITransformSupport> createTransformSupport(
      IMatchEngine& me,
      const IMatchSettings& matchSettings,
      Mat& resized,
      Mat& resizedBlurred,
      vector<vector<unique_ptr<IBestMatch>>>& draftMatches,
      IMatchSupport&) const noexcept override {
    return make_unique<TransformSupport>(me, matchSettings, resized,
                                         resizedBlurred, draftMatches);
  }
};

PreselectionOff preselectionOff;
}  // anonymous namespace

const IPreselManager& IPreselManager::concrete() noexcept {
  if (PreselectionByTinySyms)
    return preselectionOn;

  return preselectionOff;
}
