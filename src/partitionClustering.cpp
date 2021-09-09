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

#include "precompiled.h"
// This keeps precompiled.h first; Otherwise header sorting might move it

#include "partitionClustering.h"

#include "clusterEngine.h"
#include "clusterSerialization.h"
#include "jobMonitorBase.h"
#include "taskMonitor.h"
#include "warnings.h"

using namespace std;
using namespace cv;
using namespace gsl;

namespace pic2sym {

extern const bool FastDistSymToClusterComputation;
extern const double MaxAvgProjErrForPartitionClustering;
extern const double MaxRelMcOffsetForPartitionClustering;
extern const double MaxDiffAvgPixelValForPartitionClustering;

namespace syms::inline cluster {

#pragma warning(disable : WARN_THROWS_ALTHOUGH_NOEXCEPT)
unsigned PartitionClustering::formGroups(
    const VSymData& symsToGroup,
    vector<vector<unsigned>>& symsIndicesPerCluster,
    const string& fontType /* = ""*/) {
  assert(symsMonitor);  // Call only after useSymsMonitor()
  static p2s::ui::TaskMonitor partitionClustering{"partition clustering",
                                                  *symsMonitor};

  std::filesystem::path clusteredSetFile;
  ClusterIO rawClusters;
  if (!ClusterEngine::clusteredAlready(fontType, Name, clusteredSetFile) ||
      !rawClusters.loadFrom(clusteredSetFile.string())) {
    EXPECTS_OR_REPORT_AND_THROW_CONST_MSG(
        tsp, logic_error,
        HERE.function_name() +
            " should be called only after calling setTinySymsProvider()!"s);

    const VTinySyms& tinySyms = tsp->getTinySyms();
    static const double SqMaxRelMcOffsetForClustering{
        MaxRelMcOffsetForPartitionClustering *
        MaxRelMcOffsetForPartitionClustering};

#if defined _DEBUG && !defined UNIT_TESTING
    unsigned countAvgPixDiff{};
    unsigned countMcsOffset{};
    unsigned countDiff{};
    unsigned countHdiff{};
    unsigned countVdiff{};
    unsigned countBslashDiff{};
    unsigned countSlashDiff{};
#endif  // _DEBUG && !UNIT_TESTING

    const unsigned tinySymsCount{narrow_cast<unsigned>(size(tinySyms))};
    vector<int> clusterLabels(tinySymsCount, -1);

    const unsigned clustersCount{(unsigned)partition(
        tinySyms, clusterLabels,
        [&](const TinySym& a, const TinySym& b) noexcept {
#if !defined _DEBUG || defined UNIT_TESTING
          if (!FastDistSymToClusterComputation) {
            const double l1Dist{norm(a.getMat() - b.getMat(), NORM_L1)};
            return l1Dist <= MaxAvgProjErrForPartitionClustering;
          }

          if (abs(a.getAvgPixVal() - b.getAvgPixVal()) >
              MaxDiffAvgPixelValForPartitionClustering)
            return false;
          const Point2d mcDelta{a.getMc() - b.getMc()};
          const double mcDeltaY{abs(mcDelta.y)};
          if (mcDeltaY > MaxRelMcOffsetForPartitionClustering)
            return false;
          const double mcDeltaX{abs(mcDelta.x)};
          if (mcDeltaX > MaxRelMcOffsetForPartitionClustering)
            return false;
          if (mcDeltaX * mcDeltaX + mcDeltaY * mcDeltaY >
              SqMaxRelMcOffsetForClustering)
            return false;

          const auto checkProjections = [](const Mat& projA,
                                           const Mat& projB) noexcept {
            const double *pDataA, *pDataAEnd, *pDataB;
            pDataA = reinterpret_cast<const double*>(projA.datastart);
            pDataAEnd = reinterpret_cast<const double*>(projA.datalimit);
            pDataB = reinterpret_cast<const double*>(projB.datastart);
            for (double sumOfAbsDiffs{}; pDataA != pDataAEnd;) {
              sumOfAbsDiffs += abs(*pDataA++ - *pDataB++);
              if (sumOfAbsDiffs > MaxAvgProjErrForPartitionClustering)
                return false;
            }
            return true;
          };

#define ORIENTATION(Orientation) \
  a.get##Orientation##AvgProj(), b.get##Orientation##AvgProj()

          if (!checkProjections(ORIENTATION(V)))
            return false;
          if (!checkProjections(ORIENTATION(H)))
            return false;
          if (!checkProjections(ORIENTATION(BackslashDiag)))
            return false;
          if (!checkProjections(ORIENTATION(SlashDiag)))
            return false;

#undef ORIENTATION

          MatConstIterator_<double> itA{a.getMat().begin<double>()};
          MatConstIterator_<double> itAEnd{a.getMat().end<double>()};
          MatConstIterator_<double> itB{b.getMat().begin<double>()};
          for (double sumOfAbsDiffs{}; itA != itAEnd;) {
            sumOfAbsDiffs += abs(*itA++ - *itB++);
            if (sumOfAbsDiffs > MaxAvgProjErrForPartitionClustering)
              return false;
          }

          return true;

#else  // DEBUG mode and UNIT_TESTING not defined
          if (!FastDistSymToClusterComputation) {
            const double l1Dist{norm(a.getMat() - b.getMat(), NORM_L1)};
            return l1Dist <= MaxAvgProjErrForPartitionClustering;
          }

          const double avgPixDiff{abs(a.getAvgPixVal() - b.getAvgPixVal())};
          const bool bAvgPixDiff{avgPixDiff >
                                 MaxDiffAvgPixelValForPartitionClustering};
          if (bAvgPixDiff) {
            ++countAvgPixDiff;
            return false;
          }

          const double mcsOffset{norm(a.getMc() - b.getMc())};
          const bool bMcsOffset{mcsOffset >
                                MaxRelMcOffsetForPartitionClustering};
          if (bMcsOffset) {
            ++countMcsOffset;
            return false;
          }

          const auto checkDifferences = [](const Mat& matA, const Mat& matB,
                                           unsigned& counter) noexcept {
            const double l1Norm{norm(matA - matB, NORM_L1)};
            const bool contributesToWorkloadReduction{
                l1Norm > MaxAvgProjErrForPartitionClustering};
            if (contributesToWorkloadReduction) {
              ++counter;
              return false;
            }
            return true;
          };

#define FIELD(Field) a.get##Field(), b.get##Field()

          if (!checkDifferences(FIELD(VAvgProj), countVdiff))
            return false;
          if (!checkDifferences(FIELD(HAvgProj), countHdiff))
            return false;
          if (!checkDifferences(FIELD(BackslashDiagAvgProj), countBslashDiff))
            return false;
          if (!checkDifferences(FIELD(SlashDiagAvgProj), countSlashDiff))
            return false;
          if (!checkDifferences(FIELD(Mat), countDiff))
            return false;

#undef FIELD

          return true;

#endif  // _DEBUG, UNIT_TESTING
        })};

    rawClusters.reset(clustersCount, move(clusterLabels));
    cout << "\nAll the " << tinySymsCount
         << " symbols of the charmap were clustered in " << clustersCount
         << " groups" << endl;

#if defined _DEBUG && !defined UNIT_TESTING
    PRINTLN(countAvgPixDiff);
    PRINTLN(countMcsOffset);
    PRINTLN(countVdiff);
    PRINTLN(countHdiff);
    PRINTLN(countBslashDiff);
    PRINTLN(countSlashDiff);
    PRINTLN(countDiff);
#endif  // _DEBUG && !UNIT_TESTING

    rawClusters.saveTo(clusteredSetFile.string());
  }

  // Adapt clusters for filtered cmap
  symsIndicesPerCluster.assign(rawClusters.getClustersCount(),
                               vector<unsigned>());
  for (unsigned i{}, lim{narrow_cast<unsigned>(size(symsToGroup))}; i < lim;
       ++i)
    symsIndicesPerCluster[(size_t)rawClusters.getClusterLabels()
                              [symsToGroup[(size_t)i]->getSymIdx()]]
        .push_back(i);
  const auto& [newEndIt, _] = ranges::remove_if(
      symsIndicesPerCluster,
      [](const vector<unsigned>& elem) noexcept { return elem.empty(); });

  symsIndicesPerCluster.resize(
      (size_t)distance(begin(symsIndicesPerCluster), newEndIt));

  partitionClustering.taskDone();

  return narrow_cast<unsigned>(size(symsIndicesPerCluster));
}
#pragma warning(default : WARN_THROWS_ALTHOUGH_NOEXCEPT)

}  // namespace syms::inline cluster
}  // namespace pic2sym
