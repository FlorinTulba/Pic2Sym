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

#include "noClustering.h"

#include "taskMonitor.h"

using namespace std;
using namespace gsl;

namespace pic2sym {

extern const bool UsingOMP;

namespace syms::inline cluster {

unsigned NoClustering::formGroups(
    const VSymData& symsToGroup,
    vector<vector<unsigned>>& symsIndicesPerCluster,
    const string& /*fontType = ""*/) {
  assert(symsMonitor);  // Call only after useSymsMonitor()
  static p2s::ui::TaskMonitor trivialClustering{"trivial clustering",
                                                *symsMonitor};

  // One cluster per symbol - each symbol forms its own cluster
  const int clustersCount{narrow_cast<int>(size(symsToGroup))};

  symsIndicesPerCluster.assign((size_t)clustersCount, vector<unsigned>(1));

  // Accessing different vector elements => ok to parallelize
#pragma warning(disable : WARN_CODE_ANALYSIS_IGNORES_OPENMP)
#pragma omp parallel if (UsingOMP)
#pragma omp for schedule(static, 8) nowait
  for (int i{}; i < clustersCount; ++i)
    symsIndicesPerCluster[(size_t)i][0ULL] = (unsigned)i;
#pragma warning(default : WARN_CODE_ANALYSIS_IGNORES_OPENMP)

  trivialClustering.taskDone();

  return (unsigned)clustersCount;
}

}  // namespace syms::inline cluster
}  // namespace pic2sym
