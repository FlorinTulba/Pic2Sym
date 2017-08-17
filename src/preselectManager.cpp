/************************************************************************************************
 The application Pic2Sym approximates images by a
 grid of colored symbols with colored backgrounds.

 Copyrights from the libraries used by the program:
 - (c) 2016 Boost (www.boost.org)
		License: <http://www.boost.org/LICENSE_1_0.txt>
			or doc/licenses/Boost.lic
 - (c) 2015 OpenCV (www.opencv.org)
		License: <http://opencv.org/license.html>
            or doc/licenses/OpenCV.lic
 - (c) 2015 The FreeType Project (www.freetype.org)
		License: <http://git.savannah.gnu.org/cgit/freetype/freetype2.git/plain/docs/FTL.TXT>
	        or doc/licenses/FTL.txt
 - (c) 1997-2002 OpenMP Architecture Review Board (www.openmp.org)
   (c) Microsoft Corporation (Visual C++ implementation for OpenMP C/C++ Version 2.0 March 2002)
		See: <https://msdn.microsoft.com/en-us/library/8y6825x5(v=vs.140).aspx>
 - (c) 1995-2013 zlib software (Jean-loup Gailly and Mark Adler - see: www.zlib.net)
		License: <http://www.zlib.net/zlib_license.html>
            or doc/licenses/zlib.lic
 
 (c) 2016, 2017 Florin Tulba <florintulba@yahoo.com>

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
 If not, see <http://www.gnu.org/licenses/agpl-3.0.txt>.
 ***********************************************************************************************/

#include "preselectManager.h"
#include "symbolsSupportWithPreselection.h"
#include "clusterSupportWithPreselection.h"
#include "matchSupportWithPreselection.h"
#include "transformSupportWithPreselection.h"
#include "bestMatchBase.h"

using namespace std;
using namespace cv;

extern const bool PreselectionByTinySyms;

namespace {
	PreselectionOn preselectionOn;
	PreselectionOff preselectionOff;
}

const IPreselManager& IPreselManager::concrete() {
	if(PreselectionByTinySyms)
		return preselectionOn;

	return preselectionOff;
}

unique_ptr<ClustersSupport> PreselectionOn::createClusterSupport(ITinySymsProvider &tsp,
																 ClusterEngine &ce,
																 VSymData &symsSet) const {
	return make_unique<ClustersSupportWithPreselection>(tsp, ce,
														make_unique<SymsSupportWithPreselection>(),
														symsSet);
}

unique_ptr<MatchSupport> PreselectionOn::createMatchSupport(CachedData &cd,
															VSymData &symsSet,
															MatchAssessor &matchAssessor,
															const IMatchSettings &matchSettings) const {
	return make_unique<MatchSupportWithPreselection>(cd, symsSet, matchAssessor, matchSettings);
}

unique_ptr<TransformSupport> PreselectionOn::createTransformSupport(MatchEngine &me,
																	const IMatchSettings &matchSettings,
																	Mat &resized,
																	Mat &resizedBlurred,
																	vector<vector<unique_ptr<IBestMatch>>> &draftMatches,
																	MatchSupport &matchSupport) const {
	return make_unique<TransformSupportWithPreselection>(me, matchSettings, resized, resizedBlurred,
														 draftMatches, matchSupport);
}

unique_ptr<ClustersSupport> PreselectionOff::createClusterSupport(ITinySymsProvider&,
																  ClusterEngine &ce,
																  VSymData &symsSet) const {
	return make_unique<ClustersSupport>(ce, make_unique<SymsSupport>(), symsSet);
}

unique_ptr<MatchSupport> PreselectionOff::createMatchSupport(CachedData &cd,
															 VSymData&,
															 MatchAssessor&,
															 const IMatchSettings&) const {
	return make_unique<MatchSupport>(cd);
}

unique_ptr<TransformSupport> PreselectionOff::createTransformSupport(MatchEngine &me,
																	 const IMatchSettings &matchSettings,
																	 Mat &resized,
																	 Mat &resizedBlurred,
																	 vector<vector<unique_ptr<IBestMatch>>> &draftMatches,
																	 MatchSupport&) const {
	return make_unique<TransformSupport>(me, matchSettings, resized, resizedBlurred, draftMatches);
}
