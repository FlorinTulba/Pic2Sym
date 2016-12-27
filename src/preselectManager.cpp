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
 
 (c) 2016 Florin Tulba <florintulba@yahoo.com>

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
#include "matchEngine.h"
#include "transform.h"
#include "matchParams.h"
#include "symbolsSupportWithPreselection.h"
#include "clusterSupportWithPreselection.h"
#include "matchSupportWithPreselection.h"
#include "transformSupportWithPreselection.h"
#include "settings.h"

using namespace std;

extern const bool PreselectionByTinySyms;

PreselManager::PreselManager(MatchEngine &me, Transformer &tr) :
		symsSupport_(PreselectionByTinySyms ?
			new SymsSupportWithPreselection() :
			new SymsSupport()),
		clustersSupport_(PreselectionByTinySyms ?
			new ClustersSupportWithPreselection(me.fe, me.ce, *symsSupport_, me.symsSet) :
			new ClustersSupport(me.ce, *symsSupport_, me.symsSet)),
		matchSupport_(PreselectionByTinySyms ?
			new MatchSupportWithPreselection(me.cachedData, me.symsSet,
											me.matchAssessor, me.cfg.matchSettings()) :
			new MatchSupport(me.cachedData)),
		transfSupport(PreselectionByTinySyms ?
			new TransformSupportWithPreselection(me, me.cfg.matchSettings(),
												tr.resized, tr.resizedBlurred,
												tr.draftMatches, *matchSupport_) :
			new TransformSupport(me, me.cfg.matchSettings(),
								tr.resized, tr.resizedBlurred,
								tr.draftMatches)) {
	me.ce.supportedBy(*clustersSupport_);
}

PreselManager::~PreselManager() {
	delete transfSupport;
	delete matchSupport_;
	delete clustersSupport_;
	delete symsSupport_;
}

SymsSupport& PreselManager::symsSupport() const {
	return *symsSupport_;
}

ClustersSupport& PreselManager::clustersSupport() const {
	return *clustersSupport_;
}

MatchSupport& PreselManager::matchSupport() const {
	return *matchSupport_;
}

TransformSupport& PreselManager::transformSupport() const {
	return *transfSupport;

}