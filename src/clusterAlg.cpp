/****************************************************************************************
 The application Pic2Sym approximates images by a
 grid of colored symbols with colored backgrounds.

 This file belongs to the Pic2Sym project.

 Copyrights from the libraries used by the program:
 - (c) 2015 Boost (www.boost.org)
   License: <http://www.boost.org/LICENSE_1_0.txt>
            or doc/licenses/Boost.lic
 - (c) 2015 The FreeType Project (www.freetype.org)
   License: <http://git.savannah.gnu.org/cgit/freetype/freetype2.git/plain/docs/FTL.TXT>
	        or doc/licenses/FTL.txt
 - (c) 2015 OpenCV (www.opencv.org)
   License: <http://opencv.org/license.html>
            or doc/licenses/OpenCV.lic
 - (c) 1997-2002 OpenMP Architecture Review Board (www.openmp.org)
   (c) Microsoft Corporation (Visual C++ implementation for OpenMP C/C++ Version 2.0 March 2002)
   See: <https://msdn.microsoft.com/en-us/library/8y6825x5(v=vs.140).aspx>
 
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
 ****************************************************************************************/

#include "noClustering.h"
#include "ttsasClustering.h"
#include "partitionClustering.h"

#include <iostream>

using namespace std;

ClusterAlg& ClusterAlg::algByName(const string &algName) {
	ClusterAlg *pAlg = nullptr;
	if(0 == algName.compare("TTSAS")) {
		static TTSAS_Clustering alg;
		pAlg = &alg;
	} else if(0 == algName.compare("Partition")) {
		static PartitionClustering alg;
		pAlg = &alg;
	} else {
		if(0 != algName.compare("None")) {
			cerr<<"Unaware of clustering algorithm '"<<algName
				<<"'! Therefore no clustering will be used!"<<endl;
		}
		static NoClustering alg;
		pAlg = &alg;
	}
	return *pAlg;
}

ClusterAlg& ClusterAlg::useSymsMonitor(AbsJobMonitor &symsMonitor_) {
	symsMonitor = &symsMonitor_;
	return *this;
}