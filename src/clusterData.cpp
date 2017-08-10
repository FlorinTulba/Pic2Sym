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

#include "clusterData.h"
#include "symbolsSupport.h"

using namespace std;
using namespace cv;

ClusterData::ClusterData(const VSymData &symsSet, unsigned idxOfFirstSym_,
						 const vector<unsigned> &clusterSymIndices,
						 SymsSupport &symsSupport) : SymData(),
		 idxOfFirstSym(idxOfFirstSym_), sz((unsigned)clusterSymIndices.size()) {
	assert(!clusterSymIndices.empty() && !symsSet.empty());
	const double invClusterSz = 1./sz;
	const Mat &firstNegSym = symsSet[0]->getNegSym();
	const int symSz = firstNegSym.rows;
	double avgPixVal_ = 0.;
	Point2d mc_;
	vector<const ISymData*> clusterSyms; clusterSyms.reserve((size_t)sz);

	for(const auto clusterSymIdx : clusterSymIndices) {
		const ISymData &symData = *symsSet[clusterSymIdx];
		clusterSyms.push_back(&symData);

		// avgPixVal and mc are taken from the normal-size symbol (guaranteed to be non-blank)
		avgPixVal_ += symData.getAvgPixVal();
		mc_ += symData.getMc();
	}
	avgPixVal = avgPixVal_ * invClusterSz;
	mc = mc_ * invClusterSz;

	Mat synthesizedSym;
	symsSupport.computeClusterRepresentative(clusterSyms, symSz, invClusterSz, synthesizedSym, negSym);

	SymData::computeFields(synthesizedSym, *this, symsSupport.usingTinySymbols());
}

ClusterData::ClusterData(ClusterData &&other) : SymData(move(other)),
idxOfFirstSym(other.idxOfFirstSym), sz(other.sz) {}

unsigned ClusterData::getIdxOfFirstSym() const { return idxOfFirstSym; }
unsigned ClusterData::getSz() const { return sz; }
