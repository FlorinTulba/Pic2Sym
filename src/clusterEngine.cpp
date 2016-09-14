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

#include "clusterEngine.h"
#include "misc.h"

#include <set>
#include <string>
#include <iostream>
#include <numeric>

#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

extern const string ClusterAlgName;

namespace {
	/// Creates the vector smallSyms of smaller versions (5x5) of the symbols from symsSet.
	void initSmallSyms(const VSymData &symsSet, vector<const TinySymData> &smallSyms) {
		static const Size SmallSymDim(TinySymData::TinySymSz, TinySymData::TinySymSz);
		static const unsigned DiagsCountU = 2U * TinySymData::TinySymSz - 1U;
		static const double InvDiagsCount = 1. / DiagsCountU,
						InvSmallSymSz = 1. / TinySymData::TinySymSz,
						InvSmallSymArea = 1. / (TinySymData::TinySymSz * TinySymData::TinySymSz);
		
		smallSyms.reserve(symsSet.size());
		const double invInitFontSz = 1. / symsSet[0].symAndMasks[SymData::NEG_SYM_IDX].rows,
					invInitFontArea = invInitFontSz * invInitFontSz;
		for(const auto &symData : symsSet) {
			const Mat &gsi = symData.symAndMasks[SymData::GROUNDED_SYM_IDX]; // double values, 0..1 range
			Mat smallSym, flippedSmallSym,
				hAvgProj, vAvgProj,
				backslashDiagAvgProj(1, DiagsCountU, CV_64FC1, 0.),
				slashDiagAvgProj(1, DiagsCountU, CV_64FC1, 0.);
			resize(gsi, smallSym, SmallSymDim, 0., 0., CV_INTER_AREA);

			// computing average projections
			reduce(smallSym, hAvgProj, 0, CV_REDUCE_AVG);
			reduce(smallSym, vAvgProj, 1, CV_REDUCE_AVG);
			flip(smallSym, flippedSmallSym, 1); // flip around vertical axis
			for(int diagIdx = -TinySymData::TinySymSz+1, i = 0;
					diagIdx < TinySymData::TinySymSz; ++diagIdx, ++i) {
				const Mat backslashDiag = smallSym.diag(diagIdx);
				backslashDiagAvgProj.at<double>(i) = *mean(backslashDiag).val;

				const Mat slashDiag = flippedSmallSym.diag(-diagIdx);
				slashDiagAvgProj.at<double>(i) = *mean(slashDiag).val;
			}

			smallSyms.emplace_back(symData.mc * invInitFontSz, // mapping original mc to the unit square
								   symData.pixelSum * invInitFontArea, // average pixel value (0..1)
								   smallSym * InvSmallSymArea, // all pixels divided by area of tiny sym
								   
								   // all elements of the resulted projections divided by their count
								   hAvgProj * InvSmallSymSz, vAvgProj * InvSmallSymSz,
								   backslashDiagAvgProj * InvDiagsCount, slashDiagAvgProj * InvDiagsCount);
		}
	}

	/// Reorders the clusters by their size - largest ones first and collects the offsets of each cluster
	void computeClusterOffsets(vector<vector<unsigned>> &symsIndicesPerCluster, unsigned clustersCount,
							   VSymData &symsSet, VClusterData &clusters, set<unsigned> &clusterOffsets) {
		// Add the clusters in descending order of their size:

		// Typically, there are only a few clusters larger than 1 element.
		// This partition separates the actual formed clusters from one-of-a-kind elements
		// leaving less work to perform to the sort executed afterwards
		auto itFirstClusterWithOneItem = stable_partition(BOUNDS(symsIndicesPerCluster),
														  [] (const vector<unsigned> &a) {
			return a.size() > 1U; // place actual clusters at the beginning of the vector
		});

		// Stable partition and sort leave the symbols organized in a more pleasant way than using unstable algorithms.
		stable_sort(begin(symsIndicesPerCluster), itFirstClusterWithOneItem,
					[&] (const vector<unsigned> &a, const vector<unsigned> &b) {
			const size_t szA = a.size(), szB = b.size();
			// Sort in descending order of cluster sz and then in ascending order of pixelSum
			// (taken from last cluster member)
			return (szA > szB) || ((szA == szB) && (symsSet[a.back()].pixelSum < symsSet[b.back()].pixelSum));
		});

		// Sort trivial clusters in ascending order of pixelSum
		sort(itFirstClusterWithOneItem, end(symsIndicesPerCluster), // just to appear familiar while visualizing the cmap
			 [&] (const vector<unsigned> &a, const vector<unsigned> &b) {
			return symsSet[a.back()].pixelSum < symsSet[b.back()].pixelSum;
		});

		const unsigned nonTrivialClusters = (unsigned)distance(begin(symsIndicesPerCluster), itFirstClusterWithOneItem),
			symsCount = (unsigned)symsSet.size(),
			clusteredSyms = symsCount - ((unsigned)symsIndicesPerCluster.size() - nonTrivialClusters);
		cout<<"There are "<<nonTrivialClusters
			<<" non-trivial clusters that hold a total of "<<clusteredSyms<<" symbols."<<endl;
		cout<<"Largest cluster contains "<<symsIndicesPerCluster[0].size()<<" symbols"<<endl;

		VSymData newSymsSet;
		newSymsSet.reserve(symsCount);
		for(unsigned i = 0U, offset = 0U; i<clustersCount; ++i) {
			const auto &symsIndices = symsIndicesPerCluster[i];
			const unsigned clusterSz = (unsigned)symsIndices.size();
			clusterOffsets.insert(offset);
			clusters.emplace_back(symsSet, offset, symsIndices); // needs symsSet[symsIndices] !!
			for(const auto idx : symsIndices)
				newSymsSet.push_back(move(symsSet[idx])); // destroys symsSet[idx] !!

			offset += clusterSz;
		}
		clusterOffsets.insert(symsCount); // delimit last cluster
		symsSet = move(newSymsSet);
	}
} // anonymous namespace

ClusterEngine::ClusterEngine() : clustAlg(IClusterAlg::algByName(ClusterAlgName)) {}

void ClusterEngine::process(VSymData &symsSet) {
	assert(!symsSet.empty());

	clusterOffsets.clear(); clusters.clear();

	vector<const TinySymData> smallSyms;
	initSmallSyms(symsSet, smallSyms);

	// cluster smallSyms
	vector<vector<unsigned>> symsIndicesPerCluster;
	const unsigned clustersCount = clustAlg.formGroups(smallSyms, symsIndicesPerCluster);

	computeClusterOffsets(symsIndicesPerCluster, clustersCount, symsSet, clusters, clusterOffsets);
}

ClusterData::ClusterData(const VSymData &symsSet, unsigned idxOfFirstSym_,
						 const vector<unsigned> &clusterSymIndices) : SymData(),
		idxOfFirstSym(idxOfFirstSym_), sz((unsigned)clusterSymIndices.size()) {
	assert(!clusterSymIndices.empty() && !symsSet.empty());
	const Mat &firstNegSym = symsSet[0].symAndMasks[NEG_SYM_IDX];
	const int rows = firstNegSym.rows, cols = firstNegSym.cols;
	double pixelSum = 0.;
	Point2d mc;
	Mat synthesizedSym, negSynthesizedSym(rows, cols, CV_64FC1, Scalar(0.));
	for(const auto clusterSymIdx : clusterSymIndices) {
		const SymData &symData = symsSet[clusterSymIdx];
		Mat negSym;
		symData.symAndMasks[NEG_SYM_IDX].convertTo(negSym, CV_64FC1);
		negSynthesizedSym += negSym;
		pixelSum += symData.pixelSum;
		mc += symData.mc;
	}
	const double invClusterSz = 1./sz;
	negSynthesizedSym *= invClusterSz;
	synthesizedSym = 255. - negSynthesizedSym;
	negSynthesizedSym.convertTo(negSynthesizedSym, CV_8UC1);

	Mat fgMask, bgMask, edgeMask, groundedGlyph, blurOfGroundedGlyph, varianceOfGroundedGlyph;
	double minVal, maxVal; // for very small fonts, minVal might be > 0 and maxVal might be < 255
	computeFields(synthesizedSym, fgMask, bgMask, edgeMask,
				  groundedGlyph, blurOfGroundedGlyph, varianceOfGroundedGlyph,
				  minVal, maxVal);
	const_cast<double&>(this->minVal) = minVal;
	const_cast<double&>(diffMinMax) = maxVal - minVal;
	const_cast<double&>(this->pixelSum) = pixelSum * invClusterSz;
	const_cast<Point2d&>(this->mc) = mc * invClusterSz;
	const_cast<Mat&>(symAndMasks[FG_MASK_IDX]) = fgMask;
	const_cast<Mat&>(symAndMasks[BG_MASK_IDX]) = bgMask;
	const_cast<Mat&>(symAndMasks[EDGE_MASK_IDX]) = edgeMask;
	const_cast<Mat&>(symAndMasks[NEG_SYM_IDX]) = negSynthesizedSym;
	const_cast<Mat&>(symAndMasks[GROUNDED_SYM_IDX]) = groundedGlyph;
	const_cast<Mat&>(symAndMasks[BLURRED_GR_SYM_IDX]) = blurOfGroundedGlyph;
	const_cast<Mat&>(symAndMasks[VARIANCE_GR_SYM_IDX]) = varianceOfGroundedGlyph;
}
