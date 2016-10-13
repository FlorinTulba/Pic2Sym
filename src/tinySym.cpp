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

#include "tinySym.h"
#include "pixMapSym.h"

#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

extern unsigned TinySymsSz();

namespace {

#ifndef UNIT_TESTING
	/*
	It appears that the following definitions leave all the values on 0 when invoking TinySym's methods from UnitTesting project.
	Therefore using #define clauses for UnitTesting project (see the #else branch).
	*/
	static const unsigned TinySymsSize = TinySymsSz(),
						RefSymSz = TinySymsSize * (unsigned)TinySym::RatioRefTiny,
						DiagsCountTinySym = 2U * TinySymsSize - 1U;
	static const double invTinySymSz = 1. / TinySymsSize,
						invTinySymArea = invTinySymSz * invTinySymSz,
						invRefSymSz = 1. / RefSymSz,
						invRefSymArea = invRefSymSz * invRefSymSz,
						invDiagsCountTinySym = 1. / DiagsCountTinySym;
	static const Size SizeTinySyms(TinySymsSize, TinySymsSize);

#else // UNIT_TESTING is defined

#define TinySymsSize			TinySymsSz()
#define RefSymSz				(TinySymsSize * (unsigned)TinySym::RatioRefTiny)
#define DiagsCountTinySym		((TinySymsSize << 1) - 1U)
#define invTinySymSz			(1. / TinySymsSize)
#define invTinySymArea			(invRefSymSz * invRefSymSz)
#define invRefSymSz				(1. / RefSymSz)
#define invRefSymArea			(invRefSymSz * invRefSymSz)
#define invDiagsCountTinySym	(1. / DiagsCountTinySym)
#define SizeTinySyms			Size(TinySymsSize, TinySymsSize)

#endif // UNIT_TESTING
} // anonymous namespace

TinySym::TinySym() :
		mat(TinySymsSize, TinySymsSize, CV_64FC1, 0.),
		hAvgProj(1, TinySymsSize, CV_64FC1, 0.), vAvgProj(TinySymsSize, 1, CV_64FC1, 0.),
		backslashDiagAvgProj(1, DiagsCountTinySym, CV_64FC1, 0.), slashDiagAvgProj(1, DiagsCountTinySym, CV_64FC1, 0.) {}

TinySym::TinySym(const PixMapSym &refSym) :
		mc(refSym.mc * invRefSymSz), avgPixVal(refSym.glyphSum * invRefSymArea),
		backslashDiagAvgProj(1, DiagsCountTinySym, CV_64FC1), slashDiagAvgProj(1, DiagsCountTinySym, CV_64FC1) {
	const Mat refSymMat = refSym.toMatD01(RefSymSz);
	double minVal;
	minMaxIdx(refSymMat, &minVal);
	const Mat groundedRefSymMat = (minVal == 0.) ? refSymMat : (refSymMat - minVal);
	resize(groundedRefSymMat, mat, SizeTinySyms, 0., 0., INTER_AREA);

	// computing average projections
	reduce(mat, hAvgProj, 0, CV_REDUCE_AVG);
	reduce(mat, vAvgProj, 1, CV_REDUCE_AVG);

	Mat flippedMat;
	flip(mat, flippedMat, 1); // flip around vertical axis
	for(int diagIdx = -(int)TinySymsSize+1, i = 0;
			diagIdx < (int)TinySymsSize; ++diagIdx, ++i) {
		const Mat backslashDiag = mat.diag(diagIdx);
		backslashDiagAvgProj.at<double>(i) = *mean(backslashDiag).val;

		const Mat slashDiag = flippedMat.diag(-diagIdx);
		slashDiagAvgProj.at<double>(i) = *mean(slashDiag).val;
	}

	// Ensuring the sum of all elements of the following matrices is in [0..1] range
	mat *= invTinySymArea;
	hAvgProj *= invTinySymSz;
	vAvgProj *= invTinySymSz;
	backslashDiagAvgProj	*= invDiagsCountTinySym;
	slashDiagAvgProj		*= invDiagsCountTinySym;
}

TinySym::TinySym(const Point2d &mc_, double avgPixVal_, const Mat &mat_,
				 const Mat &hAvgProj_, const Mat &vAvgProj_,
				 const Mat &backslashDiagAvgProj_, const Mat &slashDiagAvgProj_) :
	mc(mc_), avgPixVal(avgPixVal_), mat(mat_), hAvgProj(hAvgProj_), vAvgProj(vAvgProj_),
	backslashDiagAvgProj(backslashDiagAvgProj_), slashDiagAvgProj(slashDiagAvgProj_) {}

#ifdef UNIT_TESTING

#undef TinySymsSize
#undef RefSymSz
#undef DiagsCountTinySym
#undef invTinySymSz
#undef invTinySymArea
#undef invRefSymSz
#undef invRefSymArea
#undef invDiagsCountTinySym
#undef SizeTinySyms

#endif // UNIT_TESTING
