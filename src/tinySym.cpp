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

#include "tinySym.h"
#include "pixMapSymBase.h"

#pragma warning ( push, 0 )

#include <opencv2/imgproc/imgproc.hpp>

#pragma warning ( pop )

using namespace std;
using namespace cv;

extern unsigned TinySymsSz();

namespace {

#ifndef UNIT_TESTING
	/*
	It appears that the following definitions leave all the values on 0 when invoking TinySym's methods from UnitTesting project.
	Therefore using #define clauses for UnitTesting project (see the #else branch).
	*/
	const unsigned TinySymsSize = TinySymsSz(),
				RefSymSz = TinySymsSize * (unsigned)ITinySym::RatioRefTiny,
				DiagsCountTinySym = 2U * TinySymsSize - 1U;
	const double invTinySymSz = 1. / TinySymsSize,
				invTinySymArea = invTinySymSz * invTinySymSz,
				invDiagsCountTinySym = 1. / DiagsCountTinySym;
	const Size SizeTinySyms((int)TinySymsSize, (int)TinySymsSize);

#else // UNIT_TESTING defined

#define TinySymsSize			TinySymsSz()
#define RefSymSz				(TinySymsSize * (unsigned)ITinySym::RatioRefTiny)
#define DiagsCountTinySym		((TinySymsSize << 1) - 1U)
#define invTinySymSz			(1. / TinySymsSize)
#define invTinySymArea			(invTinySymSz * invTinySymSz)
#define invDiagsCountTinySym	(1. / DiagsCountTinySym)
#define SizeTinySyms			Size(TinySymsSize, TinySymsSize)

#endif // UNIT_TESTING
} // anonymous namespace

TinySym::TinySym(unsigned long code_/* = ULONG_MAX*/, size_t symIdx_/* = 0ULL*/) : SymData(code_, symIdx_),
		mat((int)TinySymsSize, (int)TinySymsSize, CV_64FC1, 0.),
		hAvgProj(1, (int)TinySymsSize, CV_64FC1, 0.), vAvgProj((int)TinySymsSize, 1, CV_64FC1, 0.),
		backslashDiagAvgProj(1, (int)DiagsCountTinySym, CV_64FC1, 0.),
		slashDiagAvgProj(1, (int)DiagsCountTinySym, CV_64FC1, 0.) {}

TinySym::TinySym(const IPixMapSym &refSym) : 
		SymData(refSym.getSymCode(), refSym.getSymIdx(), refSym.getAvgPixVal(), refSym.getMc()),
		backslashDiagAvgProj(1, (int)DiagsCountTinySym, CV_64FC1),
		slashDiagAvgProj(1, (int)DiagsCountTinySym, CV_64FC1) {

	const Mat refSymMat = refSym.toMatD01(RefSymSz);

	Mat tinySymMat;
	resize(refSymMat, tinySymMat, SizeTinySyms, 0., 0., INTER_AREA);
	negSym = 255. - 255. * tinySymMat; // keep the double type for negSym of tiny symbols

	SymData::computeFields(tinySymMat, *this, true);

	mat = masks[GROUNDED_SYM_IDX].clone();

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

const Mat& TinySym::getMat() const { return mat; }
const Mat& TinySym::getHAvgProj() const { return hAvgProj; }
const Mat& TinySym::getVAvgProj() const { return vAvgProj; }
const Mat& TinySym::getBackslashDiagAvgProj() const { return backslashDiagAvgProj; }
const Mat& TinySym::getSlashDiagAvgProj() const { return slashDiagAvgProj; }

#ifdef UNIT_TESTING

#undef TinySymsSize
#undef RefSymSz
#undef DiagsCountTinySym
#undef invTinySymSz
#undef invTinySymArea
#undef invDiagsCountTinySym
#undef SizeTinySyms

#endif // UNIT_TESTING defined
