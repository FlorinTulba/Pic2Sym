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

#include "unreadableSymsFilter.h"
#include "pixMapSymBase.h"
#include "symFilterCache.h"
#include "misc.h"

#pragma warning ( push, 0 )

#include <opencv2/imgproc/imgproc.hpp>

#pragma warning ( pop )

using namespace std;
using namespace cv;

UnreadableSymsFilter::UnreadableSymsFilter(unique_ptr<ISymFilter> nextFilter_/* = nullptr*/) :
		TSymFilter(3U, "less readable symbols", std::move(nextFilter_)) {}

bool UnreadableSymsFilter::isDisposable(const IPixMapSym &pms, const SymFilterCache &sfc) {
	if(!isEnabled())
		THROW_WITH_CONST_MSG(__FUNCTION__ " should be called only for enabled filters!", logic_error);

	// Usually, fonts of size >= 20 are quite readable
	if(sfc.szU >= 20)
		return false;

	// Usually unreadable syms are not small
	extern const double MinAreaRatioForUnreadableSymsBB;
	if(sfc.bbAreaD < MinAreaRatioForUnreadableSymsBB * sfc.areaD)
		return false;

	const int winSideSE = (sfc.szU > 15U) ? 5 : 3; // masks need to be larger for larger fonts
	const Size winSE(winSideSE, winSideSE);

	Mat glyph = pms.toMatD01(sfc.szU), thresh, glyphBinAux;

	// adaptiveThreshold has some issues on uniform areas, so here's a customized implementation
#pragma warning ( disable : WARN_THREAD_UNSAFE )
	static const Point defAnchor(-1, -1);
#pragma warning ( default : WARN_THREAD_UNSAFE )

	boxFilter(glyph, thresh, -1, winSE, defAnchor, true, BORDER_CONSTANT);
	extern const double StillForegroundThreshold, ForegroundThresholdDelta;
	double toSubtract = StillForegroundThreshold;
	if(sfc.szU < 15U) {
		// for small fonts, thresholds should be much lower, to encode perceived pixel agglomeration
		const double delta = (15U-sfc.szU) * ForegroundThresholdDelta/255.;
		toSubtract += delta;
	}

	// lower the threshold, but keep all values positive
	thresh -= toSubtract;
	thresh.setTo(0., thresh < 0.);
	glyphBinAux = glyph > thresh;

	// pad the thresholded matrix with 0-s all around, to allow distanceTransform consider the borders as 0-s
	const int frameSz = (int)sfc.szU + 2;
	const Range innerFrame(1, (int)sfc.szU + 1);
	Mat glyphBin = Mat(frameSz, frameSz, CV_8UC1, Scalar(0U)), depth;
	glyphBinAux.copyTo((const Mat&)Mat(glyphBin, innerFrame, innerFrame));
	distanceTransform(glyphBin, depth, DIST_L2, DIST_MASK_PRECISE);

	double maxValAllowed = 1.9; // just below 2 (2 means there are sections at least 3 pixels thick)
	if(sfc.szU > 10U)
		maxValAllowed = sfc.szU/5.; // larger symbols have their core in a thicker shell (>2)

	Mat symCore = depth > maxValAllowed;
	const unsigned notAllowedCount = (unsigned)countNonZero(symCore);
	if(notAllowedCount > sfc.szU - 6U)
		return true; // limit the count of symCore pixels

	double maxVal; // largest symbol depth
	minMaxIdx(depth, nullptr, &maxVal, nullptr, nullptr, symCore);

	return maxVal - maxValAllowed > 1.; // limit the depth of the symCore
}