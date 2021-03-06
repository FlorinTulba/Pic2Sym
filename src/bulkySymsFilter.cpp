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
 
 (c) 2016-2019 Florin Tulba <florintulba@yahoo.com>

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

#include "bulkySymsFilter.h"
#include "pixMapSymBase.h"
#include "symFilterCache.h"
#include "misc.h"

#pragma warning ( push, 0 )

#include <opencv2/imgproc/imgproc.hpp>

#pragma warning ( pop )

using namespace std;
using namespace cv;

BulkySymsFilter::BulkySymsFilter(uniquePtr<ISymFilter> nextFilter_/* = nullptr*/) :
		TSymFilter(2U, "bulky symbols", std::move(nextFilter_)) {}

bool BulkySymsFilter::isDisposable(const IPixMapSym &pms, const SymFilterCache &sfc) {
	if(!isEnabled())
		THROW_WITH_CONST_MSG(__FUNCTION__ " should be called only for enabled filters!", logic_error);

#pragma warning ( disable : WARN_THREAD_UNSAFE )
	static const auto compErMaskSide = [] (unsigned fontSz) {
		return max(3, (((int)fontSz/2) | 1));
	};
	static unordered_map<int, Mat> circleMasks;
#pragma warning ( default : WARN_THREAD_UNSAFE )

	if(min(pms.getRows(), pms.getCols()) < (unsigned)compErMaskSide(sfc.getSzU()))
		return false;

	if(circleMasks.empty()) {
		extern const unsigned Settings_MAX_FONT_SIZE;
		for(int maskSide = 3, maxMaskSide = compErMaskSide(Settings_MAX_FONT_SIZE);
			maskSide <= maxMaskSide; maskSide += 2)
			circleMasks[maskSide] = getStructuringElement(MORPH_ELLIPSE, Size(maskSide, maskSide));
	}

	const Mat narrowGlyph = pms.asNarrowMat();
	Mat processed;

/*
	// Close with a small disk to fill any minor gaps.
	static const auto compCloseMaskSide = [] (unsigned fontSz) {
		return max(3, (((int)fontSz/6) | 1));
	};
	static const Point defAnchor(-1, -1);
	morphologyEx(narrowGlyph, processed, MORPH_CLOSE, circleMasks[compCloseMaskSide(sfc.getSzU())],
				defAnchor, 1, BORDER_CONSTANT, Scalar(0.));
*/

	// Erode with a large disk to detect large filled areas.
	erode(narrowGlyph, processed, circleMasks[compErMaskSide(sfc.getSzU())]);

	const bool result = countNonZero(processed > 45) > 0;
	return result;
}
