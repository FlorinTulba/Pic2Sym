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

#include "cachedData.h"
#include "fontEngine.h"

#include <numeric>

using namespace std;
using namespace cv;

const double CachedData::sdevMaxFgBg = 127.5;
const double CachedData::sdevMaxEdge = 255.;

void CachedData::useNewSymSize(unsigned sz_) {
	sz = sz_;
	sz_1 = sz - 1U;
	sz2 = (double)sz * sz;

	preferredMaxMcDist = sz / 8.;
	complPrefMaxMcDist = sz_1 * sqrt(2) - preferredMaxMcDist;
	patchCenter = Point2d(sz_1, sz_1) / 2.;

	consec = Mat(1, sz, CV_64FC1);
	iota(consec.begin<double>(), consec.end<double>(), 0.);
}

void CachedData::update(unsigned sz_, const FontEngine &fe_) {
	useNewSymSize(sz_);

	smallGlyphsCoverage = fe_.smallGlyphsCoverage();
}