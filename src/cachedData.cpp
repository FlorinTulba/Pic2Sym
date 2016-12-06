/************************************************************************************************
 The application Pic2Sym approximates images by a
 grid of colored symbols with colored backgrounds.

 This file belongs to the Pic2Sym project.

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

#include "cachedData.h"
#include "fontEngine.h"
#include "misc.h"

#include <numeric>

using namespace std;
using namespace cv;

extern const double DirSmooth_DesiredBaseForCenterAndCornerMcs;

namespace {
	const double SQRT2 = sqrt(2.);

	/**
	Visual Studio 2013 doesn't provide thread-safe initialization of function local static variables:
	see 'Magic statics' from https://msdn.microsoft.com/en-us/library/hh567368.aspx .

	The local class from below ensures that the static methods from CachedData are thread-safe by
	creating a file static instance of it which calls all those vulnerable methods from CachedData
	before any other threads do.

	This approach was preferred, since Visual Studio 2015 addressed already the issues and
	Double-Checked Locks or std::call_once seemed like overkill solutions.
	*/
	struct StaticInitializer {
		StaticInitializer() {
			CachedData::unitSquareCenter();
			CachedData::invComplPrefMaxMcDist();
			CachedData::a_mcsOffsetFactor();
			CachedData::b_mcsOffsetFactor();
		}
	};

	/// Computes the constants before the threads start using them
	const StaticInitializer staticInitializer;

} // anonymous namespace

/*
DirectionalSmoothness should depend MOSTLY on the angleFactor, and LESS on mcsOffsetFactor.
See these variables defined and explained in DirectionalSmoothness::score() method.

Here's the case that catches the undesired high impact of mcsOffsetFactor:

Mass center 1 is in the square's center, while the second mass center is in one corner.
Thus, angleFactor has its largest possible value (MaxAngleFactor = 2 * (2-sqrt(2)) = ~1.17),
since the angle between the mc-s is 0 (because the angle between the center and any point is 0).
But mcsOffsetFactor is computed based on a large offset between mc-s: maxMcDist/2.
To mention that maxMcDist is sqrt(2).

As MaxAngleFactor is slightly over 1, it's imperative that mcsOffsetFactor to remain
closely below 1, to convey the message that the angle is indeed a convenient one.
However, computing mcsOffsetFactor like (MaxMcDist - mcsOffset) / (MaxMcDist - PreferredMaxMcDist)
delivers a value ~0.55 for mcsOffset = maxMcDist/2.

The constant DirSmooth_DesiredBaseForCenterAndCornerMcs makes sure that
mcsOffsetFactor will have its value when mcsOffset = maxMcDist/2, instead of 0.55.

mcsOffsetFactor will further be 1 for mcsOffset = PreferredMaxMcDist

Considering the linear rule for mcsOffsetFactor:
mcsOffsetFactor = a * mcsOffset + b

we have following system of 2 equations:
a * PreferredMaxMcDist + b = 1
a * maxMcDist/2        + b = DirSmooth_DesiredBaseForCenterAndCornerMcs / MaxAngleFactor

Solution:
a = (DirSmooth_DesiredBaseForCenterAndCornerMcs / MaxAngleFactor - 1) / (maxMcDist/2 - PreferredMaxMcDist)
b = 1 - a * PreferredMaxMcDist
*/
const double CachedData::a_mcsOffsetFactor() {
	static const double maxMcDist = SQRT2,
				MaxAngleFactor = 2. * (2. - SQRT2),
				NumeratorA_mcsOffsetFactor = DirSmooth_DesiredBaseForCenterAndCornerMcs / MaxAngleFactor - 1.,
				DenominatorA_mcsOffsetFactor = .5 * maxMcDist - preferredMaxMcDist(),
				result = NumeratorA_mcsOffsetFactor / DenominatorA_mcsOffsetFactor;
	return result;
}
const double CachedData::b_mcsOffsetFactor() {
	static const double result = 1. - a_mcsOffsetFactor() * preferredMaxMcDist();
	return result;
}

const Point2d& CachedData::unitSquareCenter() {
	static const Point2d center(.5, .5);
	return center;
}

/// 1 / max possible distance between mass centers: sqrt(2) - preferredMaxMcDist
const double CachedData::invComplPrefMaxMcDist() {
	static const double result = 1. / (SQRT2 - preferredMaxMcDist());
	return result;
}

CachedData::CachedData(bool forTinySyms_/* = false*/) : forTinySyms(forTinySyms_) {}

void CachedData::useNewSymSize(unsigned sz_) {
	const double szd = (double)sz_;
	sz_1 = szd - 1.;

	consec = Mat(1, sz_, CV_64FC1);
	iota(BOUNDS_FOR_ITEM_TYPE(consec, double), (double)0.);
}

void CachedData::update(const FontEngine &fe_) {
	smallGlyphsCoverage = fe_.smallGlyphsCoverage();
}

void CachedData::update(unsigned sz_, const FontEngine &fe_) {
	useNewSymSize(sz_);
	update(fe_);
}
