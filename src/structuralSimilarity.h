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

#ifndef H_STRUCTURAL_SIMILARITY
#define H_STRUCTURAL_SIMILARITY

#include "match.h"

class BlurEngine; // Forward declaration

/**
Selecting a symbol with best structural similarity.

See https://ece.uwaterloo.ca/~z70wang/research/ssim for details.
*/
class StructuralSimilarity : public MatchAspect {
public:
	/// Blurring algorithm used to support this match aspect. The Controller sets it at start.
	static const BlurEngine &supportBlur;

	/// Providing a clue about how complex is this MatchAspect compared to the others
	double relativeComplexity() const override;

protected:
	/// Defines the scoring rule, based on all required fields computed already in MatchParams mp
	double score(const MatchParams &mp, const CachedData &cachedData) const override;

	/// Prepares required fields from MatchParams mp to be able to assess the match
	void fillRequiredMatchParams(const cv::Mat &patch,
								 const SymData &symData,
								 const CachedData &cachedData,
								 MatchParams &mp) const override;

#ifdef UNIT_TESTING // UNIT_TESTING needs the constructors as public
public:
#endif

	StructuralSimilarity(const MatchSettings &cfg);

	REGISTER_MATCH_ASPECT(StructuralSimilarity);
};

#endif // H_STRUCTURAL_SIMILARITY