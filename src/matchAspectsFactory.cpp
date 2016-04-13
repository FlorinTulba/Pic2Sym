/****************************************************************************************
 The application Pic2Sym approximates images by a
 grid of colored symbols with colored backgrounds.

 This file was created on 2016-4-13
 and belongs to the Pic2Sym project.

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

#include "matchAspectsFactory.h"
#include "matchAspects.h"

using namespace std;

std::shared_ptr<MatchAspect> MatchAspectsFactory::create(const string &aspectName,
													const CachedData &cachedData,
													const MatchSettings &ms) {
#define HANDLE_MATCH_ASPECT(Aspect) \
	if(aspectName.compare(#Aspect) == 0) \
		return std::make_shared<Aspect>(cachedData, ms)

	HANDLE_MATCH_ASPECT(StructuralSimilarity);
	HANDLE_MATCH_ASPECT(FgMatch);
	HANDLE_MATCH_ASPECT(BgMatch);
	HANDLE_MATCH_ASPECT(EdgeMatch);
	HANDLE_MATCH_ASPECT(BetterContrast);
	HANDLE_MATCH_ASPECT(GravitationalSmoothness);
	HANDLE_MATCH_ASPECT(DirectionalSmoothness);
	HANDLE_MATCH_ASPECT(LargerSym);

#undef HANDLE_ASPECT

	throw invalid_argument(aspectName + " is an invalid aspect name!");
}