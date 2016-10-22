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
#include "fontEngine.h"
#include "tinySymsDataSerialization.h"
#include "misc.h"

#include <numeric>

#include <boost/filesystem/operations.hpp>
#include FT_TRUETYPE_IDS_H

using namespace std;
using namespace cv;

extern unsigned TinySymsSz();

#ifndef UNIT_TESTING // isTinySymsDataSavedOnDisk will have a different implementation in UnitTesting

#include "appStart.h"

#include <opencv2/imgproc/imgproc.hpp>
using namespace boost::filesystem;

bool FontEngine::isTinySymsDataSavedOnDisk(const string &fontType,
										   boost::filesystem::path &tinySymsDataFile) {
	if(fontType.empty())
		return false;

	tinySymsDataFile = AppStart::dir();
	if(!exists(tinySymsDataFile.append("TinySymsDataSets")))
		create_directory(tinySymsDataFile);

	tinySymsDataFile.append(fontType).concat("_").concat(to_string(TinySymsSz())).
		concat(".tsd"); // Tiny Symbols Data => tsd

	return exists(tinySymsDataFile);
}

#endif // UNIT_TESTING

const VTinySyms& FontEngine::getTinySyms() {
	/*
	Making sure the generation of small symbols doesn't overlap with filling symsCont with normal symbols
	Both involve requesting fonts of different sizes from the same font 'library' object
	and also operating on the same 'face' object while considering the requests.
	*/
	if(!symsCont.isReady())
		THROW_WITH_CONST_MSG(__FUNCTION__ " should be called only after symsCont.setAsReady()!", logic_error);

	if(tinySyms.empty()) {
		VTinySymsIO tinySymsDataSerializer(tinySyms);
		path tinySymsDataFile;
		if(!FontEngine::isTinySymsDataSavedOnDisk(getFontType(), tinySymsDataFile)
				|| !tinySymsDataSerializer.loadFrom(tinySymsDataFile.string())) {
			static const unsigned TinySymsSize = TinySymsSz();

			/*
			Instead of requesting directly fonts of size TinySymsSize, fonts of a larger size are loaded,
			then resized to TinySymsSize, so to get more precise approximations (non-zero fractional parts)
			and to avoid hinting that increases for small font sizes.
			*/
			static const unsigned RefSymsSize = TinySymsSize * (unsigned)TinySym::RatioRefTiny,
								RefSymsSizeX64 = RefSymsSize << 6;
			static const double RefSymsSizeD = (double)RefSymsSize,
								maxGlyphSum = double(255U * RefSymsSize * RefSymsSize);

			static Mat consec(1, RefSymsSize, CV_64FC1), revConsec;
			if(revConsec.empty()) {
				iota(BOUNDS_FOR_ITEM_TYPE(consec, double), (double)0.);
				flip(consec, revConsec, 1);
				revConsec = revConsec.t();
			}
			
			FT_BBox bbox;
			double factorH, factorV;
			adjustScaling(RefSymsSize, bbox, symsCount, factorH, factorV);
			tinySyms.reserve(symsCount);
			const double szHd = factorH * RefSymsSizeX64,
						szVd = factorV * RefSymsSizeX64;
			const FT_ULong szH = (FT_ULong)floor(szHd),
							szV = (FT_ULong)floor(szVd);

			size_t i = 0U;
			FT_UInt idx;
			FT_Size_RequestRec  req;
			req.type = FT_SIZE_REQUEST_TYPE_REAL_DIM;
			req.horiResolution = req.vertResolution = 72U;
			for(FT_ULong c = FT_Get_First_Char(face, &idx); idx != 0; c = FT_Get_Next_Char(face, c, &idx), ++i) {
				FT_Error error = FT_Load_Char(face, c, FT_LOAD_RENDER);
				if(error != FT_Err_Ok)
					THROW_WITH_VAR_MSG("Couldn't load glyph: " + to_string(c) + 
										"  Error: " + to_string(error), invalid_argument);
				FT_GlyphSlot g = face->glyph;
				FT_Bitmap b = g->bitmap;
				const unsigned height = b.rows, width = b.width;
				if(height == 0U || width == 0U) {
					tinySyms.emplace_back((unsigned long)c, i); // blank whose symbol code and index are provided
					continue;
				}

				if(width > RefSymsSize || height > RefSymsSize) {
					// Adjust font size to fit the RefSymsSize x RefSymsSize square
					req.height = (FT_ULong)floor(szVd / max(1., height/RefSymsSizeD));
					req.width = (FT_ULong)floor(szHd / max(1., width/RefSymsSizeD));
					error = FT_Request_Size(face, &req);
					if(error != FT_Err_Ok)
						THROW_WITH_VAR_MSG("Couldn't set font size: " + 
											to_string(req.height) +
											" x " + to_string(req.width) +
											"  Error: " + to_string(error), invalid_argument);

					error = FT_Load_Char(face, c, FT_LOAD_RENDER);
					if(error != FT_Err_Ok)
						THROW_WITH_VAR_MSG("Couldn't load glyph: " + to_string(c) +
											"  Error: " + to_string(error), invalid_argument);
					g = face->glyph;
					b = g->bitmap;

					// Restore font size
					req.height = szV;
					req.width = szH;
					error = FT_Request_Size(face, &req);
					if(error != FT_Err_Ok)
						THROW_WITH_VAR_MSG("Couldn't set font size: " +
											to_string(req.height) +
											" x " + to_string(req.width) +
											"  Error: " + to_string(error), invalid_argument);
				}
				const FT_Int left = g->bitmap_left, top = g->bitmap_top;
				const PixMapSym refSym((unsigned long)c, i, b, (int)left, (int)top, 
									   (int)RefSymsSize, maxGlyphSum, consec, revConsec, bbox);
				tinySyms.emplace_back(refSym);
			}

			tinySymsDataSerializer.saveTo(tinySymsDataFile.string());
		}
	}

	return tinySyms;
}

void FontEngine::disposeTinySyms() {
	tinySyms.clear();
}
