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

#ifndef H_PMS_CONT_BASE
#define H_PMS_CONT_BASE

#include "pixMapSymBase.h"

#pragma warning ( push, 0 )

#include <map>

#include <ft2build.h>
#include FT_FREETYPE_H

#pragma warning ( pop )

struct SymFilterCache; // forward declaration

/// Base for the container holding PixMapSym-s of same size
struct IPmsCont /*abstract*/ {
	virtual bool isReady() const = 0;				///< is container ready to provide useful data?
	virtual unsigned getFontSz() const = 0;			///< bounding box size
	virtual unsigned getBlanksCount() const = 0;	///< how many Blank characters were within the charmap
	virtual unsigned getDuplicatesCount() const = 0;///< how many duplicate symbols were within the charmap

	/// Associations: filterId - count of detected syms
	virtual const std::map<unsigned, unsigned>& getRemovableSymsByCateg() const = 0;

	/// Max ratio for small symbols of glyph area / containing area
	virtual double getCoverageOfSmallGlyphs() const = 0;

	virtual const VPixMapSym& getSyms() const = 0;	///< data for each symbol within current charmap

	/// clears & prepares container for new entries
	virtual void reset(unsigned fontSz_ = 0U, unsigned symsCount = 0U) = 0;

	/**
	appendSym puts valid glyphs into vector 'syms'.

	Space (empty / full) glyphs are invalid.
	Also updates the count of blanks & duplicates and of any filtered out symbols.
	*/
	virtual void appendSym(FT_ULong c, size_t symIdx, FT_GlyphSlot g, FT_BBox &bb, SymFilterCache &sfc) = 0;

	virtual void setAsReady() = 0; ///< No other symbols to append. Statistics can be now computed

	virtual ~IPmsCont() = 0 {}
};

#endif // H_PMS_CONT_BASE
