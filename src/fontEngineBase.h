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

#ifndef H_FONT_ENGINE_BASE
#define H_FONT_ENGINE_BASE

#include "tinySymsProvider.h"
#include "pixMapSymBase.h"

#pragma warning ( push, 0 )

#include "std_string.h"

#include <ft2build.h>
#include FT_FREETYPE_H

#pragma warning ( pop )

class AbsJobMonitor; // forward declaration

/// FontEngine class wraps some necessary FreeType functionality.
struct IFontEngine /*abstract*/ : ITinySymsProvider {
	virtual void invalidateFont() = 0;	///< When unable to process a font type, invalidate it completely

	virtual bool newFont(const std::stringType &fontFile_) = 0;	///< Tries to use the font from 'fontFile_'
	virtual void setFontSz(unsigned fontSz_) = 0;				///< Sets the desired font height in pixels

	virtual bool setEncoding(const std::stringType &encName, bool forceUpdate = false) = 0;	///< Sets an encoding by name
	virtual bool setNthUniqueEncoding(unsigned idx) = 0;		///< Switches to nth unique encoding

	virtual unsigned upperSymsCount() const = 0;				///< upper bound of symbols count in the cmap
	virtual const VPixMapSym& symsSet() const = 0;				///< get the symsSet
	virtual double smallGlyphsCoverage() const = 0;				///< get coverageOfSmallGlyphs

	virtual const std::stringType& fontFileName() const = 0;	///< font name provided by Font Dialog
	virtual unsigned uniqueEncodings() const = 0;				///< Returns the count of unique encodings
	virtual const std::stringType& getEncoding(unsigned *pEncodingIndex = nullptr) const = 0; ///< get encoding
	virtual FT_String* getFamily() const = 0;					///< get font family
	virtual FT_String* getStyle() const = 0;					///< get font style
	virtual std::stringType getFontType() = 0;					///< type of the symbols, independent of font size

	virtual IFontEngine& useSymsMonitor(AbsJobMonitor &symsMonitor_) = 0; ///< setting the symbols monitor

	virtual ~IFontEngine() = 0 {}
};

#endif // H_FONT_ENGINE_BASE
