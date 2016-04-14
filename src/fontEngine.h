/****************************************************************************************
 The application Pic2Sym approximates images by a
 grid of colored symbols with colored backgrounds.

 This file was created on 2016-1-8
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

#ifndef H_FONT_ENGINE
#define H_FONT_ENGINE

#include "pixMapSym.h"

#include <string>

#include <boost/filesystem/path.hpp>
#include <boost/bimap/bimap.hpp>

// forward declarations
struct IController;
struct IGlyphsProgressTracker;
struct IValidateFont;
class SymSettings;

/// FontEngine class wraps some necessary FreeType functionality.
class FontEngine {
protected:
	const IController &ctrler;	///< font validation and glyph preprocessing monitor aspects of the Controller
	const IValidateFont &fontValidator;		///< font validation aspect of the Controller
	const IGlyphsProgressTracker &glyphsProgress;		///< glyph preprocessing monitor  aspect of the Controller

	FT_Library library	= nullptr;	///< the FreeType lib
	FT_Face face		= nullptr;	///< a loaded font

	const SymSettings &ss;			///< settings of this font engine
	unsigned encodingIndex = 0U;	///< the index of the selected cmap within face's charmaps array

	/// indices for each unique Encoding within cmaps array
	boost::bimaps::bimap<FT_Encoding, unsigned> uniqueEncs;

	PmsCont symsCont;				///< Container with the PixMapSym-s of current charmap

	/**
	checkFontFile Validates a new font file.
	
	When fName is valid, face_ parameter will return the successfully loaded font.
	*/
	bool checkFontFile(const boost::filesystem::path &fontPath, FT_Face &face_) const;
	void setFace(FT_Face face_, const std::string &fontFile_/* = ""*/); ///< Installs a new font

public:
	/**
	Constructs a FontEngine

	@param ctrler_ font validation and glyph preprocessing monitor aspects of the Controller
	@param ss_ font data
	*/
	FontEngine(const IController &ctrler_, const SymSettings &ss_);
	~FontEngine();
	
	bool newFont(const std::string &fontFile_);		///< Tries to use the font from <fontFile_>
	void setFontSz(unsigned fontSz_);				///< Sets the desired font height in pixels

	bool setEncoding(const std::string &encName, bool forceUpdate = false);	///< Sets an encoding by name
	bool setNthUniqueEncoding(unsigned idx);		///< Switches to nth unique encoding

	const std::vector<const PixMapSym>& symsSet() const;	///< get the symsSet
	double smallGlyphsCoverage() const;				///< get coverageOfSmallGlyphs

	const std::string& fontFileName() const;		///< font name provided by Font Dialog
	unsigned uniqueEncodings() const;				///< Returns the count of unique encodings
	const std::string& getEncoding(unsigned *pEncodingIndex = nullptr) const; ///< get encoding
	FT_String* getFamily() const;					///< get font family
	FT_String* getStyle() const;					///< get font style
};

#endif
