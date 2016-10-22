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

#ifndef H_FONT_ENGINE
#define H_FONT_ENGINE

#include "pixMapSym.h"
#include "tinySymsProvider.h"

#include <string>

#include <boost/filesystem/path.hpp>
#include <boost/bimap/bimap.hpp>

// Forward declarations
struct IController;
struct IGlyphsProgressTracker;
struct IValidateFont;
struct IPresentCmap;
class SymSettings;
class AbsJobMonitor;

/// FontEngine class wraps some necessary FreeType functionality.
class FontEngine : public ITinySymsProvider {
protected:
	const IController &ctrler;	///< cmap presenting, font validation and glyph preprocessing monitor aspects of the Controller
	const IValidateFont &fontValidator;		///< font validation aspect of the Controller
	const IGlyphsProgressTracker &glyphsProgress;		///< glyph preprocessing monitor  aspect of the Controller
	const IPresentCmap &cmapPresenter;		///< cmap presenting aspect of the Controller

	/// observer of the symbols' loading, filtering and clustering, who reports their progress
	AbsJobMonitor *symsMonitor = nullptr;

	FT_Library library	= nullptr;	///< the FreeType lib
	FT_Face face		= nullptr;	///< a loaded font

	VTinySyms tinySyms;	///< small version of all the symbols from current cmap

	const SymSettings &ss;			///< settings of this font engine
	unsigned encodingIndex = 0U;	///< the index of the selected cmap within face's charmaps array

	/// indices for each unique Encoding within cmaps array
	boost::bimaps::bimap<FT_Encoding, unsigned> uniqueEncs;

	PmsCont symsCont;				///< Container with the PixMapSym-s of current charmap
	unsigned symsCount = 0U;		///< Count of glyphs within current charmap (blanks & duplicates included)

	/**
	Validates a new font file.
	
	When fName is valid, face_ parameter will return the successfully loaded font.
	*/
	bool checkFontFile(const boost::filesystem::path &fontPath, FT_Face &face_) const;
	void setFace(FT_Face face_, const std::string &fontFile_/* = ""*/); ///< Installs a new font

	/**
	Enforces sz as vertical size and determines an optimal horizontal size,
	so that most symbols will widen enough to fill more of the drawing square,
	while preserving the designer's placement.

	@param sz desired size of these symbols
	@param bb [out] estimated future bounding box
	@param symsCount [out] count of glyphs within selected charmap
	@param factorH [out] horizontal scaling factor
	@param factorV [out] vertical scaling factor
	*/
	void adjustScaling(unsigned sz, FT_BBox &bb, unsigned &symsCount, double &factorH, double &factorV);

public:
	/**
	Constructs a FontEngine

	@param ctrler_ font validation and glyph preprocessing monitor aspects of the Controller
	@param ss_ font data
	*/
	FontEngine(const IController &ctrler_, const SymSettings &ss_);
	~FontEngine();

	bool newFont(const std::string &fontFile_);		///< Tries to use the font from 'fontFile_'
	void setFontSz(unsigned fontSz_);				///< Sets the desired font height in pixels

	bool setEncoding(const std::string &encName, bool forceUpdate = false);	///< Sets an encoding by name
	bool setNthUniqueEncoding(unsigned idx);		///< Switches to nth unique encoding

	unsigned upperSymsCount() const;				///< upper bound of symbols count in the cmap
	const std::vector<const PixMapSym>& symsSet() const;	///< get the symsSet
	double smallGlyphsCoverage() const;				///< get coverageOfSmallGlyphs

	const std::string& fontFileName() const;		///< font name provided by Font Dialog
	unsigned uniqueEncodings() const;				///< Returns the count of unique encodings
	const std::string& getEncoding(unsigned *pEncodingIndex = nullptr) const; ///< get encoding
	FT_String* getFamily() const;					///< get font family
	FT_String* getStyle() const;					///< get font style
	std::string getFontType();						///< type of the symbols, independent of font size

	/// (Creates/Loads and) Returns(/Saves) the small versions of all the symbols from current cmap
	const VTinySyms& getTinySyms() override;
	void disposeTinySyms() override; ///< Releases tiny symbols data
	/**
	Determines if fontType was already processed.
	The path to the file supposed to contain the desired tiny symbols data 
	is returned in the tinySymsDataFile parameter.
	*/
	static bool isTinySymsDataSavedOnDisk(const std::string &fontType, 
										  boost::filesystem::path &tinySymsDataFile);

	FontEngine& useSymsMonitor(AbsJobMonitor &symsMonitor_); ///< setting the symbols monitor
};

#endif
