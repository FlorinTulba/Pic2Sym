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

#ifndef H_PMS_CONT
#define H_PMS_CONT

#include "pmsContBase.h"
#include "symFilterBase.h"

struct IController; // forward declaration
struct IPixMapSym;

/// Convenience container to hold PixMapSym-s of same size
class PmsCont : public IPmsCont {
protected:
	VPixMapSym syms;	///< data for each symbol within current charmap

	// Precomputed entities during reset
	cv::Mat consec;					///< vector of consecutive values 0..fontSz-1
	cv::Mat revConsec;				///< consec reversed

	IController &ctrler;	///< updates Cmap View as soon as there are enough symbols for 1 page

	/**
	Member that allows setting filters to detect symbols with undesired features.
	Passing this field as parameter to a function/method is allowed only in dereferenced form:
	*symFilter
	*/
	std::uniquePtr<ISymFilter> symFilter = std::makeUnique<DefSymFilter>();
	std::map<unsigned, unsigned> removableSymsByCateg; ///< associations: filterId - count of detected syms

	double maxGlyphSum;				///< max sum of a glyph's pixels
	double coverageOfSmallGlyphs;	///< max ratio for small symbols of glyph area / containing area

	unsigned fontSz = 0U;			///< bounding box size

	unsigned blanks = 0U;			///< how many Blank characters were within the charmap
	unsigned duplicates = 0U;		///< how many duplicate symbols were within the charmap

	bool ready = false;				///< is container ready to provide useful data?

	/// If a symbol has a side equal to 0, it is a blank => increment blanks and return true
	bool exactBlank(unsigned height, unsigned width);
	
	/// If a symbol contains almost only white pixels it is nearly blank => increment blanks and return true
	bool nearBlank(const IPixMapSym &pms);

	/// Is the symbol an exact duplicate of one which is already in the syms container? If yes, increment duplicates and return true
	bool isDuplicate(const IPixMapSym &pms);

public:
	PmsCont(IController &ctrler_);
	PmsCont(const PmsCont&) = delete;
	void operator=(const PmsCont&) = delete;

	bool isReady() const override { return ready; }
	unsigned getFontSz() const override;
	unsigned getBlanksCount() const override;
	unsigned getDuplicatesCount() const override;
	const std::map<unsigned, unsigned>& getRemovableSymsByCateg() const override;
	double getCoverageOfSmallGlyphs() const override;
	const VPixMapSym& getSyms() const override;

	/// clears & prepares container for new entries
	void reset(unsigned fontSz_ = 0U, unsigned symsCount = 0U) override;

	/**
	appendSym puts valid glyphs into vector 'syms'.

	Space (empty / full) glyphs are invalid.
	Also updates the count of blanks & duplicates and of any filtered out symbols.
	*/
	void appendSym(FT_ULong c, size_t symIdx, FT_GlyphSlot g, FT_BBox &bb, SymFilterCache &sfc) override;

	void setAsReady() override; ///< No other symbols to append. Statistics can be now computed
};

#endif // H_PMS_CONT
