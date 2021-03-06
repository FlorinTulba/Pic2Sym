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

#include "pmsCont.h"
#include "pixMapSym.h"
#include "filledRectanglesFilter.h"
#include "gridBarsFilter.h"
#include "bulkySymsFilter.h"
#include "unreadableSymsFilter.h"
#include "sievesSymsFilter.h"
#include "symFilterCache.h"
#include "controllerBase.h"
#include "misc.h"

#pragma warning ( push, 0 )

#include <numeric>

#pragma warning ( pop )

using namespace std;
using namespace boost;
using namespace cv;

static const double OneMinEPS = 1. - EPS;

PmsCont::PmsCont(IController &ctrler_) :
		ctrler(ctrler_),
		
		// Add any additional filters as 'makeUnique<NewFilter>()' in the last set of unfilled '()'
		symFilter(makeUnique<FilledRectanglesFilter>
				(makeUnique<GridBarsFilter>
				(makeUnique<BulkySymsFilter>
				(makeUnique<UnreadableSymsFilter>
				(makeUnique<SievesSymsFilter>()))))) {}

unsigned PmsCont::getFontSz() const {
	if(!ready)
		THROW_WITH_CONST_MSG(__FUNCTION__  " cannot be called before setAsReady", logic_error);

	return fontSz;
}

unsigned PmsCont::getBlanksCount() const {
	if(!ready)
		THROW_WITH_CONST_MSG(__FUNCTION__  " cannot be called before setAsReady", logic_error);

	return blanks;
}

unsigned PmsCont::getDuplicatesCount() const {
	if(!ready)
		THROW_WITH_CONST_MSG(__FUNCTION__  " cannot be called before setAsReady", logic_error);

	return duplicates;
}

const unordered_map<unsigned, unsigned>& PmsCont::getRemovableSymsByCateg() const {
	if(!ready)
		THROW_WITH_CONST_MSG(__FUNCTION__  " cannot be called before setAsReady", logic_error);

	return removableSymsByCateg;
}

double PmsCont::getCoverageOfSmallGlyphs() const {
	if(!ready)
		THROW_WITH_CONST_MSG(__FUNCTION__  " cannot be called before setAsReady", logic_error);

	return coverageOfSmallGlyphs;
}

const VPixMapSym& PmsCont::getSyms() const {
	if(!ready)
		THROW_WITH_CONST_MSG(__FUNCTION__  " cannot be called before setAsReady", logic_error);

	return syms;
}

void PmsCont::reset(unsigned fontSz_/* = 0U*/, unsigned symsCount/* = 0U*/) {
	ready = false;
	fontSz = fontSz_;
	maxGlyphSum = (double)(255U * fontSz_ * fontSz_);
	blanks = duplicates = 0U;
	coverageOfSmallGlyphs = 0.;

	removableSymsByCateg.clear();
	syms.clear();
	if(symsCount != 0U)
		syms.reserve(symsCount);

	consec = Mat(1, (int)fontSz_, CV_64FC1);
	revConsec.release();

	iota(BOUNDS_FOR_ITEM_TYPE(consec, double), (double)0.);
	flip(consec, revConsec, 1);
	revConsec = revConsec.t();
}

bool PmsCont::exactBlank(unsigned height, unsigned width) {
	if(height > 0U && width > 0U)
		return false;

	++blanks;
	return true;
}

bool PmsCont::nearBlank(const IPixMapSym &pms) {
	if(pms.getAvgPixVal() > EPS && pms.getAvgPixVal() < OneMinEPS)
		return false;
	
	++blanks;
	return true;
}

bool PmsCont::isDuplicate(const IPixMapSym &pms) {
	for(const uniquePtr<const IPixMapSym> &prevPms : syms)
		if(dynamic_cast<const PixMapSym&>(pms) == dynamic_cast<const PixMapSym&>(*prevPms)) {
			++duplicates;
			return true;
		}
	return false;
}

void PmsCont::appendSym(FT_ULong c, size_t symIdx, FT_GlyphSlot g, FT_BBox &bb, SymFilterCache &sfc) {
	assert(!ready); // method shouldn't be called after setAsReady without reset-ing
	
	const FT_Bitmap b = g->bitmap;
	const unsigned height = b.rows, width = b.width;

	// Skip Space characters
	if(exactBlank(height, width))
		return;

	uniquePtr<const PixMapSym> pms = makeUnique<const PixMapSym>(c, symIdx, g->bitmap, g->bitmap_left, g->bitmap_top,
											(int)fontSz, maxGlyphSum, consec, revConsec, bb);
	// discard disguised Space characters
	if(nearBlank(*pms))
		return;

	// Exclude duplicates, as well
	if(isDuplicate(*pms))
		return;

	sfc.setBoundingBox(height, width);

	const optional<unsigned> matchingFilterId = symFilter->matchingFilterId(*pms, sfc);
	if(matchingFilterId) {
		auto it = removableSymsByCateg.find(*matchingFilterId);
		if(it == removableSymsByCateg.end())
			removableSymsByCateg.emplace(*matchingFilterId, 1U);
		else
			++it->second;

		extern const bool PreserveRemovableSymbolsForExamination;
		if(!PreserveRemovableSymbolsForExamination)
			return;

		const_cast<PixMapSym&>(*pms).setRemovable();
	}

	syms.push_back(std::move(pms));

#ifndef UNIT_TESTING
	ctrler.display1stPageIfFull(syms);
#endif // UNIT_TESTING not defined
}

void PmsCont::setAsReady() {
	if(ready)
		return;

	// Determine below max box coverage for smallest glyphs from the kept symsSet.
	// This will be used to favor using larger glyphs when this option is selected.
	extern const double PmsCont_SMALL_GLYPHS_PERCENT;
	const auto smallGlyphsQty = (long)round(syms.size() * PmsCont_SMALL_GLYPHS_PERCENT);

	VPixMapSym::iterator itToNthGlyphSum = next(begin(syms), smallGlyphsQty);
#ifndef AI_REVIEWER_CHECK // AI Reviewer might not parse correctly such lambda-s
	nth_element(begin(syms), itToNthGlyphSum, end(syms),
				[] (const uniquePtr<const IPixMapSym> &first,
				const uniquePtr<const IPixMapSym> &second) {
		return first->getAvgPixVal() < second->getAvgPixVal();
	});

#else // AI_REVIEWER_CHECK defined
	syms[0ULL]->getAvgPixVal();
#endif // AI_REVIEWER_CHECK

	coverageOfSmallGlyphs = (*itToNthGlyphSum)->getAvgPixVal();

	ready = true;
}
