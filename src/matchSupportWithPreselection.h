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

#ifndef H_MATCH_SUPPORT_WITH_PRESELECTION
#define H_MATCH_SUPPORT_WITH_PRESELECTION

#include "matchSupport.h"
#include "symDataBase.h"
#include "cachedData.h"

#pragma warning ( push, 0 )

#include <stack>

#pragma warning ( pop )

// Forward declarations
struct IBestMatch;
class MatchAssessor;
struct IMatchSettings;
typedef unsigned CandidateId;
typedef std::stack<CandidateId, std::vector<CandidateId>> CandidatesShortList;

/// Polymorphic support for the MatchEngine and Transformer classes reflecting the preselection mode.
class MatchSupportWithPreselection : public MatchSupport {
protected:
	CachedDataRW cdPresel;				///< cached data corresponding to tiny size symbols
	VSymData &symsSet;					///< the set of normal-size symbols
	MatchAssessor &matchAssessor;		///< match manager based on the enabled matching aspects
	const IMatchSettings &matchSettings;///< match settings

public:
	/// Filling in the rest of the data required when PreselectionByTinySyms == true
	MatchSupportWithPreselection(CachedDataRW &cd_, VSymData &symsSet_,
								 MatchAssessor &matchAssessor_,
								 const IMatchSettings &matchSettings_);

	MatchSupportWithPreselection(const MatchSupportWithPreselection&) = delete;
	MatchSupportWithPreselection(MatchSupportWithPreselection&&) = delete;
	void operator=(const MatchSupportWithPreselection&) = delete;
	void operator=(MatchSupportWithPreselection&&) = delete;

	/// cached data corresponding to the tiny size symbols
	const CachedData& cachedData() const override;

	/// update cached data corresponding to the tiny size symbols
	void updateCachedData(unsigned fontSz, const IFontEngine &fe) override;

	/// @return true if a new better match is found within this short list
	bool improvesBasedOnBatchShortList(CandidatesShortList &&shortList,	///< most promising candidates from current batch of symbols
									   IBestMatch &draftMatch	///< draft for normal symbols (hopefully improved by a match with a symbol from the shortList)
									   ) const;
};

#endif // H_MATCH_SUPPORT_WITH_PRESELECTION
