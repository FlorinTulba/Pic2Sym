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

#ifndef H_MATCH_ENGINE
#define H_MATCH_ENGINE

#include "matchEngineBase.h"
#include "symDataBase.h"
#include "cachedData.h"

// Forward declarations
struct ISettings;
struct IClusterEngine;
struct ICmapPerspective;
struct IFontEngine;
class MatchAspect;

/// MatchEngine finds best match for a patch based on current settings and symbols set.
class MatchEngine : public IMatchEngine {
protected:
	const ISettings &cfg;		///< settings for the engine
	IFontEngine &fe;			///< symbols set manager
	ICmapPerspective &cmP;		///< reorganized symbols to be visualized within the cmap viewer

	/// observer of the symbols' loading, filtering and clustering, who reports their progress
	AbsJobMonitor *symsMonitor = nullptr;

	std::stringType symsIdReady;	///< type of symbols ready to use for transformation

#ifdef UNIT_TESTING // UnitTesting project needs access to following fields
public:
#endif // UNIT_TESTING defined
	VSymData symsSet;				///< set of most information on each symbol
	CachedDataRW cachedData;		///< data precomputed by matchSupport before performing the matching series
	MatchAssessor &matchAssessor;	///< match manager based on the enabled matching aspects

protected:
	const std::uniquePtr<IClusterEngine> ce;			///< clusters manager

	// Keep this below the fields, as it depends on them
	const std::uniquePtr<IMatchSupport> matchSupport;	///< cached data management

	std::vector<const std::uniquePtr<const MatchAspect>> availAspects;	///< all the available aspects

public:
	MatchEngine(const ISettings &cfg_, IFontEngine &fe_, ICmapPerspective &cmP_);
	MatchEngine(const MatchEngine&) = delete;
	void operator=(const MatchEngine&) = delete;

	std::stringType getIdForSymsToUse() override; ///< type of the symbols determined by fe & cfg

	unsigned getSymsCount() const override;	///< to be displayed in CmapView's status bar

	const MatchAssessor& assessor() const override; ///< access to the const methods of the matchAssessor

	IMatchSupport& support() override; ///< access to matchSupport

	void updateSymbols() override;	///< using different charmap - also useful for displaying these changes
	void getReady() override;		///< called before a series of improvesBasedOnBatch

	/// @return true if a new better match is found within the new batch of symbols
	bool improvesBasedOnBatch(unsigned fromSymIdx,			///< start of the batch
							  unsigned upperSymIdx,			///< end of the batch (exclusive)
							  IBestMatch &draftMatch,		///< draft for normal/tiny symbols (hopefully improved by a match with a symbol from the new batch)
							  MatchProgress &matchProgress	///< observer notified for each new improved match
							  ) const override;

	bool usesUnicode() const override; ///< Unicode glyphs are logged as symbols, the rest as their code

	const bool& isClusteringUseful() const override; ///< Clustering should be avoided when the obtained clusters are really small

	MatchEngine& useSymsMonitor(AbsJobMonitor &symsMonitor_) override;		///< setting the symbols monitor
};

#endif // H_MATCH_ENGINE
