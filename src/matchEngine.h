/************************************************************************************************
 The application Pic2Sym approximates images by a
 grid of colored symbols with colored backgrounds.

 This file belongs to the Pic2Sym project.

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
 ***********************************************************************************************/

#ifndef H_MATCH_ENGINE
#define H_MATCH_ENGINE

#include "fontEngine.h"
#include "clusterEngine.h"
#include "cachedData.h"
#include "preselectSyms.h"
#include "countSkippedAspects.h"

#include <valarray>

// Forward declarations
struct MatchParams;
struct BestMatch;
struct Patch;
class Settings;
class MatchAspect;

/// MatchEngine finds best match for a patch based on current settings and symbols set.
class MatchEngine {
public:
	// Displaying the symbols requires dividing them into pages (ranges using iterators)
	typedef VSymData::const_iterator VSymDataCIt;
	typedef std::pair< VSymDataCIt, VSymDataCIt > VSymDataCItPair;

protected:
	const Settings &cfg;		///< settings for the engine
	FontEngine &fe;				///< symbols set manager

	/// observer of the symbols' loading, filtering and clustering, who reports their progress
	AbsJobMonitor *symsMonitor = nullptr;

	std::string symsIdReady;	///< type of symbols ready to use for transformation
	VSymData symsSet;			///< set of most information on each symbol
	VSymData tinySymsSet;		///< set of most information on each tiny symbol
	ClusterEngine ce;			///< clusters manager
	
	/// Should the symbols be compared against the patches individually, or first by clusters?
	bool matchByClusters = false;

	CachedData cachedDataForTinySyms;	///< data precomputed for tiny symbols preselection

	// matching aspects
	std::vector<std::shared_ptr<MatchAspect>> availAspects;	///< all the available aspects
	std::vector<const MatchAspect*> enabledAspects;			///< the enabled aspects only
	size_t enabledAspectsCount = 0U;						///< count of the enabled aspects

#ifdef UNIT_TESTING // UnitTesting project needs access to cachedData and invMaxIncreaseFactors
public:
#endif // UNIT_TESTING
	CachedData cachedData;	///< data precomputed by getReady before performing the matching series

	std::valarray<double> invMaxIncreaseFactors; ///< 1 over (max possible increase of the score based on remaining aspects)

public:
	MatchEngine(const Settings &cfg_, FontEngine &fe_);

	std::string getIdForSymsToUse(); ///< type of the symbols determined by fe & cfg

	/// Needed to display the cmap - returns a pair of symsSet iterators
	VSymDataCItPair getSymsRange(unsigned from, unsigned count) const;
	unsigned getSymsCount() const;	///< to be displayed in CmapView's status bar
	const std::set<unsigned>& getClusterOffsets() const;

	void updateSymbols();	///< using different charmap - also useful for displaying these changes
	void getReady();		///< called before a series of improvesBasedOnBatch

	/// @return true if a new better match is found within the new batch of symbols
	bool improvesBasedOnBatch(unsigned fromSymIdx,		///< start of the batch
							  unsigned upperSymIdx,		///< end of the batch (exclusive)
							  BestMatch &draftMatch,	///< draft for normal/tiny symbols (hopefully improved by a match with a symbol from the new batch)
							  TopCandidateMatches *tcm = nullptr ///< preselection manager when performing tiny symbols preselection
							  ) const;

	/// @return true if a new better match is found within this short list
	bool improvesBasedOnBatchShortList(CandidatesShortList &shortList,	///< most promising candidates from current batch of symbols
									   BestMatch &draftMatch	///< draft for normal symbols (hopefully improved by a match with a symbol from the shortList)
									   ) const;

	/// Determines if symData is a better match for patch than previous matching symbol.
	bool isBetterMatch(const cv::Mat &patch,	///< the patch whose approximation through a symbol is performed
					   const SymData &symData,	///< data of the new symbol/cluster compared to the patch
					   const CachedData &cd,	///< precomputed values
					   const std::valarray<double> &scoresToBeat,///< scores after each aspect that beat the current best match
					   MatchParams &mp,			///< matching parameters resulted from the comparison
					   double &score			///< achieved score of the new assessment
					   ) const;

	bool usesUnicode() const; /// Unicode glyphs are logged as symbols, the rest as their code

	MatchEngine& useSymsMonitor(AbsJobMonitor &symsMonitor_); ///< setting the symbols monitor

	const std::vector<std::shared_ptr<MatchAspect>>& availMatchAspects() const;	///< all the available aspects

	void newlyEnabledMatchAspect();		///< increments enabledAspectsCount
	void newlyDisabledMatchAspect();	///< decrements enabledAspectsCount
	
	void updateEnabledMatchAspectsCount();		///< updates enabledAspectsCount by checking which aspects are enabled
	
	size_t enabledMatchAspectsCount() const;	///< provides enabledAspectsCount

#ifdef MONITOR_SKIPPED_MATCHING_ASPECTS
	mutable size_t totalIsBetterMatchCalls = 0U; ///< used for reporting skipped aspects
	mutable std::vector<size_t> skippedAspects; ///< used for reporting skipped aspects

	/// While reporting, the particular aspects that were used during the transformation are required
	const std::vector<const MatchAspect*>& getEnabledAspects() const { return enabledAspects; }
#endif // MONITOR_SKIPPED_MATCHING_ASPECTS
};

#endif // H_MATCH_ENGINE