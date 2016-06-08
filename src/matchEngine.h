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

#ifndef H_MATCH_ENGINE
#define H_MATCH_ENGINE

#include <set>

#include "match.h"
#include "fontEngine.h"

// forward declarations
struct CachedData;
struct MatchParams;
struct BestMatch;
class Patch;

class Settings; // forward declaration

/// MatchEngine finds best match for a patch based on current settings and symbols set.
class MatchEngine : public IMatch {
public:
	// Displaying the symbols requires dividing them into pages (ranges using iterators)
	typedef VSymData::const_iterator VSymDataCIt;
	typedef std::pair< VSymDataCIt, VSymDataCIt > VSymDataCItPair;

protected:
	const Settings &cfg;		///< settings for the engine
	FontEngine &fe;				///< symbols set manager
	std::string symsIdReady;	///< type of symbols ready to use for transformation
	VSymData symsSet;			///< set of most information on each symbol
	VClusterData clusters;		///< the clustered symbols
	std::set<unsigned> clusterOffsets;	///< start index in clusters where a new cluster starts

	// matching aspects
	std::vector<std::shared_ptr<MatchAspect>> availAspects;	///< all the available aspects
	std::vector<MatchAspect*> enabledAspects;				///< only the enabled aspects

	CachedData cachedData;	///< data precomputed by getReady before performing the matching series

public:
	MatchEngine(const Settings &cfg_, FontEngine &fe_);

	std::string getIdForSymsToUse(); ///< type of the symbols determined by fe & cfg

	/// Needed to display the cmap - returns a pair of symsSet iterators
	VSymDataCItPair getSymsRange(unsigned from, unsigned count) const;
	unsigned getSymsCount() const;	///< to be displayed in CmapView's status bar
	const std::set<unsigned>& getClusterOffsets() const;

	void updateSymbols();	///< using different charmap - also useful for displaying these changes
	void getReady();		///< called before a series of findBetterMatch

	/// Returns true if a new better match was found.
	bool findBetterMatch(BestMatch &draftMatch, unsigned fromSymIdx, unsigned upperSymIdx) const;

	/// scores the match between a gray patch and a symbol based on all enabled aspects
	double assessMatch(const cv::Mat &patch,
					   const SymData &symData,
					   MatchParams &mp) const override; // IMatch override

	bool usesUnicode() const; /// Unicode glyphs are logged as symbols, the rest as their code
};

#endif