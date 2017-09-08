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

#ifndef H_MATCH_ENGINE_BASE
#define H_MATCH_ENGINE_BASE

#pragma warning ( push, 0 )

#include "std_string.h"

#pragma warning ( pop )

// Forward declarations
struct IBestMatch;
struct IMatchSupport;
class MatchAssessor;
class MatchProgress;
class AbsJobMonitor;

/// MatchEngine finds best match for a patch based on current settings and symbols set.
struct IMatchEngine /*abstract*/ {
	virtual std::stringType getIdForSymsToUse() = 0; ///< type of the symbols determined by fe & cfg

	virtual unsigned getSymsCount() const = 0;	///< to be displayed in CmapView's status bar

	virtual const MatchAssessor& assessor() const = 0; ///< access to the const methods of the matchAssessor

	virtual IMatchSupport& support() = 0; ///< access to matchSupport

	virtual void updateSymbols() = 0;	///< using different charmap - also useful for displaying these changes
	virtual void getReady() = 0;		///< called before a series of improvesBasedOnBatch

	/// @return true if a new better match is found within the new batch of symbols
	virtual bool improvesBasedOnBatch(unsigned fromSymIdx,			///< start of the batch
									  unsigned upperSymIdx,			///< end of the batch (exclusive)
									  IBestMatch &draftMatch,		///< draft for normal/tiny symbols (hopefully improved by a match with a symbol from the new batch)
									  MatchProgress &matchProgress	///< observer notified for each new improved match
									  ) const = 0;

	virtual bool usesUnicode() const = 0; ///< Unicode glyphs are logged as symbols, the rest as their code

	virtual const bool& isClusteringUseful() const = 0; ///< Clustering should be avoided when the obtained clusters are really small

	virtual IMatchEngine& useSymsMonitor(AbsJobMonitor &symsMonitor_) = 0;		///< setting the symbols monitor

	virtual ~IMatchEngine() = 0 {}
};

#endif // H_MATCH_ENGINE_BASE
