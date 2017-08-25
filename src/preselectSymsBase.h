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

#ifndef H_PRESELECT_SYMS_BASE
#define H_PRESELECT_SYMS_BASE

#pragma warning ( push, 0 )

#include <vector>
#include <stack>

#pragma warning ( pop )

/// Id of the 'candidate' symbol (index in vector&lt;ISymData&gt;)
typedef unsigned CandidateId;

///< Selected 'candidate' symbols to compete within final selection, ordered by their estimated potential
typedef std::stack<CandidateId, std::vector<CandidateId>> CandidatesShortList;

/// Interface of TopCandidateMatches
struct ITopCandidateMatches /*abstract*/ {
	/// Clears the short list and establishes a new threshold score
	virtual void reset(double origThreshScore) = 0;

	/// Attempts to put a new candidate on the short list. Returns false if his score is not good enough.
	virtual bool checkCandidate(unsigned candidateIdx, double score) = 0;

	/// Closes the selection process and orders the short list by score.
	virtual void prepareReport() = 0;

	/// Checking if there's at least one candidate on the short list during or after the selection
	virtual bool foundAny() const = 0;

	/// Providing a copy of the sorted short list (without the scores) at the end of the selection
	virtual CandidatesShortList getShortList() const = 0;

	virtual ~ITopCandidateMatches() = 0 {}
};

#endif // H_PRESELECT_SYMS_BASE
