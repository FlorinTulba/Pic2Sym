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

#include "preselectSyms.h"

#pragma warning ( push, 0 )

#include <cassert>

#pragma warning ( pop )

using namespace std;

/// Clearing containers without clear() method, but who do have empty() and pop() methods
template<class Cont>
static void clearCont(Cont &cont) {
	while(!cont.empty())
		cont.pop();
}

TopCandidateMatches::Candidate::Candidate(CandidateId idx_, double score_) :
		idx(idx_), score(score_) {}

bool TopCandidateMatches::Candidate::
		Greater::operator()(const ICandidate &c1,
							const ICandidate &c2) const {
	return c1.getScore() > c2.getScore();
}

TopCandidateMatches::TopCandidateMatches(unsigned shortListLength/* = 1U*/,
										 double origThreshScore/* = 0.*/) :
		thresholdScore(origThreshScore), n(shortListLength) {
	assert(shortListLength > 0U);
}

void TopCandidateMatches::reset(double origThreshScore) {
	shortListReady = false;
	clearCont(scrapbook);
	clearCont(shortList);

	thresholdScore = origThreshScore;
}

bool TopCandidateMatches::checkCandidate(unsigned candidateIdx, double score) {
	assert(!shortListReady); // this method should be called only before prepareReport()

	if(score <= thresholdScore)
		return false;

	if((unsigned)scrapbook.size() == n) // If the short list is full at function start
		scrapbook.pop(); // Worst candidate must be removed

	// The new better candidate enters the short list
	scrapbook.emplace(candidateIdx, score);

	if((unsigned)scrapbook.size() == n) // If the short list is full now
		// New threshold is the score of the new worst candidate
		thresholdScore = scrapbook.top().getScore();

	return true;
}

void TopCandidateMatches::prepareReport() {
	assert(!shortListReady); // this method should be called only once

	while(!scrapbook.empty()) {
		shortList.push(scrapbook.top().getIdx()); // place worse candidates to the bottom of the stack
		scrapbook.pop();
	}

	shortListReady = true;
}

bool TopCandidateMatches::foundAny() const {
	return !shortList.empty() || !scrapbook.empty();
}

CandidatesShortList TopCandidateMatches::getShortList() const {
	assert(shortListReady); // this method should be called only after prepareReport()
	return shortList;
}
