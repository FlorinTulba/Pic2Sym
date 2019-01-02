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

#ifndef H_PRESELECT_SYMS
#define H_PRESELECT_SYMS

#include "preselectSymsBase.h"

#pragma warning ( push, 0 )

#include <queue>

#pragma warning ( pop )

/**
Obtaining the top n candidate matches close-enough to or better than
the previous best known match.
*/
class TopCandidateMatches : public ITopCandidateMatches {
protected:
	/// Interface for the data for a candidate who enters the short list
	struct ICandidate /*abstract*/ {
		virtual double getScore() const = 0;
		virtual CandidateId getIdx() const = 0;

		virtual ~ICandidate() = 0 {}
	};
	
	/// Data for a candidate who enters the short list
	class Candidate : public ICandidate {
	protected:
		double score;		///< his score
		CandidateId idx;	///< id of the candidate (index in vector&lt;ISymData&gt;)

	public:
		Candidate(CandidateId idx_, double score_);

		double getScore() const override final { return score; }
		CandidateId getIdx() const override final { return idx; }

		/// Comparator based on the score
		struct Greater {
			bool operator()(const ICandidate &c1, const ICandidate &c2) const;
		};
	};

	/// Unordered version of the short list, but allowing any time to remove the worst candidate from it
	std::priority_queue<Candidate, std::vector<Candidate>, Candidate::Greater> scrapbook;

	CandidatesShortList shortList;	///< ordered short list (best first)

	/**
	Min score to enter the list.
	As long as the short list isn't full, this threshold is the same as origThreshScore.
	When the list is full, a new candidate enters the list only if it beats the score of
	the last candidate from the list (who will exit the list).
	*/
	double thresholdScore;

	unsigned n;	///< length of the short list of candidates

	bool shortListReady = false;	///< set to true by prepareReport()

public:
	/**
	Creating a selection processor.
	@param shortListLength max number of candidates from the final short list
	@param origThreshScore min score to enter the short list
	*/
	TopCandidateMatches(unsigned shortListLength = 1U,
						double origThreshScore = 0.);

	void reset(double origThreshScore) override; ///< clears the short list and establishes a new threshold score

	/// Attempts to put a new candidate on the short list. Returns false if his score is not good enough.
	bool checkCandidate(unsigned candidateIdx, double score) override;

	/// Closes the selection process and orders the short list by score.
	void prepareReport() override;

	/// Checking if there's at least one candidate on the short list during or after the selection
	bool foundAny() const override;

	/// Providing a copy of the sorted short list (without the scores) at the end of the selection
	CandidatesShortList getShortList() const override;
};

#endif // H_PRESELECT_SYMS
