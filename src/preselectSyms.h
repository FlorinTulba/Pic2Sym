/******************************************************************************
 The application Pic2Sym approximates images by a
 grid of colored symbols with colored backgrounds.

 Copyrights from the libraries used by the program:
 - (c) 2003 Boost (www.boost.org)
     License: doc/licenses/Boost.lic
     http://www.boost.org/LICENSE_1_0.txt
 - (c) 2015-2016 OpenCV (www.opencv.org)
     License: doc/licenses/OpenCV.lic
     http://opencv.org/license/
 - (c) 1996-2002, 2006 The FreeType Project (www.freetype.org)
     License: doc/licenses/FTL.txt
     http://git.savannah.gnu.org/cgit/freetype/freetype2.git/plain/docs/FTL.TXT
 - (c) 1997-2002 OpenMP Architecture Review Board (www.openmp.org)
   (c) Microsoft Corporation (implementation for OpenMP C/C++ v2.0 March 2002)
     See: https://msdn.microsoft.com/en-us/library/8y6825x5.aspx
 - (c) 1995-2017 zlib software (Jean-loup Gailly and Mark Adler - www.zlib.net)
     License: doc/licenses/zlib.lic
     http://www.zlib.net/zlib_license.html


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
 If not, see: http://www.gnu.org/licenses/agpl-3.0.txt .
 *****************************************************************************/

#ifndef H_PRESELECT_SYMS
#define H_PRESELECT_SYMS

#include "misc.h"
#include "preselectSymsBase.h"

#pragma warning(push, 0)

#include <queue>

#pragma warning(pop)

/**
Obtaining the top n candidate matches close-enough to or better than
the previous best known match.
*/
class TopCandidateMatches : public ITopCandidateMatches {
 public:
  /**
  Creating a selection processor.
  @param shortListLength max number of candidates from the final short list
  @param origThreshScore min score to enter the short list
  @throw invalid_argument if shortListLength is 0

  Exception to be only reported, not handled
  */
  TopCandidateMatches(unsigned shortListLength = 1U,
                      double origThreshScore = 0.) noexcept(!UT);

  /// Clears the short list and establishes a new threshold score
  void reset(double origThreshScore) noexcept override;

  /**
  Attempts to put a new candidate on the short list.
  @return false if his score is not good enough.
  @throw logic_error if called after prepareReport()

  Exception to be only reported, not handled
  */
  bool checkCandidate(unsigned candidateIdx,
                      double score) noexcept(!UT) override;

  /// Closes the selection process and orders the short list by score.
  void prepareReport() noexcept override;

  /// Checking if there's at least one candidate on the short list during or
  /// after the selection
  bool foundAny() const noexcept override;

  /**
  Get the sorted short list (without the scores) at the end of the selection
  @throw logic_error if called before prepareReport() or after moveShortList()

  Exception to be only reported, not handled
  */
  const CandidatesShortList& getShortList() const noexcept(!UT) override;

  /**
  Moving to dest the sorted short list (without the scores) at the end of
  the selection
  @throw logic_error if called before prepareReport()

  Exception to be only reported, not handled
  */
  void moveShortList(CandidatesShortList& dest) noexcept(!UT) override;

 private:
  /// Interface for the data for a candidate who enters the short list
  class ICandidate /*abstract*/ {
   public:
    virtual double getScore() const noexcept = 0;
    virtual CandidateId getIdx() const noexcept = 0;

    virtual ~ICandidate() noexcept {}

    // If slicing is observed and becomes a severe problem, use `= delete` for
    // all
    ICandidate(const ICandidate&) noexcept = default;
    ICandidate(ICandidate&&) noexcept = default;
    ICandidate& operator=(const ICandidate&) noexcept = default;
    ICandidate& operator=(ICandidate&&) noexcept = default;

   protected:
    constexpr ICandidate() noexcept {}
  };

  /// Data for a candidate who enters the short list
  class Candidate : public ICandidate {
   public:
    Candidate(CandidateId idx_, double score_) noexcept;

    double getScore() const noexcept final { return score; }
    CandidateId getIdx() const noexcept final { return idx; }

    /// Comparator based on the score
    class Greater {
     public:
      bool operator()(const ICandidate& c1, const ICandidate& c2) const
          noexcept;
    };

   private:
    double score;     ///< his score
    CandidateId idx;  ///< id of the candidate (index in vector&lt;ISymData&gt;)
  };

  /// Unordered version of the short list, but allowing any time to remove the
  /// worst candidate from it
  std::priority_queue<Candidate, std::vector<Candidate>, Candidate::Greater>
      scrapbook;

  CandidatesShortList shortList;  ///< ordered short list (best first)

  /**
  Min score to enter the list.
  As long as the short list isn't full, this threshold is the same as
  origThreshScore. When the list is full, a new candidate enters the list only
  if it beats the score of the last candidate from the list (who will exit the
  list).
  */
  double thresholdScore;

  unsigned n;  ///< length of the short list of candidates

  /// Set to true by prepareReport(); set to false by moveShortList()
  bool shortListReady = false;
};

#endif  // H_PRESELECT_SYMS
