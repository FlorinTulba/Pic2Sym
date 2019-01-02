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

#ifndef H_SCORE_THRESHOLDS_BASE
#define H_SCORE_THRESHOLDS_BASE

#pragma warning ( push, 0 )

#include <vector>

#pragma warning ( pop )

/**
Interface of ScoreThresholds.

Stores and updates the threshold values for intermediary scores. These values might help sparing
the computation of some matching aspects.

When UseSkipMatchAspectsHeuristic is false, this class behaves almost like a simple `double` value.
*/
struct IScoreThresholds /*abstract*/ {
	virtual double overall() const = 0;					///< provides final threshold score

	virtual size_t thresholdsCount() const = 0;			///< @return the number of intermediary scores
	virtual double operator[](size_t idx) const = 0;	///< provides the idx-th intermediary score

	virtual void update(double totalScore) = 0;			///< sets final score to totalScore

	/// Updates the thresholds for clusters (thresholds for the symbols (references) multiplied by multiplier.)
	virtual void update(double multiplier, const IScoreThresholds &references) = 0;


	// Methods used only when UseSkipMatchAspectsHeuristic is true

	/// Updates final and intermediary scores as totalScore * multipliers
	virtual void update(double totalScore, const std::vector<double> &multipliers) = 0;

	/// Makes sure that intermediary results won't be used as long as finding only bad matches
	virtual void inferiorMatch() = 0;

	/// true for empty intermediaries [triggered by inferiorMatch()]
	virtual bool representsInferiorMatch() const = 0;

	virtual ~IScoreThresholds() = 0 {}
};

#endif // H_SCORE_THRESHOLDS_BASE
