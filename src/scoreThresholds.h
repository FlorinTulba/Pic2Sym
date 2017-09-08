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

#ifndef H_SCORE_THRESHOLDS
#define H_SCORE_THRESHOLDS

#include "scoreThresholdsBase.h"

/**
Stores and updates the threshold values for intermediary scores. These values might help sparing
the computation of some matching aspects.

Substitute class of valarray<double>, customized for optimal performance of the use cases from Pic2Sym.
When UseSkipMatchAspectsHeuristic is false, this class behaves almost like a simple `double` value.
*/
class ScoreThresholds : public IScoreThresholds {
protected:
	std::vector<double> intermediaries;	///< the intermediary threshold scores
	double total = 0.;					///< the final threshold score

public:
	ScoreThresholds() = default;

	/**
	Used to set thresholds for clusters, which are the thresholds for the symbols (references)
	multiplied by multiplier.
	*/
	ScoreThresholds(double multiplier, const ScoreThresholds &references);

	ScoreThresholds(const ScoreThresholds&) = delete;
	ScoreThresholds(ScoreThresholds&&) = delete;
	void operator=(const ScoreThresholds&) = delete;
	void operator=(ScoreThresholds&&) = delete;

	double overall() const override final;			///< provides final threshold score (field total)
	size_t thresholdsCount() const override final;	///< @return the number of intermediary scores
	double operator[](size_t idx) const override;	///< provides intermediaries[idx]
	void update(double totalScore) override final;	///< sets total to totalScore

	/// Updates the thresholds for clusters (thresholds for the symbols (references) multiplied by multiplier.)
	void update(double multiplier, const IScoreThresholds &references) override;

	// Methods used only when UseSkipMatchAspectsHeuristic is true

	/// Updates total and intermediaries = totalScore*multipliers
	void update(double totalScore, const std::vector<double> &multipliers) override;

	/// Makes sure that intermediary results won't be used as long as finding only bad matches
	void inferiorMatch() override;

	/// true for empty intermediaries [triggered by inferiorMatch()]
	bool representsInferiorMatch() const override;
};

#endif // H_SCORE_THRESHOLDS
