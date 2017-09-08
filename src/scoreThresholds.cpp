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

#include "scoreThresholds.h"

#pragma warning ( push, 0 )

#include <cassert>

#pragma warning ( pop )

using namespace std;

ScoreThresholds::ScoreThresholds(double multiplier, const ScoreThresholds &references) :
			total(multiplier*references.total), intermediaries(references.intermediaries.size()) {
	const size_t factorsCount = references.intermediaries.size();
	for(size_t i = 0ULL; i < factorsCount; ++i)
		intermediaries[i] = multiplier * references.intermediaries[i];
}

double ScoreThresholds::overall() const { return total; }
size_t ScoreThresholds::thresholdsCount() const { return intermediaries.size(); }

double ScoreThresholds::operator[](size_t idx) const {
	assert(idx < intermediaries.size());
	return intermediaries[idx];
}

bool ScoreThresholds::representsInferiorMatch() const {
	return intermediaries.empty();
}

void ScoreThresholds::inferiorMatch() {
	if(!intermediaries.empty())
		intermediaries.clear();
}

void ScoreThresholds::update(double totalScore) {
	total = totalScore;
}

void ScoreThresholds::update(double totalScore, const std::vector<double> &multipliers) {
	total = totalScore;
	const size_t factorsCount = multipliers.size();
	if(intermediaries.size() != factorsCount)
		intermediaries.resize(factorsCount);
	for(size_t i = 0ULL; i < factorsCount; ++i)
		intermediaries[i] = totalScore * multipliers[i];
}

void ScoreThresholds::update(double multiplier, const IScoreThresholds &references) {
	total = multiplier * references.overall();
	const size_t factorsCount = references.thresholdsCount();
	if(intermediaries.size() != factorsCount)
		intermediaries.resize(factorsCount);
	for(size_t i = 0ULL; i < factorsCount; ++i)
		intermediaries[i] = multiplier * references[i];
}
