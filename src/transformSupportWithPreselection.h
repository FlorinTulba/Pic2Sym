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

#ifndef H_TRANSFORM_SUPPORT_WITH_PRESELECTION
#define H_TRANSFORM_SUPPORT_WITH_PRESELECTION

#include "transformSupport.h"

#pragma warning ( push, 0 )

#include <vector>

#pragma warning ( pop )

// Forward declarations
struct BestMatch;
class MatchSupport;
class MatchSupportWithPreselection;

/**
Initializes and updates draft matches.
It perform those tasks also for tiny symbols.
*/
class TransformSupportWithPreselection : public TransformSupport {
protected:
	cv::Mat resizedForTinySyms;	///< resized version of the original used by tiny symbols preselection
	cv::Mat resBlForTinySyms;	///< blurred version of the resized used by tiny symbols preselection
	std::vector<std::vector<BestMatch>> draftMatchesForTinySyms; ///< temporary best matches used by tiny symbols preselection
	MatchSupportWithPreselection &matchSupport;	///< match support

public:
	/// Requires an additional MatchSupport parameter compared to the base constructor
	TransformSupportWithPreselection(MatchEngine &me_, const MatchSettings &matchSettings_,
									 cv::Mat &resized_, cv::Mat &resizedBlurred_,
									 std::vector<std::vector<BestMatch>> &draftMatches_,
									 MatchSupport &matchSupport_);

	TransformSupportWithPreselection(const TransformSupportWithPreselection&) = delete;
	TransformSupportWithPreselection(TransformSupportWithPreselection&&) = delete;
	void operator=(const TransformSupportWithPreselection&) = delete;
	void operator=(TransformSupportWithPreselection&&) = delete;

	/// Initializes the drafts when a new image needs to be approximated
	void initDrafts(bool isColor, unsigned patchSz, unsigned patchesPerCol, unsigned patchesPerRow) override;

	/// Resets the drafts when current image needs to be approximated in a different context
	void resetDrafts(unsigned patchesPerCol) override;

	/**
	Approximates row r of patches of size patchSz from an image with given width.
	It checks only the symbols with indices in range [fromSymIdx, upperSymIdx).
	*/
	void approxRow(int r, int width, unsigned patchSz,
				   unsigned fromSymIdx, unsigned upperSymIdx, cv::Mat &result) override;
};

#endif // H_TRANSFORM_SUPPORT_WITH_PRESELECTION