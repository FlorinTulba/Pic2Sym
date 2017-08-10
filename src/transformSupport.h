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

#ifndef H_TRANSFORM_SUPPORT
#define H_TRANSFORM_SUPPORT

#pragma warning ( push, 0 )

#include <vector>
#include <memory>

#include <opencv2/core/core.hpp>

#pragma warning ( pop )

// Forward declarations
struct IBestMatch;
class MatchSettings;
class MatchEngine;

/**
Initializes and updates draft matches.
When PreselectionByTinySyms == true, it perform those tasks also for tiny symbols.
*/
class TransformSupport {
protected:
	/// Initializes a row of a draft when a new image needs to be approximated
	static void initDraftRow(std::vector<std::vector<std::unique_ptr<IBestMatch>>> &draft,
							 int r, unsigned patchesPerRow,
							 const cv::Mat &res, const cv::Mat &resBlurred, int patchSz, bool isColor);

	/// Resets a row of a draft when current image needs to be approximated in a different context
	static void resetDraftRow(std::vector<std::vector<std::unique_ptr<IBestMatch>>> &draft, int r);

	/// Update the visualized draft
	static void patchImproved(cv::Mat &result, unsigned sz, const IBestMatch &draftMatch, 
							  const cv::Range &rowRange, int startCol);

	/// Update PatchApprox for uniform Patch only during the compare with 1st sym (from 1st batch)
	static void manageUnifPatch(const MatchSettings &ms, cv::Mat &result, unsigned sz, 
								IBestMatch &draftMatch, const cv::Range &rowRange, int startCol);

	/// Determines if a given patch is worth approximating (Uniform patches don't make sense approximating)
	static bool checkUnifPatch(IBestMatch &draftMatch);

	MatchEngine &me;					///< match engine
	const MatchSettings &matchSettings;	///< match settings
	cv::Mat &resized;					///< resized version of the original
	cv::Mat &resizedBlurred;			///< blurred version of the resized original
	std::vector<std::vector<std::unique_ptr<IBestMatch>>> &draftMatches;	///< temporary best matches

public:
	/// Base constructor
	TransformSupport(MatchEngine &me_, const MatchSettings &matchSettings_,
					 cv::Mat &resized_, cv::Mat &resizedBlurred_,
					 std::vector<std::vector<std::unique_ptr<IBestMatch>>> &draftMatches_);

	TransformSupport(const TransformSupport&) = delete;
	TransformSupport(TransformSupport&&) = delete;
	void operator=(const TransformSupport&) = delete;
	void operator=(TransformSupport&&) = delete;

	virtual ~TransformSupport() {}

	/// Initializes the drafts when a new image needs to be approximated
	virtual void initDrafts(bool isColor, unsigned patchSz, unsigned patchesPerCol, unsigned patchesPerRow);

	/// Resets the drafts when current image needs to be approximated in a different context
	virtual void resetDrafts(unsigned patchesPerCol);

	/**
	Approximates row r of patches of size patchSz from an image with given width.
	It checks only the symbols with indices in range [fromSymIdx, upperSymIdx).
	*/
	virtual void approxRow(int r, int width, unsigned patchSz,
						   unsigned fromSymIdx, unsigned upperSymIdx, cv::Mat &result);
};

#endif // H_TRANSFORM_SUPPORT