/****************************************************************************************
 The application Pic2Sym approximates images by a
 grid of colored symbols with colored backgrounds.

 This file was created on 2016-4-8
 and belongs to the Pic2Sym project.

 Copyrights from the libraries used by the program:
 - (c) 2015 Boost (www.boost.org)
   License: <http://www.boost.org/LICENSE_1_0.txt>
            or doc/licenses/Boost.lic
 - (c) 2015 The FreeType Project (www.freetype.org)
   License: <http://git.savannah.gnu.org/cgit/freetype/freetype2.git/plain/docs/FTL.TXT>
	        or doc/licenses/FTL.txt
 - (c) 2015 OpenCV (www.opencv.org)
   License: <http://opencv.org/license.html>
            or doc/licenses/OpenCV.lic
 
 (c) 2016 Florin Tulba <florintulba@yahoo.com>

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
 ****************************************************************************************/

#ifndef H_MATCH_PARAMS
#define H_MATCH_PARAMS

#include <boost/optional/optional.hpp>
#include <opencv2/core/core.hpp>

// forward declarations
struct SymData;
struct CachedData;

/// Holds relevant data during patch&glyph matching
struct MatchParams {
	// These params are computed only once, if necessary, when approximating the patch
	boost::optional<cv::Point2d> mcPatch;		///< mass center for the patch
	boost::optional<cv::Mat> blurredPatch;		///< blurred version of the patch
	boost::optional<cv::Mat> blurredPatchSq;	///< blurredPatch element-wise squared
	boost::optional<cv::Mat> variancePatch;		///< blur(patch^2) - blurredPatchSq

	// These params are evaluated for each symbol compared to the patch
	boost::optional<cv::Mat> patchApprox;		///< patch approximated by a given symbol
	boost::optional<cv::Point2d> mcPatchApprox;	///< mass center for the approximation of the patch
	boost::optional<double> mcsOffset;			///< distance between the 2 mass centers
	boost::optional<double> symDensity;			///< % of the box covered by the glyph (0..1)
	boost::optional<double> fg;					///< color for fg (range 0..255)
	boost::optional<double> bg;					///< color for bg (range 0..255)
	boost::optional<double> contrast;			///< fg - bg (range -255..255)
	boost::optional<double> ssim;				///< structural similarity (-1..1)

	// ideal value for the standard deviations below is 0
	boost::optional<double> sdevFg;		///< standard deviation for fg (0..127.5)
	boost::optional<double> sdevBg;		///< standard deviation for bg  (0..127.5)
	boost::optional<double> sdevEdge;	///< standard deviation for contour (0..255)

	/**
	Prepares for next symbol to match against patch.

	When skipPatchInvariantParts = true resets everything except:
	mcPatch, blurredPatch, blurredPatchSq and variancePatch.
	*/
	void reset(bool skipPatchInvariantParts = true);

	// Methods for computing each field
	void computeFg(const cv::Mat &patch, const SymData &symData);
	void computeBg(const cv::Mat &patch, const SymData &symData);
	void computeContrast(const cv::Mat &patch, const SymData &symData);
	void computeSdevFg(const cv::Mat &patch, const SymData &symData);
	void computeSdevBg(const cv::Mat &patch, const SymData &symData);
	void computeSdevEdge(const cv::Mat &patch, const SymData &symData);
	void computeSymDensity(const SymData &symData, const CachedData &cachedData);
	void computeMcPatch(const cv::Mat &patch, const CachedData &cachedData);
	void computeMcPatchApprox(const cv::Mat &patch, const SymData &symData,
							  const CachedData &cachedData);
	void computeMcsOffset(const cv::Mat &patch, const SymData &symData, const CachedData &cachedData);
	void computePatchApprox(const cv::Mat &patch, const SymData &symData);
	void computeBlurredPatch(const cv::Mat &patch);
	void computeBlurredPatchSq(const cv::Mat &patch);
	void computeVariancePatch(const cv::Mat &patch);
	void computeSsim(const cv::Mat &patch, const SymData &symData);

#if defined _DEBUG || defined UNIT_TESTING // Next members are necessary for logging
	static const std::wstring HEADER; ///< table header when values are serialized
	friend std::wostream& operator<<(std::wostream &os, const MatchParams &mp);
#endif

#ifndef UNIT_TESTING // UnitTesting project will still have following methods as public
protected:
#endif
	/// Both computeFg and computeBg simply call this
	static void computeMean(const cv::Mat &patch, const cv::Mat &mask, boost::optional<double> &miu);

	/// Both computeSdevFg and computeSdevBg simply call this
	static void computeSdev(const cv::Mat &patch, const cv::Mat &mask,
							boost::optional<double> &miu, boost::optional<double> &sdev);
};

/// Holds the best grayscale match found at a given time
struct BestMatch {
	unsigned symIdx = UINT_MAX;			///< index within vector<PixMapSym>
	unsigned long symCode = ULONG_MAX;	///< glyph code

	double score = std::numeric_limits<double>::lowest(); ///< score of the best match

	MatchParams params;		///< parameters of the match for the best approximating glyph

	/// called when finding a better match
	void update(double score_, unsigned symIdx_, unsigned long symCode_,
				const MatchParams &params_);

#if defined _DEBUG || defined UNIT_TESTING // Next members are necessary for logging
	static const std::wstring HEADER;
	friend std::wostream& operator<<(std::wostream &os, const BestMatch &bm);

	// Unicode symbols are logged in symbol format, while other encodings log their code
	const bool unicode;					///< is the charmap in Unicode?

	BestMatch(bool isUnicode = true);
	BestMatch(const BestMatch&) = default;
	BestMatch& operator=(const BestMatch &other);
#endif
};

#endif