/****************************************************************************************
 The application Pic2Sym approximates images by a
 grid of colored symbols with colored backgrounds.

 This file belongs to the Pic2Sym project.

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
 - (c) 1997-2002 OpenMP Architecture Review Board (www.openmp.org)
   (c) Microsoft Corporation (Visual C++ implementation for OpenMP C/C++ Version 2.0 March 2002)
   See: <https://msdn.microsoft.com/en-us/library/8y6825x5(v=vs.140).aspx>
 
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

#include "patch.h"

#include <boost/optional/optional.hpp>
#include <opencv2/core/core.hpp>

// forward declarations
struct SymData;
struct CachedData;
class MatchSettings;

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
	
	friend std::wostream& operator<<(std::wostream &os, const MatchParams &mp);

#endif // defined _DEBUG || defined UNIT_TESTING

#ifndef UNIT_TESTING // UnitTesting project will still have following methods as public

protected:

#endif // UNIT_TESTING

	/// Both computeFg and computeBg simply call this
	static void computeMean(const cv::Mat &patch, const cv::Mat &mask, boost::optional<double> &miu);

	/// Both computeSdevFg and computeSdevBg simply call this
	static void computeSdev(const cv::Mat &patch, const cv::Mat &mask,
							boost::optional<double> &miu, boost::optional<double> &sdev);
};

/// A possible way to approximate the patch - average / blur / transformed glyph / hybrid
struct ApproxVariant {
	cv::Mat approx;			///< the approximation of the patch
	MatchParams params;		///< parameters of the match (empty for blur-only approximations)

	/// explicit constructor to avoid just passing directly a matrix and forgetting about the 2nd param
	explicit ApproxVariant(const cv::Mat &approx_ = cv::Mat(), ///< a possible approximation of a patch
						   const MatchParams &params_ = MatchParams() ///< the corresponding params of the match
						   ) : approx(approx_), params(params_) {}
};

/// Holds the best match found at a given time
struct BestMatch {
	const Patch patch;			///< the patch to approximate, together with some other details
	
	/// shouldn't assign, as 'patch' member is supposed to remain the same and assign can't guarantee it
	void operator=(const BestMatch&) = delete;

	ApproxVariant bestVariant;	///< best approximating variant 

	/**
	Index within vector<SymData>.
	none if patch approximation is blur-based only.
	Useful during debug, but mainly during unit testing,
	when verifying that some particular symbol can be recognized.
	*/
	boost::optional<unsigned> symIdx;

	/// pointer to vector<SymData>[symIdx] or null when patch approximation is blur-based only.
	const SymData *pSymData = nullptr;

	/// glyph code. none if patch approximation is blur-based only
	boost::optional<unsigned long> symCode;

	/// score of the best match. If patch approximation is blur-based only, score will remain -inf
	double score = std::numeric_limits<double>::lowest();

	/// Resets everything apart the patch
	BestMatch& reset();

	/// Called when finding a better match. Returns itself
	BestMatch& update(double score_, unsigned long symCode_, 
					  unsigned symIdx_, const SymData &sd,
					  const MatchParams &mp);

	/**
	It generates the approximation of the patch based on the rest of the fields.
	
	When pSymData is null, it approximates the patch with patch.blurredPatch.
	
	Otherwise it adapts the foreground and background of the original glyph to
	be as close as possible to the patch.
	For color patches, it does that for every channel.

	For low-contrast images, it generates the average of the patch.

	@param ms additional match settings that might be needed

	@return reference to the updated object
	*/
	BestMatch& updatePatchApprox(const MatchSettings &ms);

#if defined _DEBUG || defined UNIT_TESTING // Next members are necessary for logging

	friend std::wostream& operator<<(std::wostream &os, const BestMatch &bm);

	// Unicode symbols are logged in symbol format, while other encodings log their code
	bool unicode = false;			///< is Unicode the charmap's encoding

	/// Updates unicode field and returns the updated BestMatch object
	BestMatch& setUnicode(bool unicode_);

#endif // defined _DEBUG || defined UNIT_TESTING

	/// Constructor setting only 'patch'. The other fields need the setters or the 'update' method.
	BestMatch(const Patch &patch_) : patch(patch_) {}

	BestMatch(const BestMatch&) = default;
};

#endif