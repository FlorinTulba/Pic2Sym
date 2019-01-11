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

#ifndef H_MATCH_PARAMS
#define H_MATCH_PARAMS

#include "matchParamsBase.h"

/// Holds relevant data during patch&glyph matching
class MatchParams : public IMatchParamsRW {
protected:
	// These params are computed only once, if necessary, when approximating the patch
	boost::optional<double> patchSum;			///< sum of the values of the pixels of the patch
	boost::optional<cv::Mat> patchSq;			///< element-wise product patch * patch
	boost::optional<double> normPatchMinMiu;	///< norm L2 of (patch - miu patch)
	boost::optional<cv::Point2d> mcPatch;		///< mass center for the patch (range 0..1 x 0..1)
	boost::optional<cv::Mat> blurredPatch;		///< blurred version of the patch
	boost::optional<cv::Mat> blurredPatchSq;	///< blurredPatch element-wise squared
	boost::optional<cv::Mat> variancePatch;		///< blur(patch^2) - blurredPatchSq

	// These params are evaluated for each symbol compared to the patch
	boost::optional<cv::Mat> patchApprox;		///< patch approximated by a given symbol
	boost::optional<cv::Point2d> mcPatchApprox;	///< mass center for the approximation of the patch (range 0..1 x 0..1)
	boost::optional<double> mcsOffset;			///< distance between the 2 mass centers (range 0..sqrt(2))
	boost::optional<double> symDensity;			///< % of the box covered by the glyph (0..1)
	boost::optional<double> fg;					///< color for fg (range 0..255)
	boost::optional<double> bg;					///< color for bg (range 0..255)
	boost::optional<double> contrast;			///< fg - bg (range -255..255)
	boost::optional<double> ssim;				///< structural similarity (-1..1)
	boost::optional<double> absCorr;			///< absolute value of correlation (0..1)

	// ideal value for the standard deviations below is 0
	boost::optional<double> sdevFg;		///< standard deviation for fg (0..127.5)
	boost::optional<double> sdevBg;		///< standard deviation for bg  (0..127.5)
	boost::optional<double> sdevEdge;	///< standard deviation for contour (0..255)

public:
	// These params are computed only once, if necessary, when approximating the patch
	const boost::optional<cv::Point2d>& getMcPatch() const override final;		///< mass center for the patch (range 0..1 x 0..1)
#ifdef UNIT_TESTING
	const boost::optional<double>& getPatchSum() const override final;			///< sum of the values of the pixels of the patch
	const boost::optional<cv::Mat>& getPatchSq() const override final;			///< element-wise product patch * patch
	const boost::optional<double>& getNormPatchMinMiu() const override final;	///< norm L2 of (patch - miu patch)
	const boost::optional<cv::Mat>& getBlurredPatch() const override final;		///< blurred version of the patch
	const boost::optional<cv::Mat>& getBlurredPatchSq() const override final;	///< blurredPatch element-wise squared
	const boost::optional<cv::Mat>& getVariancePatch() const override final;	///< blur(patch^2) - blurredPatchSq

	// These params are evaluated for each symbol compared to the patch
	const boost::optional<cv::Mat>& getPatchApprox() const override final;		///< patch approximated by a given symbol
#endif // UNIT_TESTING defined
	const boost::optional<cv::Point2d>& getMcPatchApprox() const override final;///< mass center for the approximation of the patch (range 0..1 x 0..1)
	const boost::optional<double>& getMcsOffset() const override final;			///< distance between the 2 mass centers (range 0..sqrt(2))
	const boost::optional<double>& getSymDensity() const override final;		///< % of the box covered by the glyph (0..1)

#if defined(_DEBUG) || defined(UNIT_TESTING)
	const boost::optional<double>& getFg() const override final;				///< color for fg (range 0..255)

	const std::wstringType toWstring() const override; ///< Provides a representation of the parameters
#endif // defined(_DEBUG) || defined(UNIT_TESTING)

	const boost::optional<double>& getBg() const override final;				///< color for bg (range 0..255)
	const boost::optional<double>& getContrast() const override final;			///< fg - bg (range -255..255)
	const boost::optional<double>& getSsim() const override final;				///< structural similarity (-1..1)
	const boost::optional<double>& getAbsCorr() const override final;			///< absolute value of correlation (0..1)

	// ideal value for the standard deviations below is 0
	const boost::optional<double>& getSdevFg() const override final;			///< standard deviation for fg (0..127.5)
	const boost::optional<double>& getSdevBg() const override final;			///< standard deviation for bg  (0..127.5)
	const boost::optional<double>& getSdevEdge() const override final;			///< standard deviation for contour (0..255)

#ifdef UNIT_TESTING
	std::uniquePtr<IMatchParamsRW> clone() const override;	/// @return a copy of itself
#endif // UNIT_TESTING defined

	/**
	Prepares for next symbol to match against patch.

	When skipPatchInvariantParts = true resets everything except:
	mcPatch, patchSum, normPatchMinMiu, patchSq, blurredPatch, blurredPatchSq and variancePatch.
	*/
	MatchParams& reset(bool skipPatchInvariantParts = true) override;

	// Methods for computing each field
	void computeFg(const cv::Mat &patch, const ISymData &symData) override;
	void computeBg(const cv::Mat &patch, const ISymData &symData) override;
	void computeContrast(const cv::Mat &patch, const ISymData &symData) override;
	void computeSdevFg(const cv::Mat &patch, const ISymData &symData) override;
	void computeSdevBg(const cv::Mat &patch, const ISymData &symData) override;
	void computeSdevEdge(const cv::Mat &patch, const ISymData &symData) override;
	void computeSymDensity(const ISymData &symData) override;
	void computePatchSum(const cv::Mat &patch) override;
	void computePatchSq(const cv::Mat &patch) override;
	void computeMcPatch(const cv::Mat &patch, const CachedData &cachedData) override;
	void computeMcPatchApprox(const cv::Mat &patch, const ISymData &symData,
							  const CachedData &cachedData) override;
	void computeMcsOffset(const cv::Mat &patch, const ISymData &symData,
						  const CachedData &cachedData) override;
	void computePatchApprox(const cv::Mat &patch, const ISymData &symData) override;
	void computeBlurredPatch(const cv::Mat &patch, const CachedData &cachedData) override;
	void computeBlurredPatchSq(const cv::Mat &patch, const CachedData &cachedData) override;
	void computeVariancePatch(const cv::Mat &patch, const CachedData &cachedData) override;
	void computeSsim(const cv::Mat &patch, const ISymData &symData,
					 const CachedData &cachedData) override;
	void computeNormPatchMinMiu(const cv::Mat &patch, const CachedData &cachedData) override;
	void computeAbsCorr(const cv::Mat &patch, const ISymData &symData,
						const CachedData &cachedData) override;

	/**
	Returns an instance as for an ideal match between a symbol and a patch.
	Also avoids:
	- either cluttering the interface IMatchParamsRW with setters for creating the
		MatchParams of the perfect match.
	- or introducing a special constructor / Builder just for the perfect match
	*/
	static const MatchParams& perfectMatch();

#ifndef UNIT_TESTING // UnitTesting project will still have following methods as public
protected:
#endif // UNIT_TESTING not defined

	/// Both computeFg and computeBg simply call this
	static void computeMean(const cv::Mat &patch, const cv::Mat &mask, boost::optional<double> &miu);

	/// Both computeSdevFg and computeSdevBg simply call this
	static void computeSdev(const cv::Mat &patch, const cv::Mat &mask,
							boost::optional<double> &miu, boost::optional<double> &sdev);
};

#endif // H_MATCH_PARAMS
