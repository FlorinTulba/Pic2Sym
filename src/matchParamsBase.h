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

#ifndef H_MATCH_PARAMS_BASE
#define H_MATCH_PARAMS_BASE

#pragma warning ( push, 0 )

#if defined _DEBUG || defined UNIT_TESTING
#include "std_string.h"
#endif // _DEBUG || UNIT_TESTING

#include "std_memory.h"

#include <boost/optional/optional.hpp>

#include <opencv2/core/core.hpp>

#pragma warning ( pop )

// Forward declarations
struct ISymData;
class CachedData;
struct IMatchParamsRW;

/**
Base class (Read-only version) for the relevant parameters during patch&glyph matching.

AI Reviewer detected segregation & split personality in this interface.

Indeed, the fields & methods concerning mass centers (getMc*) are used only in
GravitationalSmoothness and DirectionalSmoothness, apart from the Unit tests.
However, those references of IMatchParams appear within the parameter list of
several methods overriding corresponding methods from MatchAspect.
So, after introducing a new interface for getMc* methods derived from IMatchParams,
the interface would be returned by a dynamic_cast<new_interface&>(param_of_type_IMatchParams).

This makes the interface separation less appealing.
*/
struct IMatchParams /*abstract*/ {
	// These params are computed only once, if necessary, when approximating the patch
	virtual const boost::optional<cv::Point2d>& getMcPatch() const = 0;		///< mass center for the patch (range 0..1 x 0..1)
#ifdef UNIT_TESTING
	virtual const boost::optional<cv::Mat>& getBlurredPatch() const = 0;	///< blurred version of the patch
	virtual const boost::optional<cv::Mat>& getBlurredPatchSq() const = 0;	///< blurredPatch element-wise squared
	virtual const boost::optional<cv::Mat>& getVariancePatch() const = 0;	///< blur(patch^2) - blurredPatchSq

	// These params are evaluated for each symbol compared to the patch
	virtual const boost::optional<cv::Mat>& getPatchApprox() const = 0;			///< patch approximated by a given symbol
#endif // UNIT_TESTING defined
	virtual const boost::optional<cv::Point2d>& getMcPatchApprox() const = 0;	///< mass center for the approximation of the patch (range 0..1 x 0..1)
	virtual const boost::optional<double>& getMcsOffset() const = 0;			///< distance between the 2 mass centers (range 0..sqrt(2))
	virtual const boost::optional<double>& getSymDensity() const = 0;			///< % of the box covered by the glyph (0..1)
#if defined(_DEBUG) || defined(UNIT_TESTING)
	virtual const boost::optional<double>& getFg() const = 0;					///< color for fg (range 0..255)
#endif // defined(_DEBUG) || defined(UNIT_TESTING)
	virtual const boost::optional<double>& getBg() const = 0;					///< color for bg (range 0..255)
	virtual const boost::optional<double>& getContrast() const = 0;				///< fg - bg (range -255..255)
	virtual const boost::optional<double>& getSsim() const = 0;					///< structural similarity (-1..1)

	// ideal value for the standard deviations below is 0
	virtual const boost::optional<double>& getSdevFg() const = 0;	///< standard deviation for fg (0..127.5)
	virtual const boost::optional<double>& getSdevBg() const = 0;	///< standard deviation for bg  (0..127.5)
	virtual const boost::optional<double>& getSdevEdge() const = 0;	///< standard deviation for contour (0..255)

	virtual ~IMatchParams() = 0 {}

#ifdef UNIT_TESTING
	virtual std::uniquePtr<IMatchParamsRW> clone() const = 0;	/// @return a configurable copy of itself
#endif // UNIT_TESTING defined

#if defined _DEBUG || defined UNIT_TESTING
	/// Provides a representation of the parameters
	virtual const std::wstringType toWstring() const = 0;
#endif // _DEBUG || UNIT_TESTING
};

#if defined _DEBUG || defined UNIT_TESTING
std::wostream& operator<<(std::wostream &wos, const IMatchParams &mp);
#endif // _DEBUG || UNIT_TESTING

/// Base class (Read-Write version) for the relevant parameters during patch&glyph matching
struct IMatchParamsRW /*abstract*/ : IMatchParams {
	/**
	Prepares for next symbol to match against patch.

	When skipPatchInvariantParts = true resets everything except:
	mcPatch, blurredPatch, blurredPatchSq and variancePatch.
	*/
	virtual IMatchParamsRW& reset(bool skipPatchInvariantParts = true) = 0;

	// Methods for computing each field
	virtual void computeFg(const cv::Mat &patch, const ISymData&) = 0;
	virtual void computeBg(const cv::Mat &patch, const ISymData&) = 0;
	virtual void computeContrast(const cv::Mat &patch, const ISymData&) = 0;
	virtual void computeSdevFg(const cv::Mat &patch, const ISymData&) = 0;
	virtual void computeSdevBg(const cv::Mat &patch, const ISymData&) = 0;
	virtual void computeSdevEdge(const cv::Mat &patch, const ISymData&) = 0;
	virtual void computeSymDensity(const ISymData&) = 0;
	virtual void computeMcPatch(const cv::Mat &patch, const CachedData&) = 0;
	virtual void computeMcPatchApprox(const cv::Mat &patch, const ISymData&,
									  const CachedData&) = 0;
	virtual void computeMcsOffset(const cv::Mat &patch, const ISymData&,
								  const CachedData&) = 0;
	virtual void computePatchApprox(const cv::Mat &patch, const ISymData&) = 0;
	virtual void computeBlurredPatch(const cv::Mat &patch, const CachedData&) = 0;
	virtual void computeBlurredPatchSq(const cv::Mat &patch, const CachedData&) = 0;
	virtual void computeVariancePatch(const cv::Mat &patch, const CachedData&) = 0;
	virtual void computeSsim(const cv::Mat &patch, const ISymData&,
							 const CachedData&) = 0;

	virtual ~IMatchParamsRW() = 0 {}
};

#endif //H_MATCH_PARAMS_BASE
