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

#ifndef H_BEST_MATCH_BASE
#define H_BEST_MATCH_BASE

#pragma warning ( push, 0 )

#if defined _DEBUG || defined UNIT_TESTING

#include "std_string.h"

#include <iostream>

#endif // defined _DEBUG || defined UNIT_TESTING

#include "std_memory.h"

#include <boost/optional/optional.hpp>

#include <opencv2/core/core.hpp>

#pragma warning ( pop )

// Forward declarations
struct IMatchParams;
struct IMatchParamsRW;
struct IPatch;
struct ISymData;
struct IMatchSettings;

/// Base class to hold the best match found at a given time
struct IBestMatch /*abstract*/ {
	/// The patch to approximate, together with some other details. No setter available
	virtual const IPatch& getPatch() const = 0;

	/// The approximation of the patch
	virtual const cv::Mat& getApprox() const = 0;

	/// Parameters of the match (none for blur-only approximations)
	virtual const boost::optional<const IMatchParams&> getParams() const = 0;

	/// Parameters of the match
	virtual const std::uniquePtr<IMatchParamsRW>& refParams() const = 0;

	/// Index within vector&lt;DerivedFrom_ISymData&gt;. none if patch approximation is blur-based only.
	virtual const boost::optional<unsigned>& getSymIdx() const = 0;

	/// Index of last cluster that was worth investigating thoroughly
	virtual const boost::optional<unsigned>& getLastPromisingNontrivialCluster() const = 0;
	/// Set the index of last cluster that was worth investigating thoroughly; @return itself
	virtual IBestMatch& setLastPromisingNontrivialCluster(unsigned clustIdx) = 0;

	/// Glyph code. none if patch approximation is blur-based only
	virtual const boost::optional<unsigned long>& getSymCode() const = 0;

	/// Score of the best match. If patch approximation is blur-based only, score will remain 0.
	virtual double getScore() const = 0;
	/// Set the score of the new best match; @return itself
	virtual IBestMatch& setScore(double score_) = 0;

	/// Resets everything apart the patch and the patch-invariant parameters
	virtual IBestMatch& reset() = 0;

	/// Called when finding a better match; @return itself
	virtual IBestMatch& update(double score_, unsigned long symCode_,
							   unsigned symIdx_, const ISymData &sd) = 0;

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
	virtual IBestMatch& updatePatchApprox(const IMatchSettings &ms) = 0;

#if defined _DEBUG || defined UNIT_TESTING // Next members are necessary for logging
	/**
	Unicode symbols will be logged in symbol format, while other encodings will log their code
	@return is Unicode the charmap's encoding
	*/
	virtual bool isUnicode() const = 0;

	/// Updates unicode field; @return itself
	virtual IBestMatch& setUnicode(bool unicode_) = 0;

	/// Provides a representation of the match
	virtual const std::wstringType toWstring() const = 0;
#endif // defined _DEBUG || defined UNIT_TESTING

	/// shouldn't assign, as 'patch' member is supposed to remain the same and assign can't guarantee it
	void operator=(const IBestMatch&) = delete;

	virtual ~IBestMatch() = 0 {}
};

#if defined _DEBUG || defined UNIT_TESTING
std::wostream& operator<<(std::wostream &wos, const IBestMatch &bm);
#endif // defined _DEBUG || defined UNIT_TESTING

#endif // H_BEST_MATCH_BASE
