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

#ifndef H_SYM_DATA
#define H_SYM_DATA

#include "matSerialization.h"
#include "floatType.h"

#pragma warning ( push, 0 )

#include <array>
#include <boost/serialization/array.hpp>

#ifdef UNIT_TESTING
#	include <map>
#endif // UNIT_TESTING defined

#include <boost/serialization/version.hpp>
#include <opencv2/core/core.hpp>

#pragma warning ( pop )

/// Most symbol information
struct SymData {
	// BUILD CLEAN WHEN THIS CHANGES!
	static const unsigned VERSION = 1U; ///< version of SymData class

	/**
	Computes most information about a symbol based on glyph parameter.
	It's also used to spare the constructor of SymData from performing computeFields' job.
	That can be useful when multiple threads add SymData items to a vector within a critical section,
	so each new SymData's fields are prepared outside the critical section to minimize blocking of
	other threads.
	*/
	static void computeFields(const cv::Mat &glyph, cv::Mat &fgMask, cv::Mat &bgMask,
							  cv::Mat &edgeMask, cv::Mat &groundedGlyph, cv::Mat &blurOfGroundedGlyph,
							  cv::Mat &varianceOfGroundedGlyph, fp &minVal, fp &diffMinMax,
							  bool forTinySym);

	/// mass center of the symbol given original fg & bg (coordinates are within a unit-square: 0..1 x 0..1)
	cv::Point2f mc = cv::Point2f(.5f, .5f);

	cv::Mat negSym;	///< negative of the symbol (0..255 byte for normal symbols; fp for tiny)

	enum { // indices of each matrix type within a MatArray object
		FG_MASK_IDX,			///< mask isolating the foreground of the glyph
		BG_MASK_IDX,			///< mask isolating the background of the glyph 
		EDGE_MASK_IDX,			///< mask isolating the edge of the glyph (transition region fg-bg)
		GROUNDED_SYM_IDX,		///< symbol shifted (in brightness) to have black background (0..1)
		BLURRED_GR_SYM_IDX,		///< blurred version of the grounded symbol (0..1)
		VARIANCE_GR_SYM_IDX,	///< variance of the grounded symbol

		MATRICES_COUNT ///< KEEP THIS LAST and DON'T USE IT AS INDEX in MatArray objects!
	};

	// For each symbol from cmap, there'll be several additional helpful matrices to store
	// along with the one for the given glyph. The enum from above should be used for selection.
	typedef std::array< cv::Mat, MATRICES_COUNT > MatArray;

	MatArray masks;				///< various masks

	size_t symIdx = 0U;			///< symbol index within cmap
	fp minVal = 0.f;			///< the value of darkest pixel, range 0..1
	fp diffMinMax = 1.f;		///< difference between brightest and darkest pixels, each in 0..1
	fp avgPixVal = 0.f;			///< average pixel value, each pixel in 0..1

	unsigned long code = ULONG_MAX;	///< the code of the symbol

	/**
	Enabled symbol filters might mark this symbol as removable,
	but PreserveRemovableSymbolsForExamination from configuration might allow it to remain
	in the active symbol set used during image transformation.

	However, when removable == true && PreserveRemovableSymbolsForExamination,
	the symbol will appear as marked (inversed) in the cmap viewer

	This field doesn't need to be serialized, as filtering options might be different
	for distinct run sessions.
	*/
	bool removable = false;

	/// Fast constructor with the fields precomputed by computeFields - suitable for critical sections
	SymData(const cv::Mat &negSym_, unsigned long code_, size_t symIdx_, fp minVal_, fp diffMinMax_,
			fp avgPixVal_, const cv::Point2f &mc_, const MatArray &masks_, bool removable_ = false);

	SymData(const SymData &other);
	SymData(SymData &&other); ///< moves the matrices from other (instead of just copying them)

	SymData& operator=(const SymData &other);
	SymData& operator=(SymData &&other); ///< moves the matrices from other (instead of just copying them)

	/**
	Serializes this SymData object (apart from 'removable' field) to ar.

	'removable' field doesn't need to be serialized,
	as filtering options might be different for distinct run sessions.
	*/
	template<class Archive>
	void serialize(Archive &ar, const unsigned int version) {
		const bool isSaving = Archive::is_saving::value;

		ar & code & symIdx;

		// SymData version 0 was using double-precision for several fields; Newer versions use single-precision.
		if(version >= 1U || isSaving) {
			ar & minVal & diffMinMax & avgPixVal;
			ar & mc.x & mc.y;
			ar & negSym;
			ar & masks;

		} else { // loading in version 0
			// SymData version 0 was using double-precision for several fields; Newer versions use single-precision.
			double minValD, diffMinMaxD, avgPixValD;
			ar & minValD & diffMinMaxD & avgPixValD;
			minVal = (fp)minValD; diffMinMax = (fp)diffMinMaxD; avgPixVal = (fp)avgPixValD;

			double mcXD, mcYD;
			ar & mcXD & mcYD;
			mc.x = (fp)mcXD; mc.y = (fp)mcYD;

			ar & negSym;
			if(negSym.type() == CV_64FC1) // this happens for tiny symbols from version 0
				negSym.convertTo(negSym, CV_FC1);

			ar & masks;
			masks[GROUNDED_SYM_IDX].convertTo(masks[GROUNDED_SYM_IDX], CV_FC1);
			masks[BLURRED_GR_SYM_IDX].convertTo(masks[BLURRED_GR_SYM_IDX], CV_FC1);
			masks[VARIANCE_GR_SYM_IDX].convertTo(masks[VARIANCE_GR_SYM_IDX], CV_FC1);
		}

		// Make sure a loaded symbol is initially not removable
		if(!isSaving && removable)
			removable = false;
	}

#ifdef UNIT_TESTING
	typedef std::map< int, const cv::Mat > IdxMatMap; ///< Used in the SymData constructor from below

	/// Constructor that allows filling only the relevant matrices from MatArray
	SymData(unsigned long code_, size_t symIdx_, fp minVal_, fp diffMinMax_, fp avgPixVal_,
			const cv::Point2f &mc_, const IdxMatMap &relevantMats, const cv::Mat &negSym_ = cv::Mat());

	/// A clone with different symIdx
	SymData clone(size_t symIdx_);
#endif // UNIT_TESTING defined

protected:
	/* Constructors callable from derived classes only */

	/// Used for creation of TinySym and ClusterData
	SymData(unsigned long code_ = ULONG_MAX, size_t symIdx_ = 0U,
			fp avgPixVal_ = 0.f, const cv::Point2f &mc_ = cv::Point2f(.5f, .5f));

	/// Used to create the TinySym centroid of a cluster 
	SymData(const cv::Point2f &mc_, fp avgPixVal_);
};

BOOST_CLASS_VERSION(SymData, SymData::VERSION);

/// VSymData - vector with most information about each symbol
typedef std::vector<const SymData> VSymData;

#endif // H_SYM_DATA