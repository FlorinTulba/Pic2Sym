/************************************************************************************************
 The application Pic2Sym approximates images by a
 grid of colored symbols with colored backgrounds.

 This file belongs to the Pic2Sym project.

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
 ***********************************************************************************************/

#ifndef H_TINY_SYM
#define H_TINY_SYM

#include "symData.h"

#include <opencv2/core/core.hpp>

struct PixMapSym; // Forward declaration

/// Data for tiny symbols
struct TinySym : SymData {
	// BUILD CLEAN WHEN THIS CHANGES!
	static const unsigned VERSION = 0U; ///< version of TinySym class

	/// Ratio between reference symbols and the shrunken symbol
	enum { RatioRefTiny = 8 };

	static void computeFields(const PixMapSym &refSym,
							  double &minVal, double &maxVal,
							  MatArray &symAndMasks,
							  cv::Mat &hAvgProj, cv::Mat &vAvgProj,
							  cv::Mat &backslashDiagAvgProj, cv::Mat &slashDiagAvgProj);

	/*
	Next 5 matrices from below are for the grounded version, not the original.
	Each would normally contain elements in range 0..1, but all of them were
	divided by the number of elements of the corresponding matrix.
	The reason behind this division is that the matrices are norm() compared and when the matrices
	are normalized (by the above mentioned division), the norm() result ranges for all of them
	in 0..1.
	So, despite they have, respectively: n^2, n, n, 2*n-1 and 2*n-1 elements,
	comparing 2 of the same kind with norm() produces values within a UNIQUE range 0..1.
	Knowing that, we can define a SINGLE threshold for all 5 matrices that establishes
	when 2 matrices of the same kind are similar.
	
	The alternative was to define/derive a threshold for each individual category (n^2, n, 2*n-1),
	but this requires adapting these new thresholds to every n - configurable size of tiny symbols.

	So, the normalization allows setting a single threshold for comparing tiny symbols of any configured size:
	- MaxAvgProjErrForPartitionClustering for partition clustering
	- TTSAS_Threshold_Member for TTSAS clustering
	*/
	cv::Mat mat;					///< grounded version of the small symbol divided by TinySymArea
	cv::Mat hAvgProj;				///< horizontal projection divided by TinySymSz
	cv::Mat vAvgProj;				///< vertical projection divided by TinySymSz
	cv::Mat backslashDiagAvgProj;	///< normal diagonal projection divided by TinySymDiagsCount
	cv::Mat slashDiagAvgProj;		///< inverse diagonal projection divided by TinySymDiagsCount

	TinySym(unsigned long code_ = ULONG_MAX, size_t symIdx_ = 0U); ///< Empty symbols with the code & index from cmap
	TinySym(const PixMapSym &refSym); ///< Creates tiny symbol based on a much larger reference symbol

	/// Used to create the centroid of a cluster
	TinySym(const cv::Point2d &mc_, double avgPixVal_, const cv::Mat &mat_,
			const cv::Mat &hAvgProj_, const cv::Mat &vAvgProj_,
			const cv::Mat &backslashDiagAvgProj_, const cv::Mat &slashDiagAvgProj_);

	/// Serializes this TinySym object to ar
	template<class Archive>
	void serialize(Archive &ar, const unsigned int version) {
		SymData::serialize(ar, version);
		ar & mat & 
			hAvgProj & vAvgProj & 
			backslashDiagAvgProj & slashDiagAvgProj;
	}
};

BOOST_CLASS_VERSION(TinySym, TinySym::VERSION);

/// container with TinySym-s
typedef std::vector<const TinySym> VTinySyms;

#endif // H_TINY_SYM