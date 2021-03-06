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

#ifndef H_TINY_SYM
#define H_TINY_SYM

#include "symData.h"
#include "tinySymBase.h"

#pragma warning ( push, 0 )

#ifndef AI_REVIEWER_CHECK
#	include <boost/serialization/array.hpp>
#	include <boost/serialization/base_object.hpp>
#	include <boost/serialization/version.hpp>
#endif // AI_REVIEWER_CHECK not defined

#pragma warning ( pop )

struct IPixMapSym; // Forward declaration

#pragma warning( disable : WARN_INHERITED_VIA_DOMINANCE )

/**
Data for tiny symbols.

ITinySym is virtual inherited to allow further diamond inheritance - see ICentroid and Centroid.
*/
class TinySym : public SymData, public virtual ITinySym {
#ifdef UNIT_TESTING // Unit Testing project may need these fields as public
public:
#else // UNIT_TESTING not defined - keep fields as protected
protected:
#endif // UNIT_TESTING
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

public:
#ifndef AI_REVIEWER_CHECK
	// BUILD CLEAN WHEN THIS CHANGES!
	static const unsigned VERSION = 0U; ///< version of ITinySym class
#endif // AI_REVIEWER_CHECK not defined

	TinySym(unsigned long code_ = ULONG_MAX, size_t symIdx_ = 0ULL); ///< Empty symbols with the code & index from cmap
	TinySym(const IPixMapSym &refSym); ///< Creates tiny symbol based on a much larger reference symbol

#ifdef UNIT_TESTING
	TinySym(const cv::Mat &negSym_, const cv::Point2d &mc_ = cv::Point2d(.5,.5), double avgPixVal_ = 0.); ///< generate the tiny symbol based on a negative
#endif // UNIT_TESTING defined

	/// Grounded version of the small symbol divided by TinySymArea
	const cv::Mat& getMat() const override final;

	/// Horizontal projection divided by TinySymSz
	const cv::Mat& getHAvgProj() const override final;

	/// Vertical projection divided by TinySymSz
	const cv::Mat& getVAvgProj() const override final;

	/// Normal diagonal projection divided by TinySymDiagsCount
	const cv::Mat& getBackslashDiagAvgProj() const override final;

	/// Inverse diagonal projection divided by TinySymDiagsCount
	const cv::Mat& getSlashDiagAvgProj() const override final;

	/**
	The classes with symbol data might need to aggregate more information.
	Thus, these classes could have several versions while some of them have serialized instances.

	When loading such older classes, the extra information needs to be deduced.
	It makes sense to resave the file with the additional data to avoid recomputing it
	when reloading the same file.

	The method below helps checking if the loaded classes are the newest ones or not.
	Saved classes always use the newest class version.

	Before serializing the first object of this class, the method should return false.
	*/
	static bool olderVersionDuringLastIO(); // There are no concurrent I/O operations on TinySym

private:
	friend class boost::serialization::access;

	/// UINT_MAX or the class version of the last loaded/saved object
	static unsigned VERSION_FROM_LAST_IO_OP; // There are no concurrent I/O operations on TinySym

	/// Serializes this TinySym object to ar
	template<class Archive>
	void serialize(Archive &ar, const unsigned version) {
		if(version > VERSION)
			THROW_WITH_VAR_MSG( // source file will be rewritten to reflect this (downgraded) VERSION
				"Cannot serialize future version (" + to_string(version) + ") of "
				"TinySym class (now at version " + to_string(VERSION) + ")!",
 				std::domain_error);

#ifndef AI_REVIEWER_CHECK
		ar & boost::serialization::base_object<SymData>(*this);
#endif // AI_REVIEWER_CHECK not defined

		ar & mat &
			hAvgProj & vAvgProj &
			backslashDiagAvgProj & slashDiagAvgProj;

		if(version != VERSION_FROM_LAST_IO_OP)
			VERSION_FROM_LAST_IO_OP = version;
	}
};

#pragma warning( default : WARN_INHERITED_VIA_DOMINANCE )

#ifndef AI_REVIEWER_CHECK
BOOST_CLASS_VERSION(TinySym, TinySym::VERSION);
#endif // AI_REVIEWER_CHECK not defined

/// container with TinySym-s
typedef std::vector<const TinySym> VTinySyms;

#endif // H_TINY_SYM
