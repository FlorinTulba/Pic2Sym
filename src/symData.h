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

#ifndef H_SYM_DATA
#define H_SYM_DATA

#include "symDataBase.h"
#include "matSerialization.h"
#include "misc.h"

#pragma warning ( push, 0 )

#ifdef UNIT_TESTING
#	include <unordered_map>
#endif // UNIT_TESTING defined

#ifndef AI_REVIEWER_CHECK
#	include <boost/serialization/array.hpp>
#	include <boost/serialization/version.hpp>
#endif // AI_REVIEWER_CHECK not defined

#pragma warning ( pop )

extern const double INV_255();

struct IPixMapSym; // Forward declaration

/// Most symbol information
class SymData : public virtual ISymData {
#ifdef UNIT_TESTING // Unit Testing project may need these fields as public
public:
#else // UNIT_TESTING not defined - keep fields as protected
protected:
#endif // UNIT_TESTING
	/// mass center of the symbol given original fg & bg (coordinates are within a unit-square: 0..1 x 0..1)
	cv::Point2d mc = cv::Point2d(.5, .5);

	cv::Mat negSym;	///< negative of the symbol (0..255 byte for normal symbols; double for tiny)
	cv::Mat symMiu0;///< The pixel values (double) are shifted so that the average pixel value (miu) is 0

	MatArray masks;			///< various masks

	size_t symIdx = 0ULL;	///< symbol index within cmap
	double minVal = 0.;		///< the value of darkest pixel, range 0..1
	double diffMinMax = 1.;	///< difference between brightest and darkest pixels, each in 0..1
	double avgPixVal = 0.;	///< average pixel value, each pixel in 0..1
	double normSymMiu0 = 0.;///< norm L2 of (symbol - average pixel value)

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

	/// Computes symMiu0 and normSymMiu0 from sd based on glyph and its miu
	static void computeSymMiu0Related(const cv::Mat &glyph, double miu, SymData &sd);

	/**
	Computes most information about a symbol based on glyph parameter.
	It's also used to spare the constructor of SymData from performing computeFields' job.
	That can be useful when multiple threads add SymData items to a vector within a critical section,
	so each new SymData's fields are prepared outside the critical section to minimize blocking of
	other threads.
	*/
	static void computeFields(const cv::Mat &glyph, SymData &sd, bool forTinySym);

public:
#ifndef AI_REVIEWER_CHECK
	// BUILD CLEAN WHEN THIS CHANGES!
	static const unsigned VERSION = 1U; ///< version of ISymData class
#endif // AI_REVIEWER_CHECK not defined

	/// Fast constructor with the fields precomputed by computeFields - suitable for critical sections
	SymData(const cv::Mat &negSym_, const cv::Mat &symMiu0_,
			unsigned long code_, size_t symIdx_,
			double minVal_, double diffMinMax_,
			double avgPixVal_, double normSymMiu0_,
			const cv::Point2d &mc_,
			const MatArray &masks_,
			bool removable_ = false);
	SymData(const IPixMapSym &pms, unsigned sz, bool forTinySym);

	SymData(const SymData &other);
	SymData(SymData &&other); ///< moves the matrices from other (instead of just copying them)

	SymData& operator=(const SymData &other);
	SymData& operator=(SymData &&other); ///< moves the matrices from other (instead of just copying them)

	~SymData() {}

	/// mass center of the symbol given original fg & bg (coordinates are within a unit-square: 0..1 x 0..1)
	const cv::Point2d& getMc() const override final;

	/// negative of the symbol (0..255 byte for normal symbols; double for tiny)
	const cv::Mat& getNegSym() const override final;

	/// The pixel values (double) are shifted so that the average pixel value (miu) is 0 
	const cv::Mat& getSymMiu0() const override final;

	/// norm L2 of (symbol - average pixel value)
	double getNormSymMiu0() const override final;

	/// various masks
	const MatArray& getMasks() const override final;

	/// symbol index within cmap
	size_t getSymIdx() const override final;

#ifdef UNIT_TESTING
	/// the value of darkest pixel, range 0..1
	double getMinVal() const override final;
#endif // UNIT_TESTING defined

	/// difference between brightest and darkest pixels, each in 0..1
	double getDiffMinMax() const override final;

	/// average pixel value, each pixel in 0..1
	double getAvgPixVal() const override final;

	/// the code of the symbol
	unsigned long getCode() const override final;

	/**
	Enabled symbol filters might mark this symbol as removable,
	but PreserveRemovableSymbolsForExamination from configuration might allow it to remain
	in the active symbol set used during image transformation.

	However, when removable == true && PreserveRemovableSymbolsForExamination,
	the symbol will appear as marked (inversed) in the cmap viewer

	This field doesn't need to be serialized, as filtering options might be different
	for distinct run sessions.
	*/
	bool isRemovable() const override final;

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
	static bool olderVersionDuringLastIO(); // There are no concurrent I/O operations on SymData

#ifdef UNIT_TESTING
	typedef std::unordered_map< int, const cv::Mat > IdxMatMap; ///< Used in the SymData constructor from below

	/// Constructor that allows filling only the relevant matrices from MatArray
	SymData(unsigned long code_, size_t symIdx_, double minVal_, double diffMinMax_,
			double avgPixVal_, double normSymMiu0_,
			const cv::Point2d &mc_, const IdxMatMap &relevantMats,
			const cv::Mat &negSym_ = cv::Mat(), const cv::Mat &symMiu0_ = cv::Mat());

	/// A clone with different symIdx
	std::uniquePtr<const SymData> clone(size_t symIdx_) const;
#endif // UNIT_TESTING defined

protected:
	/* Constructors callable from derived classes only */

	/// Used for creation of TinySym and ClusterData
	SymData(unsigned long code_ = ULONG_MAX, size_t symIdx_ = 0ULL,
			double avgPixVal_ = 0., const cv::Point2d &mc_ = cv::Point2d(.5, .5));

	/// Used to create the TinySym centroid of a cluster 
	SymData(const cv::Point2d &mc_, double avgPixVal_);

private:
	friend class boost::serialization::access;

	/// UINT_MAX or the class version of the last loaded/saved object
	static unsigned VERSION_FROM_LAST_IO_OP; // There are no concurrent I/O operations on SymData

	/**
	Serializes this SymData object (apart from 'removable' field) to ar.

	'removable' field doesn't need to be serialized,
	as filtering options might be different for distinct run sessions.
	*/
	template<class Archive>
	void serialize(Archive &ar, const unsigned version) {
		if(version > VERSION)
			THROW_WITH_VAR_MSG( // source file will be rewritten to reflect this (downgraded) VERSION
				"Cannot serialize future version (" + to_string(version) + ") of "
				"SymData class (now at version " + to_string(VERSION) + ")!",
				std::domain_error);

		ar & code & symIdx & minVal & diffMinMax & avgPixVal;
		ar & mc.x & mc.y;
		ar & negSym;

#ifndef AI_REVIEWER_CHECK
		ar & masks;
#endif // AI_REVIEWER_CHECK not defined

#pragma warning( disable : WARN_CONST_COND_EXPR )
		if(Archive::is_loading::value)
#pragma warning( default : WARN_CONST_COND_EXPR )
		{
			if(version > 0U) {
				ar & symMiu0 & normSymMiu0;
			} else {
				cv::Mat sym = negSym.clone();
				sym.convertTo(sym, CV_64FC1, INV_255());
				computeSymMiu0Related(sym, avgPixVal, *this);
			}

			if(removable)
				removable = false; // Make sure a loaded symbol is initially not removable

		} else { // Saving
			ar & symMiu0 & normSymMiu0;
		}

		if(version != VERSION_FROM_LAST_IO_OP)
			VERSION_FROM_LAST_IO_OP = version;
	}
};

#ifndef AI_REVIEWER_CHECK
BOOST_CLASS_VERSION(SymData, SymData::VERSION);
#endif // AI_REVIEWER_CHECK not defined

#endif // H_SYM_DATA
