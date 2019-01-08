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

#ifdef UNIT_TESTING
#	include "../test/mockClusterSerialization.h"

#else // UNIT_TESTING not defined

#ifndef H_CLUSTER_SERIALIZATION
#define H_CLUSTER_SERIALIZATION

#include "misc.h"

#pragma warning ( push, 0 )

#include <vector>

#ifndef AI_REVIEWER_CHECK
#	include <boost/serialization/vector.hpp>
#	include <boost/serialization/version.hpp>
#endif // AI_REVIEWER_CHECK not defined

#pragma warning ( pop )

/// Clusters data that needs to be serialized
class ClusterIO {
	friend class boost::serialization::access;

	/// UINT_MAX or the class version of the last loaded/saved object
	static unsigned VERSION_FROM_LAST_IO_OP; // There are no concurrent I/O operations on ClusterIO

	/// Serializes this ClusterIO object to ar
	template<class Archive>
	void serialize(Archive &ar, const unsigned version) {
		if(version > VERSION)
			THROW_WITH_VAR_MSG( // source file will be rewritten to reflect this (downgraded) VERSION
				"Cannot serialize future version (" + to_string(version) + ") of "
				"ClusterIO class (now at version " + to_string(VERSION) + ")!",
				std::domain_error);

		ar & clustersCount;
#ifndef AI_REVIEWER_CHECK
		ar & clusterLabels;
#endif // AI_REVIEWER_CHECK not defined

		if(version != VERSION_FROM_LAST_IO_OP)
			VERSION_FROM_LAST_IO_OP = version;
	}

protected:
	/// assigned cluster for each symbol when sorted as within the cmap (by symIdx)
	std::vector<int> clusterLabels;	

	unsigned clustersCount = 0U;		///< total number of clusters

public:
	// BUILD CLEAN WHEN THIS CHANGES!
	static const unsigned VERSION = 0U; ///< version of ClusterIO class

	ClusterIO() = default;
	ClusterIO(const ClusterIO&) = delete;
	ClusterIO(ClusterIO&&) = delete;
	void operator=(const ClusterIO&) = delete;
	ClusterIO& operator=(ClusterIO &&other);

	/// Overwrites current content with the items read from file located at path. Returns false when loading fails.
	bool loadFrom(const std::stringType &path);

	/// Writes current content to file located at path. Returns false when saving fails.
	bool saveTo(const std::stringType &path) const;

	void reset(unsigned clustersCount_, std::vector<int> &&clusterLabels_);

	const std::vector<int>& getClusterLabels() const { return clusterLabels; }
	unsigned getClustersCount() const { return clustersCount; }

	/**
	The classes with cluster data might need to aggregate more information.
	Thus, these classes could have several versions while some of them have serialized instances.

	When loading such older classes, the extra information needs to be deduced.
	It makes sense to resave the file with the additional data to avoid recomputing it
	when reloading the same file.

	The method below helps checking if the loaded classes are the newest ones or not.
	Saved classes always use the newest class version.

	Before serializing the first object of this class, the method should return false.
	*/
	static bool olderVersionDuringLastIO(); // There are no concurrent I/O operations on ClusterIO
};

#ifndef AI_REVIEWER_CHECK
BOOST_CLASS_VERSION(ClusterIO, ClusterIO::VERSION);
#endif // AI_REVIEWER_CHECK not defined

#endif // H_CLUSTER_SERIALIZATION

#endif // UNIT_TESTING not defined
