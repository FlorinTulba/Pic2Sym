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

#ifndef UNIT_TESTING

#include "clusterSerialization.h"
#include "serializer.h"

#pragma warning ( push, 0 )

#include <fstream>
#include <iostream>

#ifndef AI_REVIEWER_CHECK
#	include <boost/archive/binary_oarchive.hpp>
#	include <boost/archive/binary_iarchive.hpp>
#endif // AI_REVIEWER_CHECK not defined

#pragma warning ( pop )

using namespace std;
using namespace boost::archive;

unsigned ClusterIO::VERSION_FROM_LAST_IO_OP = UINT_MAX;

ClusterIO& ClusterIO::operator=(ClusterIO &&other) {
	if(this != &other) {
		clustersCount = other.clustersCount;
		clusterLabels = std::move(other.clusterLabels);
	}
	return *this;
}

bool ClusterIO::loadFrom(const stringType &path) {
	ifstream ifs(path, ios::binary);
	if(!ifs) {
		cerr<<"Couldn't find / open: " <<path<<endl;
		return false;
	}

	ClusterIO draftClusters; // load clusters in a draft object
#ifndef AI_REVIEWER_CHECK
	if(false == load<binary_iarchive>(ifs, path, draftClusters))
		return false;
#endif // AI_REVIEWER_CHECK

	*this = std::move(draftClusters);

	if(olderVersionDuringLastIO()) {
		ifs.close();

		// Rewriting the file. Same thread is used.
		if(saveTo(path))
			cout<<"Updated `"<<path<<"` because it used older versions of some classes required during loading!"<<endl;
	}

	return true;
}

bool ClusterIO::saveTo(const stringType &path) const {
	ofstream ofs(path, ios::binary | ios::trunc);
	if(!ofs) {
		cerr<<"Couldn't create / truncate: " <<path<<endl;
		return false;
	}

#ifndef AI_REVIEWER_CHECK
	return save<binary_oarchive>(ofs, path, *this);
#else // AI_REVIEWER_CHECK defined
	return true;
#endif // AI_REVIEWER_CHECK
}

void ClusterIO::reset(unsigned clustersCount_, vector<int> &&clusterLabels_) {
	clustersCount = clustersCount_;
	clusterLabels = move(clusterLabels_);
}

bool ClusterIO::olderVersionDuringLastIO() {
	return VERSION_FROM_LAST_IO_OP < VERSION;
}

#endif // UNIT_TESTING not defined
