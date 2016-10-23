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

#include "tinySymsDataSerialization.h"

VTinySymsIO::VTinySymsIO(VTinySyms &tinySyms_) : tinySyms(tinySyms_) {}

#ifndef UNIT_TESTING

#include "serializer.h"

#include <fstream>
#include <iostream>

#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>

using namespace std;
using namespace boost::archive;

bool VTinySymsIO::loadFrom(const string &path) {
	ifstream ifs(path, ios::binary);
	if(!ifs) {
		cerr<<"Couldn't find / open: " <<path<<endl;
		return false;
	}

	if(false == load<binary_iarchive>(ifs, path, *this)) {
		tinySyms.clear(); // leaves *this in a consistent state
		return false;
	}
	
	return true;
}

bool VTinySymsIO::saveTo(const string &path) const {
	ofstream ofs(path, ios::binary | ios::trunc);
	if(!ofs) {
		cerr<<"Couldn't create / truncate: " <<path<<endl;
		return false;
	}

	return save<binary_oarchive>(ofs, path, *this);;
}

#endif // UNIT_TESTING