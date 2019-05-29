/******************************************************************************
 The application Pic2Sym approximates images by a
 grid of colored symbols with colored backgrounds.

 Copyrights from the libraries used by the program:
 - (c) 2003 Boost (www.boost.org)
     License: doc/licenses/Boost.lic
     http://www.boost.org/LICENSE_1_0.txt
 - (c) 2015-2016 OpenCV (www.opencv.org)
     License: doc/licenses/OpenCV.lic
     http://opencv.org/license/
 - (c) 1996-2002, 2006 The FreeType Project (www.freetype.org)
     License: doc/licenses/FTL.txt
     http://git.savannah.gnu.org/cgit/freetype/freetype2.git/plain/docs/FTL.TXT
 - (c) 1997-2002 OpenMP Architecture Review Board (www.openmp.org)
   (c) Microsoft Corporation (implementation for OpenMP C/C++ v2.0 March 2002)
     See: https://msdn.microsoft.com/en-us/library/8y6825x5.aspx
 - (c) 1995-2017 zlib software (Jean-loup Gailly and Mark Adler - www.zlib.net)
     License: doc/licenses/zlib.lic
     http://www.zlib.net/zlib_license.html


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
 If not, see: http://www.gnu.org/licenses/agpl-3.0.txt .
 *****************************************************************************/

#include "precompiled.h"

#ifndef UNIT_TESTING

#include "serializer.h"
#include "tinySymsDataSerialization.h"

#pragma warning(push, 0)

#include <fstream>
#include <iostream>

#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>

#pragma warning(pop)

using namespace std;
using namespace boost::archive;

unsigned VTinySymsIO::VERSION_FROM_LAST_IO_OP = UINT_MAX;

VTinySymsIO::VTinySymsIO(VTinySyms& tinySyms_) noexcept : tinySyms(tinySyms_) {}

bool VTinySymsIO::loadFrom(const string& path) noexcept {
  ifstream ifs(path, ios::binary);
  if (!ifs) {
    cerr << "Couldn't find / open: " << path << endl;
    return false;
  }

  if (false == load<binary_iarchive>(ifs, path, *this)) {
    tinySyms.clear();  // leaves *this in a consistent state
    return false;
  }

  if (olderVersionDuringLastIO()) {
    ifs.close();

    // Rewriting the file. Same thread is used, as a detached thread needs
    // non-'volatile' tinySyms
    if (saveTo(path))
      cout << "Updated `" << path
           << "` because it used older versions "
              "of some classes required during loading!"
           << endl;
  }

  return true;
}

bool VTinySymsIO::saveTo(const string& path) const noexcept {
  if (ofstream ofs(path, ios::binary | ios::trunc); ofs)
    return save<binary_oarchive>(ofs, path, *this);

  cerr << "Couldn't create / truncate: " << path << endl;
  return false;
}

#pragma warning(disable : WARN_EXPR_ALWAYS_FALSE)
bool VTinySymsIO::olderVersionDuringLastIO() noexcept {
  return TinySym::olderVersionDuringLastIO() ||
         VERSION_FROM_LAST_IO_OP < VERSION;
}
#pragma warning(default : WARN_EXPR_ALWAYS_FALSE)

#endif  // UNIT_TESTING
