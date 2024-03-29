/******************************************************************************
 The application Pic2Sym approximates images by a
 grid of colored symbols with colored backgrounds.

 Copyrights from the libraries used by the program:
 - (c) 2003-2021 Boost (www.boost.org)
     License: doc/licenses/Boost.lic
     http://www.boost.org/LICENSE_1_0.txt
 - (c) 2015-2021 OpenCV (www.opencv.org)
     License: doc/licenses/OpenCV.lic
     http://opencv.org/license/
 - (c) 1996-2021 The FreeType Project (www.freetype.org)
     License: doc/licenses/FTL.txt
     http://git.savannah.gnu.org/cgit/freetype/freetype2.git/plain/docs/FTL.TXT
 - (c) 1997-2021 OpenMP Architecture Review Board (www.openmp.org)
   (c) Microsoft Corporation (implementation for OpenMP C/C++ v2.0 March 2002)
     See: https://msdn.microsoft.com/en-us/library/8y6825x5.aspx
 - (c) 1995-2021 zlib software (Jean-loup Gailly and Mark Adler - www.zlib.net)
     License: doc/licenses/zlib.lic
     http://www.zlib.net/zlib_license.html
 - (c) 2015-2021 Microsoft Guidelines Support Library - github.com/microsoft/GSL
     License: doc/licenses/MicrosoftGSL.lic
     https://raw.githubusercontent.com/microsoft/GSL/main/LICENSE


 (c) 2016-2021 Florin Tulba <florintulba@yahoo.com>

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

/*=============================================================================

 The optional feature of compressing / decompressing reusable data to save disk
 space was implemented with Boost Iostreams, who needs an additional compressing
 software library. The chosen library was zlib.

 When USE_ZLIB_COMPRESSION is defined, the mentioned feature is activated.
 To disable it, just comment its definition in 'compressOption.h' file

 =============================================================================*/

#ifdef UNIT_TESTING
#error Should not include this file when UNIT_TESTING is defined

#else  // UNIT_TESTING not defined

#ifndef H_SERIALIZER
#define H_SERIALIZER

#include "compressOption.h"
#include "warnings.h"

#ifdef USE_ZLIB_COMPRESSION

#pragma warning(push, 0)

#include <boost/iostreams/filter/zlib.hpp>
#include <boost/iostreams/filtering_streambuf.hpp>

#include <iomanip>
#include <iostream>
#include <string>

#pragma warning(pop)

#endif  // USE_ZLIB_COMPRESSION

/**
Overwriting content of obj with data read from src known as srcName.
When USE_ZLIB_COMPRESSION is defined, it assumes the read data should be
decompressed.

NOTE:
When loading fails, it could happen after parts of the obj have been already
overwritten. So obj might be in an inconsistent state. Therefore, when loading
fails, either try rollback-ing the incomplete changes, or just perform the load
on a draft and only if loading succeeds, copy/move draft's content onto the
target object.

@return false if any errors are detected; true otherwise
*/
template <class Archive, class Source, class T>
bool load(Source& src, const std::string& srcName, T& obj) noexcept {
#pragma warning(disable : WARN_SEH_NOT_CAUGHT)
  try {
#ifdef USE_ZLIB_COMPRESSION
    // The object to be read is compressed (disk-saving considerations), so it
    // needs decompression:
    boost::iostreams::filtering_istreambuf fsIn;
    fsIn.push(boost::iostreams::zlib_decompressor());
    fsIn.push(src);

    Archive inA{fsIn};

#else  // USE_ZLIB_COMPRESSION is not defined below
    Archive inA{src};    // Reading from an uncompressed Source

#endif  // USE_ZLIB_COMPRESSION

    inA >> obj;

    return true;

  } catch (const std::exception& e) {
    std::cerr << "Couldn't load data from: '" << srcName
              << "'!\nReason: " << e.what() << std::endl;
    return false;
  }
#pragma warning(default : WARN_SEH_NOT_CAUGHT)
}

/**
Writing content of obj to sink known by name sinkName.
When USE_ZLIB_COMPRESSION is defined, it assumes obj should be compressed before
writing to sink.

@return false if any errors are detected; true otherwise
*/
template <class Archive, class Sink, class T>
bool save(Sink& sink, const std::string& sinkName, const T& obj) noexcept {
#pragma warning(disable : WARN_SEH_NOT_CAUGHT)
  try {
#ifdef USE_ZLIB_COMPRESSION
    /*
    Compressing this object before writing it achieves around 40% - 60% less
    required disk space. So, the uncompressed version would require around 166%
    - 250% more space than the compressed one.

    Incurred time penalty coming from compression / decompression is compensated
    by less disk accesses. Since time-performance looks almost the same with or
    without compression, compression is more attractive, because it saves some
    disk space.

    Largest generated file with current zlib compression is 21.2 MB
    ('DengXian_Regular_UNICODE_5.tsd', which contains more than 28000 symbols).
    Its uncompressed size would be 35.4 MB (167% more space).
    */
    boost::iostreams::filtering_ostreambuf fsOut;

    // Passing 'boost::iostreams::zlib::best_compression' as parameter
    // to 'zlib_compressor()' from below brings just a minor required disk space
    // reduction
    fsOut.push(boost::iostreams::zlib_compressor());
    fsOut.push(sink);

    Archive outA{fsOut};

#else  // USE_ZLIB_COMPRESSION is not defined below
    Archive outA{sink};  // Writing the uncompressed obj to the Sink

#endif  // USE_ZLIB_COMPRESSION

    outA << obj;

    return true;

  } catch (const std::exception& e) {
    std::cerr << "Couldn't save data to: '" << sinkName
              << "'!\nReason: " << e.what() << std::endl;
    return false;
  }
#pragma warning(default : WARN_SEH_NOT_CAUGHT)
}

#endif  // H_SERIALIZER

#endif  // UNIT_TESTING
