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

/*=======================================================================================

When USE_ZLIB_COMPRESSION is defined, this particular file makes use of the zlib library:

		http://www.zlib.net/  - (c) 1995-2013 Jean-loup Gailly and Mark Adler

Boost Iostreams doesn't include zlib and it has to be be added separately.

The compression / decompression is just an OPTIONAL feature for the Pic2Sym project
and it's used rarely only for saving some disk space.

That's why zlib wasn't included in the generic license header.

So, when the OPTIONAL support for compression / decompression isn't desired,
just comment its definition in 'compressOption.h' file

=======================================================================================*/

#ifndef H_SERIALIZER
#define H_SERIALIZER

#include "compressOption.h"

#ifdef USE_ZLIB_COMPRESSION

#include <string>

#include <boost/iostreams/filtering_streambuf.hpp>
#include <boost/iostreams/filter/zlib.hpp>

#endif // USE_ZLIB_COMPRESSION

/**
Overwriting content of obj with data read from src known as srcName.
When USE_ZLIB_COMPRESSION is defined, it assumes the read data should be decompressed.

@return false if any errors are detected; true otherwise
*/
template<class Archive, class Source, class T>
bool load(Source &src, const std::string &srcName, T &obj) {
	try {
#ifdef USE_ZLIB_COMPRESSION
		// The object to be read is compressed (disk-saving considerations), so it needs decompression:
		boost::iostreams::filtering_istreambuf fsIn;
		fsIn.push(boost::iostreams::zlib_decompressor());
		fsIn.push(src);

		Archive inA(fsIn);

#else // USE_ZLIB_COMPRESSION is not defined below
		Archive inA(src); // Reading from an uncompressed Source

#endif // USE_ZLIB_COMPRESSION

		inA >> obj;

		return true;

	} catch(...) {
		cerr<<"Couldn't load data from: " <<srcName<<endl;
		return false;
	}
}

/**
Writing content of obj to sink known by name sinkName.
When USE_ZLIB_COMPRESSION is defined, it assumes obj should be compressed before writing to sink.

@return false if any errors are detected; true otherwise
*/
template<class Archive, class Sink, class T>
bool save(Sink &sink, const std::string &sinkName, const T &obj) {
	try {
#ifdef USE_ZLIB_COMPRESSION
		/*
		Compressing this object before writing it achieves around 40% - 60% less required disk space.
		So, the uncompressed version would require around 166% - 250% more space than the compressed one.
	
		Incurred time penalty coming from compression / decompression is compensated by less disk accesses.
		Since time-performance looks almost the same with or without compression,
		compression is more attractive, because it saves some disk space.

		Largest generated file with current zlib compression is 21.2 MB ('DengXian_Regular_UNICODE_5.tsd',
		which contains more than 28000 symbols). Its uncompressed size would be 35.4 MB (167% more space).
		*/
		boost::iostreams::filtering_ostreambuf fsOut;

		// Passing 'boost::iostreams::zlib::best_compression' as parameter
		// to 'zlib_compressor()' from below brings just a minor required disk space reduction
		fsOut.push(boost::iostreams::zlib_compressor());
		fsOut.push(sink);

		Archive outA(fsOut);

#else // USE_ZLIB_COMPRESSION is not defined below
		Archive outA(sink); // Writing the uncompressed obj to the Sink

#endif // USE_ZLIB_COMPRESSION

		outA << obj;

		return true;

	} catch(...) {
		cerr<<"Couldn't save data to: " <<sinkName<<endl;
		return false;
	}
}

#endif // H_SERIALIZER