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

#ifdef UNIT_TESTING
#	include "../test/mockTinySymsDataSerialization.h"

#else // UNIT_TESTING not defined

#ifndef H_TINY_SYMS_DATA_SERIALIZATION
#define H_TINY_SYMS_DATA_SERIALIZATION

#include "tinySym.h"

#pragma warning ( push, 0 )

#include <boost/serialization/vector.hpp>

#pragma warning ( pop )

/// Clusters data that needs to be serialized
struct VTinySymsIO {
	// BUILD CLEAN WHEN THIS CHANGES!
	static const unsigned VERSION = 0U; ///< version of VTinySymsIO class

	/// reference to the tiny symbols to be serialized
	VTinySyms &tinySyms;

	VTinySymsIO(VTinySyms &tinySyms_);
	void operator=(const VTinySymsIO&) = delete;

	/// Serializes this VTinySymsIO object to ar
	template<class Archive>
	void serialize(Archive &ar, const unsigned int /*version*/) {
		ar & tinySyms;
	}

	/// Overwrites current content with the items read from file located at path. Returns false when loading fails.
	bool loadFrom(const std::string &path);

	/// Writes current content to file located at path. Returns false when saving fails.
	bool saveTo(const std::string &path) const;
};

BOOST_CLASS_VERSION(VTinySymsIO, VTinySymsIO::VERSION);

#endif // H_TINY_SYMS_DATA_SERIALIZATION

#endif // UNIT_TESTING not defined
