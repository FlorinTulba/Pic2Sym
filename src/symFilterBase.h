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
 
 (c) 2016, 2017 Florin Tulba <florintulba@yahoo.com>

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

#ifndef H_SYM_FILTER_BASE
#define H_SYM_FILTER_BASE

#pragma warning ( push, 0 )

#include <boost/optional/optional.hpp>

#pragma warning ( pop )

// Forward declarations
struct IPixMapSym;
struct SymFilterCache;

/// Interface used for filtering out some of the symbols from the charmap
struct ISymFilter /*abstract*/ {
	ISymFilter() {}
	ISymFilter(const ISymFilter&) = delete;
	void operator=(const ISymFilter&) = delete;

	/// Returns the id of the filter which detected that the symbol exhibits some undesired features.
	virtual boost::optional<unsigned> matchingFilterId(const IPixMapSym&, const SymFilterCache&) const = 0;

	virtual ~ISymFilter() = 0 {}
};

/// Implicit Symbol Filter, which just approves any symbol and is enabled by default
struct DefSymFilter : ISymFilter {
	DefSymFilter() : ISymFilter() {}
	DefSymFilter(const DefSymFilter&) = delete;
	void operator=(const DefSymFilter&) = delete;

	boost::optional<unsigned> matchingFilterId(const IPixMapSym&, const SymFilterCache&) const override { return boost::none; }
};

#endif // H_SYM_FILTER_BASE
