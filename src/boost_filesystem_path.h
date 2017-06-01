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

#ifndef H_BOOST_FILESYSTEM_PATH
#define H_BOOST_FILESYSTEM_PATH

// Avoid using boost preprocessor when checking design of the project with AI Reviewer
#ifdef AI_REVIEWER_CHECK

#include <string>
#include <iostream>

namespace boost {
	namespace filesystem {
		class path {
		public:
			path(...) {}

			path& operator=(const char*) { return *this; }
			path& operator=(const std::string&) { return *this; }
			path& operator=(const path&) { return *this; }

			const std::string string() const { return ""; }
			const std::wstring  wstring() const { return L""; }
			const char* c_str() const { return nullptr; }

			path stem() const { return *this; }

			path& append(...) { return *this; }
			path& concat(...) { return *this; }
			path& operator/=(const path&) { return *this; }

			bool empty() const { return true; }
			bool has_parent_path() const { return true; }
			path& remove_filename() { return *this; }
			path& replace_extension(...) { return *this; }

			int compare(...) const { return 0; }
		};
	}
}

std::ostream& operator<<(std::ostream &os, const boost::filesystem::path&) { return os; }

#else // AI_REVIEWER_CHECK was not defined
#include <boost/filesystem/path.hpp>
#endif // AI_REVIEWER_CHECK

#endif // H_BOOST_FILESYSTEM_PATH