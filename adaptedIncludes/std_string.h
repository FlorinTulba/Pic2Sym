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

#ifndef H_STD_STRING
#define H_STD_STRING

/*
AI Reviewer appears to need some help to appropriately keep track of the functions / methods
using std::[w]string type either as return value or for the parameters.

The adopted solution is to define a parallel `std::[w]stringType` used for
function / method signatures (and for fields and global variables, as well - as a precaution).

It was defined in the std namespace simply for convenience.
It can be constructed from / converted to the corresponding std::[w]string.

When compiling without AI_REVIEWER_CHECK (this happens also for UNIT_TESTING),
[w]stringType is defined as [w]string.

Boost.filesystem.path contains a `string_type`, so the `[w]stringType` name
was chosen to avoid a name clash.
*/

// Including <string> anyway, to ensure that:
//	- the expected sub-includes are collected
//	- std::char_traits and std::allocator are available for defining the parallel [w]stringType
#include <string>

#ifdef AI_REVIEWER_CHECK

#include <iostream>

#include <opencv2/core/core.hpp> // for converting basicStringType to cv::String

namespace std {
	/// Replacement of basic_string, with minimum types, fields and methods
	template<
		class CharT,
		class Traits = char_traits<CharT>,
		class Allocator = allocator<CharT>>
	class basicStringType {
	public:
		static const size_t npos = -1;

		/// basicStringType::iterator with minimal interface
		class iterator {
			CharT c;

		public:
			typedef ptrdiff_t difference_type;
			typedef CharT value_type;
			typedef CharT* pointer;
			typedef CharT& reference;
			typedef random_access_iterator_tag iterator_category;

			iterator(...) {}

			value_type& operator*() { return c; }
			const value_type& operator*() const { return c; }

			bool operator==(const iterator&) const { return true; }
			bool operator!=(const iterator&) const { return true; }
			iterator& operator++() { return *this; }
			iterator operator++(int) { return *this; }
		};

		/// basicStringType::const_iterator with minimal interface
		class const_iterator {
			CharT c;

		public:
			typedef ptrdiff_t difference_type;
			typedef CharT value_type;
			typedef const CharT* pointer;
			typedef const CharT& reference;
			typedef random_access_iterator_tag iterator_category;

			const_iterator(...) {}

			const value_type& operator*() const { return c; }

			bool operator==(const const_iterator&) const { return true; }
			bool operator!=(const const_iterator&) const { return true; }
			const_iterator& operator++() { return *this; }
			const_iterator operator++(int) { return *this; }
		};

		// Can be constructed from anything ([w]string included)
		template<class ...Types>
		basicStringType(Types&& ...) {}

		// Can be converted to std::[w]string or cv::String
		operator basic_string<CharT>() const { return basic_string<CharT>(); }
		operator cv::String() const { return cv::String(); }

		// Copy / move constructors / assignment operators
		basicStringType(const basicStringType&) {}
		basicStringType(basicStringType&&) {}
		basicStringType& operator=(const basicStringType&) { return *this; }
		basicStringType& operator=(basicStringType&&) { return *this; }
		basicStringType& operator=(const CharT*) { return *this; }

		// Local concat operators
		basicStringType operator+(const CharT*) const { return basicStringType(); }
		basicStringType operator+(const basicStringType&) const { return basicStringType(); }

		// forward [const] iterable
		iterator begin() { return iterator(); }
		const_iterator begin() const { return const_iterator(); }
		const_iterator cbegin() const { return const_iterator(); }
		iterator end() { return iterator(); }
		const_iterator end() const { return const_iterator(); }
		const_iterator cend() const { return const_iterator(); }

		// Simplified [w]string API
		basicStringType& assign(...) { return *this; }
		void clear() {}
		int compare(...) const { return 0; }
		const CharT* c_str() const { return nullptr; }
		bool empty() const { return true; }
		size_t find(...) const { return 0ULL; }
		size_t length() const { return 0ULL; }
		basicStringType substr(...) const { return basicStringType(); }
	};

	/// Replacement of getline
	template<class CharT>
	istream& getline(istream &is, basicStringType<CharT>&) { return is; }

	/// Replacement of less&lt;[w]string&gt;
	template<class CharT>
	struct less<basicStringType<CharT>> {
		bool operator()(const basicStringType<CharT>&, const basicStringType<CharT>&) const { return true; }
	};

	/// Replacement of less&lt;const [w]string&gt;
	template<class CharT>
	struct less<const basicStringType<CharT>> {
		bool operator()(const basicStringType<CharT>&, const basicStringType<CharT>&) const { return true; }
	};

	// std::[c]begin and std::[c]end for [w]stringType
	template<class CharT>
	typename basicStringType<CharT>::iterator begin(basicStringType<CharT> &s) {
		return s.begin();
	}
	template<class CharT>
	typename basicStringType<CharT>::const_iterator cbegin(const basicStringType<CharT> &s) {
		return s.cbegin();
	}
	template<class CharT>
	typename basicStringType<CharT>::iterator end(basicStringType<CharT> &s) {
		return s.end(); 
	}
	template<class CharT>
	typename basicStringType<CharT>::const_iterator cend(const basicStringType<CharT> &s) {
		return s.cend();
	}

	/// Replacement of string
	using stringType = basicStringType<char>;

	/// Replacement of to_string
	stringType to_string(...) { return stringType(); }

	/// Replacement of wstring
	using wstringType = basicStringType<wchar_t>;

	/// Replacement of to_wstring
	wstringType to_wstring(...) { return wstringType(); }
} // end namespace std

template<class CharT>
std::basic_ostream<CharT>& operator<<(std::basic_ostream<CharT> &os,
									  const std::basicStringType<CharT>&) {
	return os;
}

template<class CharT>
std::basicStringType<CharT> operator+(const char*, const std::basicStringType<CharT>&) {
	return std::basicStringType<CharT>();
}

#else // AI_REVIEWER_CHECK was not defined

// Just use [w]string everywhere for normal / unit testing compilation
#define stringType string
#define wstringType wstring

#endif // AI_REVIEWER_CHECK

#endif // H_STD_STRING
