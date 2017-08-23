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

#ifndef H_STD_MEMORY
#define H_STD_MEMORY

/*
AI Reviewer appears to need some help to appropriately keep track of the functions / methods
using std::(unique|shared)_ptr type either as return value or for the parameters.

The adopted solution is to define a parallel `std::(unique|shared)Ptr` used for
function / method signatures (and for fields and global variables, as well - as a precaution).

It was defined in the std namespace simply for convenience.
It can be constructed from / converted to the corresponding std::(unique|shared)_ptr.

When compiling without AI_REVIEWER_CHECK (this happens also for UNIT_TESTING),
(unique|shared)Ptr is defined as (unique|shared)_ptr.
*/

// Including <memory> anyway, to ensure that the expected sub-includes are collected
#include <memory>

#ifdef AI_REVIEWER_CHECK

namespace std {
	/// Base class with minimal interface for the placeholders of smart pointers
	template<class T>
	class smartPtr /*abstract*/ {
	protected:
		T *p = nullptr;

		smartPtr() = default; // protected => abstract class

	public:
		T& operator*() const { return *p; }
		T* operator->() const { return p; }

		T* get() const { return p; }

		explicit operator bool() const { return false; }
		bool operator!() const { return true; }
	};

	/// Replacement of unique_ptr, with minimum fields and methods
	template<class T>
	class uniquePtr : public smartPtr<T> {
	public:
		/// Construct from anything
		template<class ...Types>
		uniquePtr(Types&& ...) : smartPtr() {}

		// Can't copy uniquePtr
		template<class T2>
		uniquePtr(const uniquePtr<T2>&) = delete;
		template<class T2>
		void operator=(const uniquePtr<T2>&) = delete;
	};

	template<class T, class ...Types>
	uniquePtr<T> makeUnique(Types&& ...) { return uniquePtr<T>(); }

	/// Replacement of shared_ptr, with minimum fields and methods
	template<class T>
	class sharedPtr : public smartPtr<T> {
	public:
		/// Construct from anything
		template<class ...Types>
		sharedPtr(Types&& ...) : smartPtr() {}
	};

	template<class T, class ...Types>
	sharedPtr<T> makeShared(Types&& ...) { return sharedPtr<T>(); }
} // end namespace std

template<class T>
bool operator==(nullptr_t, const std::uniquePtr<T>&) { return true; }

#else // AI_REVIEWER_CHECK was not defined

// Just use (unique|shared)_ptr and make_(unique|shared) everywhere for normal / unit testing compilation
#define uniquePtr unique_ptr
#define makeUnique make_unique

#define sharedPtr shared_ptr
#define makeShared make_shared

#endif // AI_REVIEWER_CHECK

#endif // H_STD_MEMORY
