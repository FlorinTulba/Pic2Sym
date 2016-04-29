/****************************************************************************************
 The application Pic2Sym approximates images by a
 grid of colored symbols with colored backgrounds.

 This file was created on 2016-4-27
 and belongs to the Pic2Sym project.

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

#ifndef H_OMP_TRACE
#define H_OMP_TRACE

#ifndef UNIT_TESTING
#include <omp.h>
#endif

#if defined _DEBUG && !defined UNIT_TESTING

#include <cstdio>

extern omp_lock_t ompTraceLock;

/// Defines 'ompPrintf' function as 'printf' in Debug mode for original thread's team
#define ompPrintf(cond, formatText, ...) \
	if(cond) { \
			omp_set_lock(&ompTraceLock); \
			printf("[Thread %d] " #cond " : " formatText " (" __FILE__ ":%d)\n", \
				omp_get_thread_num(), __VA_ARGS__, __LINE__); \
			omp_unset_lock(&ompTraceLock); \
		}

#else // _DEBUG not defined or in UNIT_TESTING mode

#define ompPrintf(...)

#endif // _DEBUG

#endif // H_OMP_TRACE