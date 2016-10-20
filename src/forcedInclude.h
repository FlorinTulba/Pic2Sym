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

// Header included first by all files from both projects

#ifndef H_FORCED_INCLUDE
#define H_FORCED_INCLUDE

/**
USE_ZLIB_COMPRESSION should be defined when (de)compression of generated files is desired.
Comment the define when the mentioned feature is not desired.
*/
#define USE_ZLIB_COMPRESSION
#ifdef USE_ZLIB_COMPRESSION
#	define BOOST_IOSTREAMS_NO_LIB
#endif // USE_ZLIB_COMPRESSION


/**
GENERATE_OPEN_MP_TRACE should be defined when traces from OpenMP are desired
in the main project. The UnitTesting project doesn't generate any OpenMP traces.
*/
//#define GENERATE_OPEN_MP_TRACE


/**
Original provided fonts are typically not square, so they need to be reshaped
sometimes even twice, to fit within a square of a desired size - symbol's size.

VIEW_CONCLUSIONS_FROM_RESHAPING_LOADED_FONTS should be defined when interested
in the details about a set of reshaped fonts.
*/
//#define VIEW_CONCLUSIONS_FROM_RESHAPING_LOADED_FONTS


/**
INSPECT_FFT_MAGNITUDE_SPECTRUM can be used in Debug mode to view the magnitude spectrum
from a 2D FFT transform in natural order.
A breakpoint should be set on a line after the shifting of the spectrum was performed
and the spectrum can be inspected as a matrix.
*/
//#define INSPECT_FFT_MAGNITUDE_SPECTRUM

#endif // H_FORCED_INCLUDE