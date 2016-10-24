/************************************************************************************************
 The application Pic2Sym approximates images by a
 grid of colored symbols with colored backgrounds.

 This file belongs to the UnitTesting project.

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

#ifndef H_FILE_ITERATION_HELPER
#define H_FILE_ITERATION_HELPER

#include <boost/preprocessor/cat.hpp>

/// Generates [Name][Suffix] while waiting first their evaluation, unlike Name##Suffix
#define suffixedItem(Name, Suffix) \
	BOOST_PP_CAT(Name, Suffix)

/// Creates test suite [SuiteName][Suffix] that uses the FixtName fixture class
#define FixtureTestSuiteSuffix(FixtName, SuiteName, Suffix) \
	BOOST_FIXTURE_TEST_SUITE(suffixedItem(SuiteName, Suffix), FixtName)

/// Defines test case named Name and ensures it will show its name plus some information when launched
#define AutoTestCase1(Name, Info1) \
	BOOST_AUTO_TEST_CASE(suffixedItem(Name, Info1)) { \
		BOOST_TEST_MESSAGE("Running " BOOST_PP_STRINGIZE(Name) BOOST_PP_STRINGIZE(Info1))

/// Defines data test case named Name and ensures it will show its name plus some information when launched
#define DataTestCase(Name, Info, ...) \
	BOOST_DATA_TEST_CASE(suffixedItem(Name, Info), __VA_ARGS__)

#endif // H_FILE_ITERATION_HELPER