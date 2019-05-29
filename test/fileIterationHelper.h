/******************************************************************************
 The application Pic2Sym approximates images by a
 grid of colored symbols with colored backgrounds.

 Copyrights from the libraries used by the program:
 - (c) 2003 Boost (www.boost.org)
     License: doc/licenses/Boost.lic
     http://www.boost.org/LICENSE_1_0.txt
 - (c) 2015-2016 OpenCV (www.opencv.org)
     License: doc/licenses/OpenCV.lic
     http://opencv.org/license/
 - (c) 1996-2002, 2006 The FreeType Project (www.freetype.org)
     License: doc/licenses/FTL.txt
     http://git.savannah.gnu.org/cgit/freetype/freetype2.git/plain/docs/FTL.TXT
 - (c) 1997-2002 OpenMP Architecture Review Board (www.openmp.org)
   (c) Microsoft Corporation (implementation for OpenMP C/C++ v2.0 March 2002)
     See: https://msdn.microsoft.com/en-us/library/8y6825x5.aspx
 - (c) 1995-2017 zlib software (Jean-loup Gailly and Mark Adler - www.zlib.net)
     License: doc/licenses/zlib.lic
     http://www.zlib.net/zlib_license.html


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
 If not, see: http://www.gnu.org/licenses/agpl-3.0.txt .
 *****************************************************************************/

#ifndef H_FILE_ITERATION_HELPER
#define H_FILE_ITERATION_HELPER

#pragma warning(push, 0)

#include <boost/preprocessor/cat.hpp>

#pragma warning(pop)

/// Generates [Name][Suffix] while waiting first their evaluation, unlike
/// Name##Suffix
#define suffixedItem(Name, Suffix) BOOST_PP_CAT(Name, Suffix)

/// Creates test suite [SuiteName][Suffix] that uses the FixtName fixture class
#define FIXTURE_TEST_SUITE_SUFFIX(FixtName, SuiteName, Suffix) \
  BOOST_FIXTURE_TEST_SUITE(suffixedItem(SuiteName, Suffix), FixtName)

/// Defines test case named Name and ensures it will show its name plus some
/// information when launched
#define TITLED_AUTO_TEST_CASE_(Name, Info1)                \
  BOOST_AUTO_TEST_CASE(suffixedItem(Name, Info1)) {        \
    BOOST_TEST_MESSAGE("Running " BOOST_PP_STRINGIZE(Name) \
                           BOOST_PP_STRINGIZE(Info1));

/// Defines data test case named Name and ensures it will show its name plus
/// some information when launched
#define DATA_TEST_CASE_SUFFIX(Name, Info, ...) \
  BOOST_DATA_TEST_CASE(suffixedItem(Name, Info), __VA_ARGS__)

#endif  // H_FILE_ITERATION_HELPER
