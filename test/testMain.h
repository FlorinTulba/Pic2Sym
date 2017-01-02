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

#ifndef H_TEST_MAIN
#define H_TEST_MAIN

#include "match.h"
#include "matchParams.h"

#pragma warning ( push, 0 )

#include <boost/test/unit_test.hpp>
#include <boost/preprocessor/cat.hpp>

#pragma warning ( pop )

/// Defines test case named Name and ensures it will show its name when launched
#define AutoTestCase(Name) \
	BOOST_AUTO_TEST_CASE(Name) { \
		BOOST_TEST_MESSAGE("Running " BOOST_PP_STRINGIZE(Name))

/// unit testing namespace
namespace ut {

	/// Generates an uniformly-distributed random unsigned
	unsigned randUnifUint();

	/**
	Generates an uniformly-distributed random unsigned char.

	@param minIncl fist possible random value
	@param maxIncl last possible random value
	@return the random value
	*/
	unsigned char randUnsignedChar(unsigned char minIncl = 0U, unsigned char maxIncl = 255U);

	/// Used for a global fixture to reinitialize Controller's fields for each test
	struct Controller {

		/*
		Which Controller's fields to reinitialize.
		The global fixture sets them to true.
		After initialization each is set to false.
		*/
		static bool initImg, initFontEngine, initMatchEngine,
			initTransformer, initPreselManager, initComparator, initControlPanel;
	};

	/// Mock MatchEngine
	struct MatchEngine {};

	/// Fixture to be used before every test
	struct Fixt {
		Fixt();		///< set up
		~Fixt();	///< tear down
	};

	/**
	When detecting mismatches during Unit Testing, it displays a comparator window with them.

	@param testTitle the name of the test producing mismatches.
	It's appended with a unique id to distinguish among homonym tests
	from different unit testing sessions.
	@param mismatches vector of BestMatch objects
	*/
	void showMismatches(const std::string &testTitle,
		const std::vector<const BestMatch> &mismatches);
}

#endif