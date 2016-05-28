/****************************************************************************************
 The application Pic2Sym approximates images by a
 grid of colored symbols with colored backgrounds.

 This file belongs to the UnitTesting project.

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

#include "testMain.h"

#include "controller.h"

using namespace cv;

BOOST_FIXTURE_TEST_SUITE(Controller_Tests, ut::Fixt)
	BOOST_AUTO_TEST_CASE(Check_Controller) {
		BOOST_TEST_MESSAGE("Running Check_Controller ...");

		Settings s(std::move(MatchSettings()));
		Controller c(s);

		BOOST_REQUIRE(!c.performTransformation()); // no font, no image

		BOOST_REQUIRE_NO_THROW(c.newFontFamily("res\\BPmonoBold.ttf"));
		BOOST_REQUIRE_NO_THROW(c.newFontEncoding("APPLE_ROMAN"));
		BOOST_REQUIRE_NO_THROW(c.newFontSize(10U));

		BOOST_REQUIRE(!c.performTransformation()); // no image yet

		BOOST_REQUIRE(!c.newImage(Mat())); // wrong image

		Mat testPatch(c.getFontSize(), c.getFontSize(), CV_8UC1, Scalar(127)),
			testColorPatch(c.getFontSize(), c.getFontSize(), CV_8UC3, Scalar::all(127));

		BOOST_REQUIRE(c.newImage(testPatch)); // image ok
		BOOST_REQUIRE(c.performTransformation()); // ok

		BOOST_REQUIRE(c.newImage(testColorPatch)); // image ok
		BOOST_REQUIRE(c.performTransformation()); // ok
	}
BOOST_AUTO_TEST_SUITE_END() // Controller_Tests
