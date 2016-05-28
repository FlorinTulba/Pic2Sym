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
#include "settings.h"
#include "controller.h"

using namespace cv;
using namespace std;

namespace ut {
	/// Creates a Controller with default settings
	struct ControllerFixt : Fixt {
	private:
		Settings s; ///< tests shouldn't touch the settings

	protected:
		::Controller c; ///< the controller provided to the tests

	public:
		ControllerFixt() : Fixt(),
			s(),
			c(s) {}
	};

	/// Provides a font to the tests using this specialized fixture
	struct ControllerFixtUsingACertainFont : ControllerFixt {
		ControllerFixtUsingACertainFont(const string &fontPath = "res\\BPmonoBold.ttf") : ControllerFixt() {
			try {
				c.newFontFamily(fontPath);
			} catch(...) {
				cerr<<"Couldn't set '"<<fontPath<<"' font"<<endl;
			}
		}
	};
}

// Main Controller test suite
BOOST_FIXTURE_TEST_SUITE(Controller_Tests, ut::ControllerFixt)
	BOOST_AUTO_TEST_CASE(AttemptTransformation_NoSettings_NoTransformationPossible) {
		BOOST_TEST_MESSAGE("Running AttemptTransformation_NoSettings_NoTransformationPossible");
		BOOST_REQUIRE(!c.performTransformation()); // no font, no image
	}

	BOOST_AUTO_TEST_CASE(ProvidingAnImageToController_SetWrongImage_FailToSetImage) {
		BOOST_TEST_MESSAGE("Running ProvidingAnImageToController_SetWrongImage_FailToSetImage");
		BOOST_REQUIRE(!c.newImage(Mat()));
	}

	BOOST_AUTO_TEST_CASE(ProvidingAnImageToController_SetGrayImage_OkToSetImage) {
		BOOST_TEST_MESSAGE("Running ProvidingAnImageToController_SetGrayImage_OkToSetImage");
		Mat testPatch(c.getFontSize(), c.getFontSize(), CV_8UC1, Scalar(127));
		BOOST_REQUIRE(c.newImage(testPatch));
	}

	BOOST_AUTO_TEST_CASE(ProvidingAnImageToController_SetColorImage_OkToSetImage) {
		BOOST_TEST_MESSAGE("Running ProvidingAnImageToController_SetColorImage_OkToSetImage");
		Mat testColorPatch(c.getFontSize(), c.getFontSize(), CV_8UC3, Scalar::all(127));
		BOOST_REQUIRE(c.newImage(testColorPatch));
	}

	BOOST_AUTO_TEST_CASE(ProvidingAFontToController_UseBPmonoBold_NoThrow) {
		BOOST_TEST_MESSAGE("Running CheckController_UseBPmonoBold_NoThrow");
		BOOST_REQUIRE_NO_THROW(c.newFontFamily("res\\BPmonoBold.ttf"));
	}

	// Child Controller test suite whose tests use all BpMonoBold font
	BOOST_FIXTURE_TEST_SUITE(Controller_Tests_Using_BpMonoBoldFont, ut::ControllerFixtUsingACertainFont)
		BOOST_AUTO_TEST_CASE(CheckNewEncoding_UseAppleRomanFromBPmonoBold_NoThrow) {
			BOOST_TEST_MESSAGE("Running CheckNewEncoding_UseAppleRomanFromBPmonoBold_NoThrow");
			BOOST_REQUIRE_NO_THROW(c.newFontEncoding("APPLE_ROMAN"));
		}

		BOOST_AUTO_TEST_CASE(CheckNewSize_UseSize10FromAppleRomanOfBPmonoBold_NoThrow) {
			BOOST_TEST_MESSAGE("Running CheckNewSize_UseSize10FromAppleRomanOfBPmonoBold_NoThrow");
			c.newFontEncoding("APPLE_ROMAN");
			BOOST_REQUIRE_NO_THROW(c.newFontSize(10U));
		}

		BOOST_AUTO_TEST_CASE(AttemptTransformation_NoImageSet_NoTransformationPossible) {
			BOOST_TEST_MESSAGE("Running AttemptTransformation_NoImageSet_NoTransformationPossible");
			BOOST_REQUIRE(!c.performTransformation()); // no image yet
		}

		BOOST_AUTO_TEST_CASE(AttemptTransformation_SetGrayImage_OkToTransformImage) {
			BOOST_TEST_MESSAGE("Running AttemptTransformation_SetGrayImage_OkToTransformImage");

			Mat testPatch(c.getFontSize(), c.getFontSize(), CV_8UC1, Scalar(127));
			c.newImage(testPatch);
			BOOST_REQUIRE(c.performTransformation());
		}

		BOOST_AUTO_TEST_CASE(AttemptTransformation_SetColorImage_OkToTransformImage) {
			BOOST_TEST_MESSAGE("Running AttemptTransformation_SetColorImage_OkToTransformImage");

			Mat testColorPatch(c.getFontSize(), c.getFontSize(), CV_8UC3, Scalar::all(127));
			c.newImage(testColorPatch);
			BOOST_REQUIRE(c.performTransformation());
		}
	BOOST_AUTO_TEST_SUITE_END() // Controller_Tests_Using_BpMonoBoldFont

BOOST_AUTO_TEST_SUITE_END() // Controller_Tests

