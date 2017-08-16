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

#include "testMain.h"
#include "selectBranch.h"
#include "settings.h"
#include "symSettingsBase.h"
#include "imgSettingsBase.h"
#include "matchSettingsBase.h"
#include "controller.h"
#include "controlPanelActionsBase.h"

using namespace cv;
using namespace std;

namespace ut {
	/// Creates a Controller with default settings
	struct ControllerFixt : Fixt {
	private:
		Settings s; ///< tests shouldn't touch the settings

	protected:
		::Controller c; ///< the controller provided to the tests
		std::shared_ptr<IControlPanelActions> cpa;

	public:
		ControllerFixt() : Fixt(),
			s(),
			c(s),
			cpa(c.getControlPanelActions()) {}
	};

	/// Provides a font to the tests using this specialized fixture
	struct ControllerFixtUsingACertainFont : ControllerFixt {
		ControllerFixtUsingACertainFont(const string &fontPath = "res\\BPmonoBold.ttf") : ControllerFixt() {
			try {
				cpa->newFontFamily(fontPath);
			} catch(...) {
				cerr<<"Couldn't set '"<<fontPath<<"' font"<<endl;
			}
		}
	};
}

// Main Controller test suite
BOOST_FIXTURE_TEST_SUITE(Controller_Tests, ut::ControllerFixt)
	AutoTestCase(AttemptTransformation_NoSettings_NoTransformationPossible);
		BOOST_REQUIRE(!cpa->performTransformation()); // no font, no image
	}

	AutoTestCase(ProvidingAnImageToController_SetWrongImage_FailToSetImage);
		BOOST_REQUIRE(!cpa->newImage(Mat()));
	}

	AutoTestCase(ProvidingAnImageToController_SetGrayImage_OkToSetImage);
		Mat testPatch(c.getFontSize(), c.getFontSize(), CV_8UC1, Scalar(127));
		BOOST_REQUIRE(cpa->newImage(testPatch));
	}

	AutoTestCase(ProvidingAnImageToController_SetColorImage_OkToSetImage);
		Mat testColorPatch(c.getFontSize(), c.getFontSize(), CV_8UC3, Scalar::all(127));
		BOOST_REQUIRE(cpa->newImage(testColorPatch));
	}

	AutoTestCase(ProvidingAFontToController_UseBPmonoBold_NoThrow);
		BOOST_REQUIRE_NO_THROW(cpa->newFontFamily("res\\BPmonoBold.ttf"));
	}

	// Child Controller test suite whose tests use all BpMonoBold font
	BOOST_FIXTURE_TEST_SUITE(Controller_Tests_Using_BpMonoBoldFont, ut::ControllerFixtUsingACertainFont)
		AutoTestCase(CheckNewEncoding_UseAppleRomanFromBPmonoBold_NoThrow);
			BOOST_REQUIRE_NO_THROW(cpa->newFontEncoding("APPLE_ROMAN"));
		}

		AutoTestCase(CheckNewSize_UseSize10FromAppleRomanOfBPmonoBold_NoThrow);
			cpa->newFontEncoding("APPLE_ROMAN");
			BOOST_REQUIRE_NO_THROW(cpa->newFontSize(10U));
		}

		AutoTestCase(AttemptTransformation_NoImageSet_NoTransformationPossible);
			BOOST_REQUIRE(!cpa->performTransformation()); // no image yet
		}

		AutoTestCase(AttemptTransformation_SetGrayImage_OkToTransformImage);
			Mat testPatch(c.getFontSize(), c.getFontSize(), CV_8UC1, Scalar(127));
			cpa->newImage(testPatch);
			cpa->newUnderGlyphCorrectnessFactor(1.); // enable just one random aspect, to avoid missing enabled aspects
			BOOST_REQUIRE(cpa->performTransformation());
		}

		AutoTestCase(AttemptTransformation_NoEnabledAspects_NoTransformationPossible);
			Mat testPatch(c.getFontSize(), c.getFontSize(), CV_8UC1, Scalar(127));
			cpa->newImage(testPatch);
			BOOST_REQUIRE(!cpa->performTransformation());
		}

		AutoTestCase(AttemptTransformation_SetColorImage_OkToTransformImage);
			Mat testColorPatch(c.getFontSize(), c.getFontSize(), CV_8UC3, Scalar::all(127));
			cpa->newImage(testColorPatch);
			cpa->newUnderGlyphCorrectnessFactor(1.); // enable just one random aspect, to avoid missing enabled aspects
			BOOST_REQUIRE(cpa->performTransformation());
		}
	BOOST_AUTO_TEST_SUITE_END() // Controller_Tests_Using_BpMonoBoldFont

BOOST_AUTO_TEST_SUITE_END() // Controller_Tests
