/******************************************************************************
 The application Pic2Sym approximates images by a
 grid of colored symbols with colored backgrounds.

 Copyrights from the libraries used by the program:
 - (c) 2003-2021 Boost (www.boost.org)
     License: doc/licenses/Boost.lic
     http://www.boost.org/LICENSE_1_0.txt
 - (c) 2015-2021 OpenCV (www.opencv.org)
     License: doc/licenses/OpenCV.lic
     http://opencv.org/license/
 - (c) 1996-2021 The FreeType Project (www.freetype.org)
     License: doc/licenses/FTL.txt
     http://git.savannah.gnu.org/cgit/freetype/freetype2.git/plain/docs/FTL.TXT
 - (c) 1997-2021 OpenMP Architecture Review Board (www.openmp.org)
   (c) Microsoft Corporation (implementation for OpenMP C/C++ v2.0 March 2002)
     See: https://msdn.microsoft.com/en-us/library/8y6825x5.aspx
 - (c) 1995-2021 zlib software (Jean-loup Gailly and Mark Adler - www.zlib.net)
     License: doc/licenses/zlib.lic
     http://www.zlib.net/zlib_license.html
 - (c) 2015-2021 Microsoft Guidelines Support Library - github.com/microsoft/GSL
     License: doc/licenses/MicrosoftGSL.lic
     https://raw.githubusercontent.com/microsoft/GSL/main/LICENSE


 (c) 2016-2021 Florin Tulba <florintulba@yahoo.com>

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

#include "precompiled.h"
// This keeps precompiled.h first; Otherwise header sorting might move it

#include "testMain.h"

#include "controlPanelActionsBase.h"
#include "controller.h"
#include "imgSettingsBase.h"
#include "matchSettingsBase.h"
#include "selectBranch.h"
#include "settings.h"

using namespace cv;
using namespace std;
using namespace gsl;

namespace pic2sym::ut {
/// Creates a Controller with default settings
class ControllerFixt : public Fixt {
 protected:
  ControllerFixt() noexcept
      : Fixt(), s(), c{s}, cpa(&c.getControlPanelActions()) {}

  int fontSize() const { return (int)c.getFontSize(); }

 private:
  /// Tests shouldn't touch the settings; Keep it above pic2sym::Controller c
  p2s::cfg::Settings s;

 protected:
  /// The controller provided to the tests; Keep it below Settings s
  pic2sym::Controller c;
  not_null<IControlPanelActions*> cpa;
};

/// Provides a font to the tests using this specialized fixture
class ControllerFixtUsingACertainFont : public ControllerFixt {
 public:
  ControllerFixtUsingACertainFont(
      const string& fontPath = "res/BPmonoBold.ttf") noexcept
      : ControllerFixt() {
    try {
      cpa->newFontFamily(fontPath, (unsigned)fontSize());
    } catch (const exception& e) {
      cerr << "Couldn't set '" << fontPath << "' font!\nReason: " << e.what()
           << endl;
    }
  }
};

}  // namespace pic2sym::ut

using namespace pic2sym;

// Main Controller test suite
BOOST_FIXTURE_TEST_SUITE(Controller_Tests, ut::ControllerFixt)
TITLED_AUTO_TEST_CASE(
    AttemptTransformation_NoSettings_NoTransformationPossible) {
  BOOST_REQUIRE(!cpa->performTransformation());  // no font, no image
  TITLED_AUTO_TEST_CASE_END
}

TITLED_AUTO_TEST_CASE(
    ProvidingAnImageToController_SetWrongImage_FailToSetImage) {
  BOOST_REQUIRE(!cpa->newImage(Mat()));
  TITLED_AUTO_TEST_CASE_END
}

TITLED_AUTO_TEST_CASE(ProvidingAnImageToController_SetGrayImage_OkToSetImage) {
  Mat testPatch{fontSize(), fontSize(), CV_8UC1, Scalar{127.}};
  BOOST_REQUIRE(cpa->newImage(testPatch));
  TITLED_AUTO_TEST_CASE_END
}

TITLED_AUTO_TEST_CASE(ProvidingAnImageToController_SetColorImage_OkToSetImage) {
  Mat testColorPatch{fontSize(), fontSize(), CV_8UC3, Scalar::all(127.)};
  BOOST_REQUIRE(cpa->newImage(testColorPatch));
  TITLED_AUTO_TEST_CASE_END
}

TITLED_AUTO_TEST_CASE(ProvidingAFontToController_UseBPmonoBold_NoThrow) {
  BOOST_REQUIRE_NO_THROW(
      cpa->newFontFamily("res\\BPmonoBold.ttf", (unsigned)fontSize()));
  TITLED_AUTO_TEST_CASE_END
}

// Child Controller test suite whose tests use all BpMonoBold font
BOOST_FIXTURE_TEST_SUITE(Controller_Tests_Using_BpMonoBoldFont,
                         ut::ControllerFixtUsingACertainFont)
TITLED_AUTO_TEST_CASE(CheckNewEncoding_UseAppleRomanFromBPmonoBold_NoThrow) {
  BOOST_REQUIRE_NO_THROW(cpa->newFontEncoding("APPLE_ROMAN"));
  TITLED_AUTO_TEST_CASE_END
}

TITLED_AUTO_TEST_CASE(
    CheckNewSize_UseSize10FromAppleRomanOfBPmonoBold_NoThrow) {
  cpa->newFontEncoding("APPLE_ROMAN");
  BOOST_REQUIRE_NO_THROW(cpa->newFontSize(10U));
  TITLED_AUTO_TEST_CASE_END
}

TITLED_AUTO_TEST_CASE(
    AttemptTransformation_NoImageSet_NoTransformationPossible) {
  BOOST_REQUIRE(!cpa->performTransformation());  // no image yet
  TITLED_AUTO_TEST_CASE_END
}

TITLED_AUTO_TEST_CASE(AttemptTransformation_SetGrayImage_OkToTransformImage) {
  Mat testPatch{fontSize(), fontSize(), CV_8UC1, Scalar{127.}};
  cpa->newImage(testPatch);

  // enable just one random aspect, to avoid missing enabled aspects
  cpa->newUnderGlyphCorrectnessFactor(1.);
  BOOST_REQUIRE(cpa->performTransformation());
  TITLED_AUTO_TEST_CASE_END
}

TITLED_AUTO_TEST_CASE(
    AttemptTransformation_NoEnabledAspects_NoTransformationPossible) {
  Mat testPatch{fontSize(), fontSize(), CV_8UC1, Scalar{127.}};
  cpa->newImage(testPatch);
  BOOST_REQUIRE(!cpa->performTransformation());
  TITLED_AUTO_TEST_CASE_END
}

TITLED_AUTO_TEST_CASE(AttemptTransformation_SetColorImage_OkToTransformImage) {
  Mat testColorPatch{fontSize(), fontSize(), CV_8UC3, Scalar::all(127.)};
  cpa->newImage(testColorPatch);

  // enable just one random aspect, to avoid missing enabled aspects
  cpa->newUnderGlyphCorrectnessFactor(1.);
  BOOST_REQUIRE(cpa->performTransformation());
  TITLED_AUTO_TEST_CASE_END
}
BOOST_AUTO_TEST_SUITE_END()  // Controller_Tests_Using_BpMonoBoldFont

BOOST_AUTO_TEST_SUITE_END()  // Controller_Tests
