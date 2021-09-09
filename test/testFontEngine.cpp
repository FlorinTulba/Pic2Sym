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

#include "controller.h"
#include "fontEngine.h"
#include "imgSettingsBase.h"
#include "jobMonitor.h"
#include "matchSettingsBase.h"
#include "misc.h"
#include "pixMapSym.h"
#include "progressNotifier.h"
#include "selectBranch.h"
#include "settings.h"
#include "symSettingsBase.h"

#pragma warning(push, 0)

#include <iostream>
#include <numeric>

#pragma warning(pop)

using namespace std;
using namespace boost;
using namespace cv;
using namespace gsl;

namespace pic2sym {

extern const bool PreserveRemovableSymbolsForExamination;
extern const unsigned Settings_MIN_FONT_SIZE;
extern const unsigned Settings_MAX_FONT_SIZE;

using namespace syms;

namespace ut {
/// Fixture reducing some declarative effort from tests
class FontEngineFixtComputations : public Fixt {
 public:
  /// Creates a fixture with the provided sz value
  FontEngineFixtComputations(unsigned sz_ = 10U) : Fixt() { setSz(sz_); }

  const Mat& getConsec() const { return consec; }
  const Mat& getRevConsec() const { return revConsec; }
  unsigned getSz() const { return sz; }
  double getArea() const { return area; }
  double getMaxGlyphSum() const { return maxGlyphSum; }

  /// Setter of sz. Updates consec and revConsec
  void setSz(unsigned sz_) {
    sz = sz_;
    area = double(sz) * sz;
    maxGlyphSum = 255. * area;

    // 0. parameter prevents using initializer_list ctor of Mat
    consec = Mat{1, (int)sz_, CV_64FC1, 0.};
    iota(BOUNDS_FOR_ITEM_TYPE(consec, double), (double)0.);  // 0..sz-1
    flip(consec, revConsec, 1);                              // sz-1..0
    revConsec = revConsec.t();
  }

 private:
  /// Patch side length. Use setSz and getSz to access it within tests
  unsigned sz;

  double area;         ///< square of sz
  double maxGlyphSum;  ///< max pixel sum for a glyph

  /// Column/row vectors of consecutive values (updated by setSz; getters
  /// available)
  Mat consec, revConsec;

 protected:
  /// Data defining the glyph. Uses ASCENDING vertical axis
  vector<unsigned char> pixels;

  /// Pixels will have rows x cols elements
  unsigned char rows{};
  unsigned char cols{};

  /// Location of glyph within the wrapping square
  unsigned char left{};
  unsigned char top{};

 public:
  /// Average pixel value to be computed within each test case
  double apv{};

  Point2d mc;  ///< mass-center to be computed within each test case
};

/// Provides a FontEngine object to the tests using this specialized fixture
class FontEngineFixtConfig : public Fixt {
 public:
  /// Creates the fixture providing a FontEngine object
  FontEngineFixtConfig() : Fixt(), s(), c{s}, jm{"", nullptr, 0.} {
    try {
      pfe = dynamic_cast<FontEngine*>(
          &c.getFontEngine(s.getSS()).useSymsMonitor(jm));

      // Forcing PreserveRemovableSymbolsForExamination on true during each test
      // case. Disposing removable symbols (marked by symbol filters) would
      // provide a variable set of input symbols to the tests, which would need
      // to update some expected values for each change within the filter
      // configuration. Thus, keeping all removable symbols lets these tests
      // unaffected by filter changes. Besides, testing on all symbols is
      // sufficient to check the correctness of the covered cases.
      if (!PreserveRemovableSymbolsForExamination)
        *refPreserveRemovableSymbolsForExamination = true;
    } catch (const runtime_error&) {
      cerr << "Couldn't create FontEngine" << endl;
    }
  }

  /// Reestablishes old value of PreserveRemovableSymbolsForExamination
  ~FontEngineFixtConfig() noexcept override {
    if (PreserveRemovableSymbolsForExamination !=
        origPreserveRemovableSymbolsForExamination)
      *refPreserveRemovableSymbolsForExamination =
          origPreserveRemovableSymbolsForExamination;
  }

 private:
  p2s::cfg::Settings s;  ///< default settings. Tests shouldn't touch it

  /// Controller using default settings. Tests shouldn't touch it
  pic2sym::Controller c;

 protected:
  p2s::ui::JobMonitor jm;

  /// Pointer to the FontEngine object needed within tests
  FontEngine* pfe = nullptr;

  /// initial value of PreserveRemovableSymbolsForExamination
  bool origPreserveRemovableSymbolsForExamination{
      PreserveRemovableSymbolsForExamination};

  /// Pointer to non-const PreserveRemovableSymbolsForExamination
  not_null<bool*> refPreserveRemovableSymbolsForExamination =
      const_cast<bool*>(&PreserveRemovableSymbolsForExamination);
};
}  // namespace ut

}  // namespace pic2sym

using namespace pic2sym;
using namespace syms;

// Test suite checking PixMapSym functionality
BOOST_FIXTURE_TEST_SUITE(FontEngine_Tests_Computations,
                         ut::FontEngineFixtComputations)
TITLED_AUTO_TEST_CASE(ComputeMassCenterAndAvgPixVal_0RowsOfData_CenterAnd0) {
  cols = 5U;
  top = narrow_cast<unsigned char>(getSz() - 1U);

  PixMapSym::computeMcAndAvgPixVal(
      getSz(), getMaxGlyphSum(), pixels, rows, cols, left, top, getConsec(),
      getRevConsec(), mc,
      apv);  // measured based on a DESCENDING vertical axis
  BOOST_REQUIRE(!apv);
  BOOST_TEST(mc.x == .5, test_tools::tolerance(1e-4));
  BOOST_TEST(mc.y == .5, test_tools::tolerance(1e-4));
  TITLED_AUTO_TEST_CASE_END
}

TITLED_AUTO_TEST_CASE(ComputeMassCenterAndAvgPixVal_0ColumnsOfData_CenterAnd0) {
  rows = 4U;
  top = narrow_cast<unsigned char>(getSz() - 1U);

  PixMapSym::computeMcAndAvgPixVal(
      getSz(), getMaxGlyphSum(), pixels, rows, cols, left, top, getConsec(),
      getRevConsec(), mc,
      apv);  // measured based on a DESCENDING vertical axis
  BOOST_REQUIRE(!apv);
  BOOST_TEST(mc.x == .5, test_tools::tolerance(1e-4));
  BOOST_TEST(mc.y == .5, test_tools::tolerance(1e-4));
  TITLED_AUTO_TEST_CASE_END
}

TITLED_AUTO_TEST_CASE(ComputeMassCenterAndAvgPixVal_AllDataIs0_CenterAnd0) {
  rows = cols = 5U;
  top = narrow_cast<unsigned char>(getSz() - 1U);
  pixels.assign((size_t)rows * cols, 0U);

  PixMapSym::computeMcAndAvgPixVal(
      getSz(), getMaxGlyphSum(), pixels, rows, cols, left, top, getConsec(),
      getRevConsec(), mc,
      apv);  // measured based on a DESCENDING vertical axis
  BOOST_TEST(apv == 0., test_tools::tolerance(1e-4));
  BOOST_TEST(mc.x == .5, test_tools::tolerance(1e-4));
  BOOST_TEST(mc.y == .5, test_tools::tolerance(1e-4));
  TITLED_AUTO_TEST_CASE_END
}

TITLED_AUTO_TEST_CASE(
    ComputeMassCenterAndAvgPixVal_2ChosenPixels_ExpectedValues) {
  rows = 1U;
  cols = 7U;
  left = 2U;
  pixels.assign((size_t)rows * cols, 0U);
  // 2 fixed points at a distance of 6: 170(2, 0) and 85(8, 0)
  pixels[0] = narrow_cast<unsigned char>(170);
  pixels[cols - 1] =
      narrow_cast<unsigned char>(85);  // 170 = 85*2, and 170 + 85 = 255

  PixMapSym::computeMcAndAvgPixVal(getSz(), getMaxGlyphSum(), pixels, rows,
                                   cols, left, top, getConsec(), getRevConsec(),
                                   mc, apv);
  BOOST_TEST(apv == 1. / getArea(),
             test_tools::tolerance(1e-4));  // 170 + 85 = 255
  // mc is measured based on a DESCENDING vertical axis
  BOOST_TEST(
      mc.x == 4. / (getSz() - 1U),
      test_tools::tolerance(1e-4));  // 4 is at one third the distance 2..8
  BOOST_TEST(mc.y == 0., test_tools::tolerance(1e-4));
  TITLED_AUTO_TEST_CASE_END
}

TITLED_AUTO_TEST_CASE(
    ComputeMassCenterAndAvgPixVal_UniformPatch_CenterAndAverage) {
  rows = cols = narrow_cast<unsigned char>(getSz());
  top = narrow_cast<unsigned char>(getSz() - 1U);
  const auto uc = ut::randUnsignedChar(1U);
  cout << "Checking patch filled with value " << (unsigned)uc << endl;
  pixels.assign((size_t)rows * cols, uc);

  PixMapSym::computeMcAndAvgPixVal(getSz(), getMaxGlyphSum(), pixels, rows,
                                   cols, left, top, getConsec(), getRevConsec(),
                                   mc, apv);
  BOOST_TEST(apv == getArea() * uc / getMaxGlyphSum(),
             test_tools::tolerance(1e-4));
  // mc is measured based on a DESCENDING vertical axis
  BOOST_TEST(mc.x == .5, test_tools::tolerance(1e-4));
  BOOST_TEST(mc.y == .5, test_tools::tolerance(1e-4));
  TITLED_AUTO_TEST_CASE_END
}

TITLED_AUTO_TEST_CASE(
    ComputeMassCenterAndAvgPixVal_SinglePixelNon0_PixelPositionAndPixelValueDivMaxGlyphSum) {
  rows = cols = 1U;
  left = ut::randUnsignedChar(0U, narrow_cast<unsigned char>(getSz() - 1U));
  top = ut::randUnsignedChar(0U, narrow_cast<unsigned char>(getSz() - 1U));
  const auto uc = ut::randUnsignedChar(1U);
  pixels.push_back(uc);
  cout << "Checking patch with a single non-zero pixel at: top="
       << (unsigned)top << ", left=" << (unsigned)left << endl;

  PixMapSym::computeMcAndAvgPixVal(getSz(), getMaxGlyphSum(), pixels, rows,
                                   cols, left, top, getConsec(), getRevConsec(),
                                   mc, apv);
  BOOST_TEST(apv == uc / getMaxGlyphSum(), test_tools::tolerance(1e-4));
  // mc is measured based on a DESCENDING vertical axis
  BOOST_TEST(mc.x == (double)left / (getSz() - 1U),
             test_tools::tolerance(1e-4));
  BOOST_TEST(mc.y == (double)top / (getSz() - 1U), test_tools::tolerance(1e-4));
  TITLED_AUTO_TEST_CASE_END
}

TITLED_AUTO_TEST_CASE(
    ComputeMassCenterAndAvgPixVal_3by3UniformArea_CenterOfAreaAnd9MulPixelValueDivMaxGlyphSum) {
  rows = cols = 3U;
  left = ut::randUnsignedChar(0U, narrow_cast<unsigned char>(getSz() - 3U));
  top = ut::randUnsignedChar(2U, narrow_cast<unsigned char>(getSz() - 1U));
  const auto uc = ut::randUnsignedChar(1U);
  pixels.assign((size_t)rows * cols, uc);  // all pixels are 'uc'

  PixMapSym::computeMcAndAvgPixVal(getSz(), getMaxGlyphSum(), pixels, rows,
                                   cols, left, top, getConsec(), getRevConsec(),
                                   mc, apv);
  BOOST_TEST(apv == (double)rows * cols * uc / getMaxGlyphSum(),
             test_tools::tolerance(1e-4));
  // mc is measured based on a DESCENDING vertical axis
  BOOST_TEST(mc.x == ((double)left + 1.) / (getSz() - 1U),
             test_tools::tolerance(1e-4));
  BOOST_TEST(mc.y == ((double)top - 1.) / (getSz() - 1U),
             test_tools::tolerance(1e-4));
  TITLED_AUTO_TEST_CASE_END
}

TITLED_AUTO_TEST_CASE(
    ComputeMassCenterAndAvgPixVal_2RandomChosenPixels_ComputedValues) {
  rows = cols = narrow_cast<unsigned char>(getSz());
  top = narrow_cast<unsigned char>(getSz() - 1U);
  pixels.assign((size_t)rows * cols, 0U);
  // random 2 points: p1(x1, y1) and p2(x2, y2)
  const unsigned char p1{ut::randUnsignedChar(1U)};  // random value 1..255
  const unsigned char p2{ut::randUnsignedChar(1U)};  // random value 1..255
  const unsigned char x1{ut::randUnsignedChar(
      0U, narrow_cast<unsigned char>(getSz() - 1U))};  // random value 0..sz-1
  const unsigned char y1{ut::randUnsignedChar(
      0U, narrow_cast<unsigned char>(getSz() - 1U))};  // random value 0..sz-1
  unsigned char x2{ut::randUnsignedChar(
      0U, narrow_cast<unsigned char>(getSz() - 1U))};  // random value 0..sz-1
  unsigned char y2{ut::randUnsignedChar(
      0U, narrow_cast<unsigned char>(getSz() - 1U))};  // random value 0..sz-1

  assert(getSz() >= 2U);
  while (x1 == x2 && y1 == y2) {  // Ensuring the 2 points don't overlap
    x2 = ut::randUnsignedChar(
        0U, narrow_cast<unsigned char>(getSz() - 1U));  // random value 0..sz-1
    y2 = ut::randUnsignedChar(
        0U, narrow_cast<unsigned char>(getSz() - 1U));  // random value 0..sz-1
  }

  pixels[x1 + (size_t)y1 * cols] = p1;
  pixels[x2 + (size_t)y2 * cols] = p2;
  cout << "Checking mass-center for 2 pixels: " << (unsigned)p1 << '('
       << (unsigned)x1 << ',' << (unsigned)y1 << "); " << (unsigned)p2 << '('
       << (unsigned)x2 << ',' << (unsigned)y2 << ')' << endl;

  PixMapSym::computeMcAndAvgPixVal(getSz(), getMaxGlyphSum(), pixels, rows,
                                   cols, left, top, getConsec(), getRevConsec(),
                                   mc, apv);
  BOOST_TEST(apv == ((double)p1 + p2) / getMaxGlyphSum(),
             test_tools::tolerance(1e-4));  // apv = (p1+p2)/(255*area)
  // mc is measured based on a DESCENDING vertical axis
  // ( (x1*p1+x2*p2)/(p1+p2)  ,  sz-1-(y1*p1+y2*p2)/(p1+p2) ) all downscaled by
  // (sz-1)
  BOOST_TEST(mc.x == ((double)x1 * p1 + (double)x2 * p2) /
                         (((double)p1 + p2) * (getSz() - 1U)),
             test_tools::tolerance(1e-4));
  BOOST_TEST(mc.y == 1. - (((double)y1 * p1 + (double)y2 * p2) /
                           (((double)p1 + p2) * (getSz() - 1U))),
             test_tools::tolerance(1e-4));
  TITLED_AUTO_TEST_CASE_END
}
BOOST_AUTO_TEST_SUITE_END()  // FontEngine_Tests_Computations

// Test suite checking FontEngine constraints
BOOST_FIXTURE_TEST_SUITE(FontEngine_Tests_Config, ut::FontEngineFixtConfig)
TITLED_AUTO_TEST_CASE(
    IncompleteFontConfig_NoFontFile_logicErrorsForFontOperations) {
  if (!pfe)
    return;

  FontEngine& fe = *pfe;
  string name;
  BOOST_CHECK_THROW(fe.setFontSz(10U), logic_error);
  BOOST_CHECK_THROW(fe.setEncoding("UNICODE"), logic_error);
  BOOST_CHECK_THROW(fe.setNthUniqueEncoding(0U), logic_error);
  BOOST_CHECK_THROW(fe.symsSet(), logic_error);
  BOOST_CHECK_THROW(fe.smallGlyphsCoverage(), logic_error);
  BOOST_CHECK_THROW(fe.uniqueEncodings(), logic_error);
  BOOST_CHECK_THROW(fe.upperSymsCount(), logic_error);
  BOOST_CHECK_THROW(fe.getEncoding(), logic_error);
  BOOST_CHECK(fe.getFamily() && !strlen(fe.getStyle()));
  BOOST_CHECK(fe.getStyle() && !strlen(fe.getStyle()));

  BOOST_REQUIRE_NO_THROW(name = fe.fontFileName());
  BOOST_REQUIRE(name.empty());
  TITLED_AUTO_TEST_CASE_END
}

TITLED_AUTO_TEST_CASE(FontConfig_IncorrectFontFile_CannotSetFont) {
  if (!pfe)
    return;

  bool correct{false};
  BOOST_CHECK_NO_THROW(correct = pfe->newFont(""));  // bad font file name
  BOOST_REQUIRE(!correct);
  TITLED_AUTO_TEST_CASE_END
}

TITLED_AUTO_TEST_CASE(FontConfig_NonScalableFont_CannotSetFont) {
  if (!pfe)
    return;

  bool correct{false};
  BOOST_CHECK_NO_THROW(correct = pfe->newFont("res\\vga855.fon"));
  BOOST_REQUIRE(!correct);
  TITLED_AUTO_TEST_CASE_END
}

TITLED_AUTO_TEST_CASE(FontConfig_CorrectFontFile_SettingFontOk) {
  if (!pfe)
    return;

  bool correct{false};
  BOOST_REQUIRE_NO_THROW(correct = pfe->newFont("res\\BPmonoBold.ttf"));
  BOOST_REQUIRE(correct);
  TITLED_AUTO_TEST_CASE_END
}

TITLED_AUTO_TEST_CASE(
    FontConfig_NoFontSize_CannotGetSymsSetNorSmallGlyphCoverage) {
  if (!pfe)
    return;

  FontEngine& fe = *pfe;
  unsigned unEncs{};
  unsigned encIdx{};
  string enc, fname;
  FT_String *fam = nullptr, *style = nullptr;
  fe.newFont("res\\BPmonoBold.ttf");

  // No font size throws
  BOOST_CHECK_THROW(fe.symsSet(), logic_error);
  BOOST_CHECK_THROW(fe.smallGlyphsCoverage(), logic_error);

  // Everything else ok
  BOOST_REQUIRE_NO_THROW(fname = fe.fontFileName());
  BOOST_CHECK(!fname.empty());

  BOOST_REQUIRE_NO_THROW(unEncs = fe.uniqueEncodings());
  BOOST_REQUIRE(unEncs == 2U);

  BOOST_REQUIRE_NO_THROW(enc = fe.getEncoding(&encIdx));
  BOOST_CHECK(enc == "UNICODE" && !encIdx);

  BOOST_REQUIRE_NO_THROW(fam = fe.getFamily());
  BOOST_CHECK(!strcmp(fam, "BPmono"));

  BOOST_REQUIRE_NO_THROW(style = fe.getStyle());
  BOOST_CHECK(!strcmp(style, "Bold"));

  // Setting Encodings
  BOOST_REQUIRE_NO_THROW(enc = fe.setEncoding(enc));  // same encoding
  BOOST_CHECK(!enc.empty());

  BOOST_REQUIRE_NO_THROW(enc = fe.setEncoding("APPLE_ROMAN"));  // new encoding
  BOOST_CHECK(!enc.empty());

  BOOST_CHECK(fe.getEncoding(&encIdx) == "APPLE_ROMAN");
  BOOST_CHECK(encIdx == 1U);

  // Recheck that no font size throws for the new encoding
  BOOST_CHECK_THROW(fe.symsSet(), logic_error);
  BOOST_CHECK_THROW(fe.smallGlyphsCoverage(), logic_error);
  TITLED_AUTO_TEST_CASE_END
}

TITLED_AUTO_TEST_CASE(FontConfig_CompleteConfig_NoProblemsExpected) {
  if (!pfe)
    return;

  FontEngine& fe = *pfe;
  fe.newFont("res\\BPmonoBold.ttf");

  BOOST_REQUIRE_THROW(fe.setFontSz(Settings_MIN_FONT_SIZE - 1U),
                      invalid_argument);
  BOOST_REQUIRE_NO_THROW(fe.setFontSz(Settings_MIN_FONT_SIZE));  // ok
  BOOST_REQUIRE_THROW(fe.setFontSz(Settings_MAX_FONT_SIZE + 1U),
                      invalid_argument);
  BOOST_REQUIRE_NO_THROW(fe.setFontSz(Settings_MAX_FONT_SIZE));  // ok

  BOOST_REQUIRE_NO_THROW(fe.setFontSz(10U));  // ok
  BOOST_CHECK_NO_THROW(fe.symsSet());
  BOOST_TEST(fe.smallGlyphsCoverage() == 0.107'764'7,
             test_tools::tolerance(1e-4));

  BOOST_REQUIRE(fe.setEncoding("APPLE_ROMAN"));  // APPLE_ROMAN
  BOOST_REQUIRE_NO_THROW(fe.setFontSz(15U));
  BOOST_CHECK_NO_THROW(fe.symsSet());
  BOOST_TEST(fe.smallGlyphsCoverage() == 0.107'485'8,
             test_tools::tolerance(1e-4));
  TITLED_AUTO_TEST_CASE_END
}
BOOST_AUTO_TEST_SUITE_END()  // FontEngine_Tests_Config
