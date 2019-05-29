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

#include "precompiled.h"

/*
Iterating this file twice, for both values of the boolean setting
PreselectionByTinySyms. It's simpler than duplicating each test or using the
BOOST_DATA_TEST_CASE approach.
*/
#if !BOOST_PP_IS_ITERATING
#pragma warning(push, 0)

#include <iomanip>

#include <boost/preprocessor/iteration/iterate.hpp>

#pragma warning(pop)

// Common part until #else (included just once)
#include "bestMatch.h"
#include "blur.h"
#include "controlPanelActionsBase.h"
#include "controller.h"
#include "correlationAspect.h"
#include "fileIterationHelper.h"
#include "imgSettings.h"
#include "matchAspects.h"
#include "matchAssessment.h"
#include "matchEngine.h"
#include "matchParams.h"
#include "matchProgressWithPreselection.h"
#include "matchSettings.h"
#include "matchSupportWithPreselection.h"
#include "misc.h"
#include "patch.h"
#include "preselectSyms.h"
#include "preselectionHelper.h"
#include "scoreThresholds.h"
#include "selectBranch.h"
#include "settings.h"
#include "structuralSimilarity.h"
#include "symData.h"
#include "symSettings.h"
#include "testMain.h"

#pragma warning(push, 0)

#include <algorithm>
#include <iterator>
#include <numeric>
#include <optional>
#include <random>
#include <unordered_set>

#include <boost/test/data/test_case.hpp>

#include <opencv2/imgproc/imgproc.hpp>

#pragma warning(pop)

using namespace cv;
using namespace std;
using namespace boost;

extern template class std::optional<double>;
extern template class unordered_set<unsigned>;

extern const string StructuralSimilarity_BlurType;
extern const string ClusterAlgName;
extern const unsigned ShortListLength;
extern const double AdmitOnShortListEvenForInferiorScoreFactor;
extern const bool Transform_BlurredPatches_InsteadOf_Originals;
extern unsigned TinySymsSz();
static const unsigned TinySymsSize = TinySymsSz();

#define tol test_tools::tolerance

namespace ut {
/// dummy value
constexpr double NOT_RELEVANT_D = 0.;
/// dummy value
constexpr unsigned long NOT_RELEVANT_UL = ULONG_MAX;
/// dummy value
constexpr size_t NOT_RELEVANT_SZ = 0ULL;
/// dummy value
const Point2d NOT_RELEVANT_POINT;
/// dummy value
const Mat NOT_RELEVANT_MAT;

/**
Changes randomly the foreground and background of patchD255.

However, it will keep fg & bg at least 30 brightness units apart.

patchD255 has min value 255*minVal01 and max value = min value +
255*diffMinMax01.
*/
void alterFgBg(Mat& patchD255, double minVal01, double diffMinMax01) {
  const auto newFg = randUnsignedChar();
  unsigned char newBg;
  int newDiff;
  do {
    newBg = randUnsignedChar();
    newDiff = (int)newFg - (int)newBg;
  } while (abs(newDiff) <
           30);  // keep fg & bg at least 30 brightness units apart

  patchD255 =
      (patchD255 - (minVal01 * 255)) * (newDiff / (255 * diffMinMax01)) + newBg;
#ifdef _DEBUG
  double newMinVal, newMaxVal;
  minMaxIdx(patchD255, &newMinVal, &newMaxVal);
  assert(newMinVal > -.5);
  assert(newMaxVal < 255.5);
#endif  // _DEBUG
}

/**
Adds white noise to patchD255 with max amplitude maxAmplitude0255.
The percentage of affected pixels from patchD255 is affectedPercentage01.
*/
void addWhiteNoise(Mat& patchD255,
                   double affectedPercentage01,
                   unsigned char maxAmplitude0255) {
  const int side = patchD255.rows, area = side * side,
            affectedCount = (int)(affectedPercentage01 * area);

  int noise;
  const unsigned twiceMaxAmplitude = ((unsigned)maxAmplitude0255) << 1;
  int prevVal, below, above, newVal;
  unordered_set<unsigned> affected;
  unsigned linearized;
  for (int i = 0; i < affectedCount; ++i) {
    do {
      linearized = randUnifUint() % (unsigned)area;
    } while (affected.find(linearized) != affected.cend());
    affected.insert(linearized);

    // Coordinate inside the Mat, expressed as quotient and remainder of
    // linearized
    const auto [posQuot, posRem] = div((int)linearized, side);

    prevVal = (int)round(patchD255.at<double>(posQuot, posRem));

    below = max(0, min((int)maxAmplitude0255, prevVal));
    above = max(0, min((int)maxAmplitude0255, 255 - prevVal));
    do {
      noise = (int)(randUnifUint() % (unsigned)(above + below + 1)) - below;
    } while (noise == 0);

    newVal = prevVal + noise;
    patchD255.at<double>(posQuot, posRem) = newVal;
    assert(newVal > -.5 && newVal < 255.5);
  }
}

/// Creates the matrix 'randUc' of sz x sz random unsigned chars
void randInit(unsigned sz, Mat& randUc) {
  vector<unsigned char> randV;
  generate_n(back_inserter(randV), sz * sz, [] { return randUnsignedChar(); });
  randUc = Mat(sz, sz, CV_8UC1, (void*)randV.data());
}

/// Creates matrix 'invHalfD255' sz x sz with first half black (0) and second
/// half white (255)
void updateInvHalfD255(unsigned sz, Mat& invHalfD255) {
  invHalfD255 = Mat(sz, sz, CV_64FC1, Scalar(0.));
  invHalfD255.rowRange(sz / 2, sz) = 255.;
}

/**
Creates 2 unique_ptr to 2 symbol data objects.

@param sz patch side length
@param sdWithHorizEdgeMask symbol data unique_ptr for a glyph whose vertical
halves are white and black. The 2 rows mid height define a horizontal edge mask
@param sdWithVertEdgeMask symbol data unique_ptr for a glyph whose vertical
halves are white and black. The 2 columns mid width define a vertical edge mask,
which simply instructs where to look for edges within this glyph. It doesn't
correspond with the actual horizontal edge of the glyph, but it will check the
patches for a vertical edge.
*/
void updateSymDataOfHalfFullGlyphs(
    unsigned sz,
    std::unique_ptr<SymData>& sdWithHorizEdgeMask,
    std::unique_ptr<SymData>& sdWithVertEdgeMask) {
  // 2 rows mid height
  Mat horBeltUc = Mat(sz, sz, CV_8UC1, Scalar(0U));
  horBeltUc.rowRange(sz / 2 - 1, sz / 2 + 1) = 255U;

  // 1st horizontal half - full
  Mat halfUc = Mat(sz, sz, CV_8UC1, Scalar(0U));
  halfUc.rowRange(0, sz / 2) = 255U;
  Mat halfD1 = Mat(sz, sz, CV_64FC1, Scalar(0.));
  halfD1.rowRange(0, sz / 2) = 1.;
  const Mat halfD1Miu0 = halfD1 - .5;

  // 2nd horizontal half - full
  Mat invHalfUc = 255U - halfUc;

  // A glyph half 0, half 255 (horizontally)
  sdWithHorizEdgeMask = std::make_unique<SymData>(
      NOT_RELEVANT_UL,  // glyph code (not relevant here)
      NOT_RELEVANT_SZ,  // symbol index (not relevant here)
      0.,               // min glyph value (0..1 range)
      1.,               // difference between min and max glyph (0..1 range)
      .5,               // avgPixVal = 255*(sz^2/2)/(255*sz^2) = 1/2
      .5 * sz,          // normSymMiu0
      Point2d(.5, (sz / 2. - 1.) /
                      (2. *
                       (sz - 1U))),  // glyph's mass center downscaled by (sz-1)
      SymData::IdxMatMap{{(int)ISymData::MaskType::Fg, halfUc},
                         {(int)ISymData::MaskType::Bg, invHalfUc},
                         {(int)ISymData::MaskType::Edge, horBeltUc},

                         // grounded glyph is same as glyph (min is already 0)
                         {(int)ISymData::MaskType::GroundedSym, halfD1}},
      invHalfUc, halfD1Miu0);

  // A glyph half 0, half 255 (vertically)
  sdWithVertEdgeMask = std::make_unique<SymData>(
      NOT_RELEVANT_UL,  // glyph code (not relevant here)
      NOT_RELEVANT_SZ,  // symbol index (not relevant here)
      0.,               // min glyph value (0..1 range)
      1.,               // difference between min and max glyph (0..1 range)
      .5,               // avgPixVal = 255*(sz^2/2)/(255*sz^2) = 1/2
      .5 * sz,          // normSymMiu0
      Point2d((sz / 2. - 1.) / (2. * (sz - 1U)),
              .5),  // glyph's mass center downscaled by (sz-1)
      SymData::IdxMatMap{{(int)ISymData::MaskType::Fg, halfUc.t()},
                         {(int)ISymData::MaskType::Bg, invHalfUc.t()},
                         {(int)ISymData::MaskType::Edge, horBeltUc.t()},

                         // grounded glyph is same as glyph (min is already 0)
                         {(int)ISymData::MaskType::GroundedSym, halfD1.t()}},
      invHalfUc.t(), halfD1Miu0.t());
}

/// Fixture helping tests computing matching parameters
template <bool PreselMode>
class MatchParamsFixt : public PreselFixt<PreselMode> {
 public:
  /**
  Creates a fixture useful for the tests computing match parameters.

  @param sz_ patch side length. Select an even value, as the tests need to set
  exactly half pixels
  */
  explicit MatchParamsFixt(unsigned sz_ = 50U) noexcept
      : PreselFixt(), cd(PreselectionByTinySyms) {
    setSz(sz_);
  }

  /// Random initialization of randUc and computing corresponding randD1 and
  /// randD255
  void randInitPatch() noexcept {
    randInit(sz, randUc);
    randUc.convertTo(randD1, CV_64FC1, 1. / 255);
    randUc.convertTo(randD255, CV_64FC1);
  }

  const Mat& getEmptyUc() const noexcept { return emptyUc; }
  const Mat& getEmptyD() const noexcept { return emptyD; }
  const Mat& getFullUc() const noexcept { return fullUc; }
  const Mat& getConsec() const noexcept { return consec; }
  const Mat& getRandUc() const noexcept { return randUc; }
  const Mat& getRandD255() const noexcept { return randD255; }
  const Mat& getInvHalfD255() const noexcept { return invHalfD255; }
  const CachedData& getCd() const noexcept { return cd; }
  const SymData& getSdWithHorizEdgeMask() const noexcept {
    return *sdWithHorizEdgeMask;
  }
  const SymData& getSdWithVertEdgeMask() const noexcept {
    return *sdWithVertEdgeMask;
  }
  unsigned getSz() const noexcept { return sz; }
  unsigned getSzSq() const noexcept { return szSq; }

  /// Updates sz, szSq, cd, consec and the matrices empty, random and full
  void setSz(unsigned sz_) noexcept {
    sz = sz_;
    szSq = sz * sz;
    emptyUc = Mat(sz, sz, CV_8UC1, Scalar(0U));
    emptyD = Mat(sz, sz, CV_64FC1, Scalar(0.));
    fullUc = Mat(sz, sz, CV_8UC1, Scalar(255U));
    cd.useNewSymSize(sz);
    consec = cd.getConsec().clone();

    randInitPatch();

    updateInvHalfD255(sz, invHalfD255);
    updateSymDataOfHalfFullGlyphs(sz, sdWithHorizEdgeMask, sdWithVertEdgeMask);
  }

  /// Patch side length (tests should use provided public accessor methods)
  unsigned sz;
  unsigned szSq;  ///< sz^2
  Mat emptyUc;    ///< Empty matrix sz x sz of unsigned chars
  Mat emptyD;     ///< Empty matrix sz x sz of doubles
  Mat fullUc;     ///< sz x sz matrix filled with 255 (unsigned char)

  /// sz x sz matrix vertically split in 2. One half white, the other black
  Mat invHalfD255;
  Mat consec;       ///< 0 .. sz-1 consecutive double values
  CachedDataRW cd;  ///< cached data based on sz
  Mat randUc;       ///< sz x sz random unsigned chars
  Mat randD1;       ///< sz x sz random doubles (0 .. 1)
  Mat randD255;     ///< sz x sz random doubles (0 .. 255)
  std::unique_ptr<SymData> sdWithHorizEdgeMask, sdWithVertEdgeMask;

  std::optional<double> miu;   ///< mean computed within tests
  std::optional<double> sdev;  ///< standard deviation computed within tests
  MatchParams mp;              ///< matching parameters computed during tests
  double minV;  ///< min of the error between the patch and its approximation
  double maxV;  ///< max of the error between the patch and its approximation
};

/// Fixture for the matching aspects
template <bool PreselMode>
class MatchAspectsFixt : public PreselFixt<PreselMode> {
 public:
  /**
  Creates a fixture useful for the tests checking the match aspects.

  @param sz_ patch side length. Select an even value, as the tests need to set
  exactly half pixels
  */
  explicit MatchAspectsFixt(unsigned sz_ = 50U) noexcept
      : PreselFixt(), cd(PreselectionByTinySyms) {
    setSz(sz_);
  }

  /// Updates sz and area
  void setSz(unsigned sz_) noexcept {
    sz = sz_;
    area = sz * sz;
  }
  unsigned getSz() const noexcept { return sz; }
  unsigned getArea() const noexcept { return area; }

  /// Patch side length (tests should use provided public accessor methods)
  unsigned sz;

  unsigned area;  ///< sz^2 (Use getter within tests)

 protected:
  CachedDataRW cd;   ///< cached data that can be changed during tests
  MatchSettings ms;  ///< determines which aspect is tested

 public:
  MatchParams mp;  ///< tests compute these match parameters
  Mat patchD255;   ///< declares the patch to be approximated
  double res;   ///< assessment of the match between the patch and the resulted
                ///< approximation
  double minV;  ///< min of the error between the patch and its approximation
  double maxV;  ///< max of the error between the patch and its approximation

  /// help for blur
  const IBlurEngine& blurSupport =
      BlurEngine::byName(StructuralSimilarity_BlurType);
};

template <bool PreselMode>
class AlteredCmapFixture : public PreselFixt<PreselMode> {
 public:
  const string oldClusterAlgName = ClusterAlgName;

  explicit AlteredCmapFixture(const string& clustAlgName) noexcept
      : PreselFixt() {
    if (ClusterAlgName != clustAlgName)
      const_cast<string&>(ClusterAlgName) = clustAlgName;
  }

  ~AlteredCmapFixture() noexcept {
    if (ClusterAlgName != oldClusterAlgName)
      const_cast<string&>(ClusterAlgName) = oldClusterAlgName;
  }
};

/// Parameterized test case. Uniform patches get approximated by completely
/// faded glyphs.
void checkParams_UniformPatch_GlyphConvergesToPatch(unsigned sz,
                                                    const CachedData& cd,
                                                    const SymData& symData) {
  const auto valRand = randUnsignedChar(1U);
  const double valRandSq = (double)valRand * valRand;
  Mat unifPatchD255(sz, sz, CV_64FC1, Scalar(valRand));
  MatchParams mp;
  double minV, maxV;
  mp.computePatchApprox(unifPatchD255, symData);
  BOOST_REQUIRE(mp.getPatchApprox());
  minMaxIdx(mp.getPatchApprox().value(), &minV, &maxV);
  BOOST_TEST(minV == (double)valRand, tol(1e-4));
  BOOST_TEST(maxV == (double)valRand, tol(1e-4));
  // Its mass-center will be ( (sz-1)/2 , (sz-1)/2 )
  mp.computeMcPatchApprox(unifPatchD255, symData, cd);
  BOOST_REQUIRE(mp.getMcPatchApprox());
  BOOST_TEST(mp.getMcPatchApprox()->x == .5, tol(1e-4));
  BOOST_TEST(mp.getMcPatchApprox()->y == .5, tol(1e-4));
  mp.computeSdevEdge(unifPatchD255, symData);
  BOOST_REQUIRE(mp.getSdevEdge());
  BOOST_TEST(*mp.getSdevEdge() == 0., tol(1e-4));
  mp.computePatchSum(unifPatchD255);
  BOOST_REQUIRE(mp.getPatchSum());
  BOOST_TEST(*mp.getPatchSum() == valRand * cd.getSzSq(), tol(1e-4));
  mp.computePatchSq(unifPatchD255);
  BOOST_REQUIRE(mp.getPatchSq());
  minMaxIdx(mp.getPatchSq().value(), &minV, &maxV);
  BOOST_TEST(minV == valRandSq, tol(1e-4));
  BOOST_TEST(maxV == valRandSq, tol(1e-4));
  mp.computeNormPatchMinMiu(unifPatchD255, cd);
  BOOST_REQUIRE(mp.getNormPatchMinMiu());
  BOOST_TEST(*mp.getNormPatchMinMiu() == 0., tol(1e-4));
  mp.computeAbsCorr(unifPatchD255, symData, cd);
  BOOST_REQUIRE(mp.getAbsCorr());
  BOOST_TEST(*mp.getAbsCorr() == 1., tol(1e-4));
}

/// Parameterized test case. A glyph which is the inverse of a patch converges
/// to the patch.
void checkParams_TestedGlyphIsInverseOfPatch_GlyphConvergesToPatch(
    unsigned sz,
    const Mat& invHalfD255,
    const CachedData& cd,
    const SymData& symData,
    bool vert = false) {
  MatchParams mp;
  double minV, maxV;
  mp.computePatchApprox(invHalfD255, symData);
  BOOST_REQUIRE(mp.getPatchApprox());
  minMaxIdx(mp.getPatchApprox().value() - invHalfD255, &minV, &maxV);
  BOOST_TEST(minV == 0., tol(1e-4));
  BOOST_TEST(maxV == 0., tol(1e-4));
  // Its mass-center will be ( (sz-1)/2 ,  (3*sz/2-1)/2 ) all downsized by
  // (sz-1)
  mp.computeMcPatchApprox(invHalfD255, symData, cd);
  BOOST_REQUIRE(mp.getMcPatchApprox());
  const double mcDim1 = .5, mcDim2 = (3. * sz / 2. - 1.) / (2. * (sz - 1.));
  if (!vert) {
    BOOST_TEST(mp.getMcPatchApprox()->x == mcDim1, tol(1e-4));
    BOOST_TEST(mp.getMcPatchApprox()->y == mcDim2, tol(1e-4));
  } else {
    BOOST_TEST(mp.getMcPatchApprox()->y == mcDim1, tol(1e-4));
    BOOST_TEST(mp.getMcPatchApprox()->x == mcDim2, tol(1e-4));
  }
  mp.computeSdevEdge(invHalfD255, symData);
  BOOST_REQUIRE(mp.getSdevEdge());
  BOOST_TEST(*mp.getSdevEdge() == 0., tol(1e-4));
  mp.computePatchSum(invHalfD255);
  BOOST_REQUIRE(mp.getPatchSum());
  BOOST_TEST(*mp.getPatchSum() == 127.5 * cd.getSzSq(), tol(1e-4));
  mp.computePatchSq(invHalfD255);
  BOOST_REQUIRE(mp.getPatchSq());
  minMaxIdx(mp.getPatchSq().value() - invHalfD255.mul(invHalfD255), &minV,
            &maxV);
  BOOST_TEST(minV == 0., tol(1e-4));
  BOOST_TEST(maxV == 0., tol(1e-4));
  mp.computeNormPatchMinMiu(invHalfD255, cd);
  BOOST_REQUIRE(mp.getNormPatchMinMiu());
  BOOST_TEST(*mp.getNormPatchMinMiu() == 127.5 * sz, tol(1e-4));
  mp.computeAbsCorr(invHalfD255, symData, cd);
  BOOST_REQUIRE(mp.getAbsCorr());
  BOOST_TEST(*mp.getAbsCorr() == 1., tol(1e-4));
}

/// Parameterized test case. A glyph which is the highest-contrast version of a
/// patch converges to the patch.
void checkParams_TestHalfFullGlyphOnDimmerPatch_GlyphLoosesContrast(
    unsigned sz,
    const Mat& emptyD,
    const CachedData& cd,
    const SymData& symData,
    bool vert = false) {
  MatchParams mp;
  double minV, maxV;
  // Testing the mentioned glyph on a patch half 85, half 170=2*85
  Mat twoBandsD255 = emptyD.clone();
  twoBandsD255.rowRange(0, sz / 2) = 170.;
  twoBandsD255.rowRange(sz / 2, sz) = 85.;
  if (vert)
    twoBandsD255 = twoBandsD255.t();

  mp.computePatchApprox(twoBandsD255, symData);
  BOOST_REQUIRE(mp.getPatchApprox());
  minMaxIdx(mp.getPatchApprox().value() - twoBandsD255, &minV, &maxV);
  BOOST_TEST(minV == 0., tol(1e-4));
  BOOST_TEST(maxV == 0., tol(1e-4));
  // Its mass-center will be ( (sz-1)/2 ,  (5*sz-6)/12 )
  mp.computeMcPatchApprox(twoBandsD255, symData, cd);
  BOOST_REQUIRE(mp.getMcPatchApprox());
  const double mcDim1 = .5, mcDim2 = (5. * sz - 6.) / (12. * (sz - 1.));
  if (!vert) {
    BOOST_TEST(mp.getMcPatchApprox()->x == mcDim1, tol(1e-4));
    BOOST_TEST(mp.getMcPatchApprox()->y == mcDim2, tol(1e-4));
  } else {
    BOOST_TEST(mp.getMcPatchApprox()->y == mcDim1, tol(1e-4));
    BOOST_TEST(mp.getMcPatchApprox()->x == mcDim2, tol(1e-4));
  }
  mp.computeSdevEdge(twoBandsD255, symData);
  BOOST_REQUIRE(mp.getSdevEdge());
  BOOST_TEST(*mp.getSdevEdge() == 0., tol(1e-4));
  mp.computePatchSum(twoBandsD255);
  BOOST_REQUIRE(mp.getPatchSum());
  BOOST_TEST(*mp.getPatchSum() == 127.5 * cd.getSzSq(), tol(1e-4));
  mp.computePatchSq(twoBandsD255);
  BOOST_REQUIRE(mp.getPatchSq());
  minMaxIdx(mp.getPatchSq().value() - twoBandsD255.mul(twoBandsD255), &minV,
            &maxV);
  BOOST_TEST(minV == 0., tol(1e-4));
  BOOST_TEST(maxV == 0., tol(1e-4));
  mp.computeNormPatchMinMiu(twoBandsD255, cd);
  BOOST_REQUIRE(mp.getNormPatchMinMiu());
  BOOST_TEST(*mp.getNormPatchMinMiu() == 42.5 * sz, tol(1e-4));
  mp.computeAbsCorr(twoBandsD255, symData, cd);
  BOOST_REQUIRE(mp.getAbsCorr());
  BOOST_TEST(*mp.getAbsCorr() == 1., tol(1e-4));
}

// msArray and fonts from below are used to generate the data sets used within
// CheckAlteredCmap_UsingAspects_ExpectLessThan3or55PercentErrors test below

/// array of all MatchSetting-s configurations to be tested for all selected
/// font configurations
const MatchSettings msArray[] = {
    MatchSettings().set_kSsim(1.), MatchSettings().set_kCorrel(1.),
    MatchSettings().set_kSdevFg(1.).set_kSdevEdge(1.).set_kSdevBg(1.)};

/**
map of fonts to be tested.

The elements are the full combination of font family and the desired encoding.

Below are some variants:
Courier Bold UNICODE ("C:\\Windows\\Fonts\\courbd.ttf") > 2800 glyphs; There are
2 almost identical COMMA-s and QUOTE-s. Envy Code R UNICODE
("C:\\Windows\\Fonts\\Envy Code R Bold.ttf") > 600 glyphs Bp Mono Bold
("res\\BPmonoBold.ttf") - 210 glyphs for UNICODE, 134 for APPLE_ROMAN
*/
map<string, string> fonts{{"res\\BPmonoBold.ttf", "APPLE_ROMAN"}};

typedef decltype(fonts)::value_type
    StrStrPair;  // used to specify that such pairs shouldn't be displayed
}  // namespace ut

using namespace ut;

BOOST_TEST_DONT_PRINT_LOG_VALUE(StrStrPair)

/*
Iterating this file 2 times, with counter values from 0 to 1.
0 will be used for PreselectionByTinySyms set on false
1 will be used for PreselectionByTinySyms set on true
*/
#define BOOST_PP_ITERATION_LIMITS (0, 1)
#define BOOST_PP_FILENAME_1 "testMatch.cpp" /* __FILE__ didn't work! */
#include BOOST_PP_ITERATE()

#else  // BOOST_PP_IS_ITERATING is 1 (true) -- The rest of the file is iterated
       // twice

#if BOOST_PP_ITERATION() == 0
#define UsePreselection false
#define SuiteSuffix _noPreselection

#elif BOOST_PP_ITERATION() == 1
#undef UsePreselection
#define UsePreselection true
#undef SuiteSuffix
#define SuiteSuffix _withPreselection

#else  // BOOST_PP_ITERATION() >= 2
#undef UsePreselection
#undef SuiteSuffix
#endif  // BOOST_PP_ITERATION()

/**
Trying to identify all the glyphs from a font family based on certain match
settings. The symbols within a given font family are altered:
- changing their foreground and background to random values while ensuring a
minimal contrast
- using additive noise

Applying this test for more combinations of MatchSetting-s (msArray) and font
families (fonts), with and without the preselection mechanism enabled
(UsePreselection).

Normally, when the preselection is disabled, there are less than 3%
misidentified symbols. For non-altered symbols, there should be no such errors.

Unfortunately, the preselection enforces the identification to be performed
MOSTLY based on tiny versions of the glyphs, and the percentage of the errors
increases with the size of the symbol set. This is because an increasing number
of such tiny symbols will be hard to distinguish based on masks that became
really inadequate (for instance - a large circular mask becomes a small square
or a dot). So, BpMono Bold family with around 200 symbols generates less than
13% errors, but Courier Bold Unicode family with &gt; 2000 symbols generates a
bit less than 55% misidentifications

GENERALLY, no matter the preselection, Structural Similarity and Correlation
identify correctly more symbols than (Foreground, Background and Edge) aspects.
*/
DATA_TEST_CASE_SUFFIX(
    CheckAlteredCmap_UsingAspects_ExpectLessThan3or55PercentErrors,
    SuiteSuffix,
    // For Cartesian product (all combinations) use below '*'
    // For zip (only the combinations formed by pairs with same indices) use '^'
    boost::unit_test::data::make(msArray) * fonts,
    ms,
    font) {
  // Disabling clustering for this test,
  // as the result is highly sensitive to the accuracy of the clustering,
  // which depends on quite a number of adjustable parameters.
  // Every time one of these parameters is modified,
  // this test might cross the accepted mismatches threshold.
  AlteredCmapFixture<UsePreselection> fixt("None");  // mandatory

  const auto& [fontFamily, encoding] = font;

  ostringstream oss;
  oss << "CheckAlteredCmap_UsingAspects_ExpectLessThan3or55PercentErrors for "
      << fontFamily << '(' << encoding << ") with " << ms << ' '
      << (PreselectionByTinySyms ? "with" : "without") << " Preselection";
  string nameOfTest = oss.str();
  for (const char toReplace : string(" \t\n\\/():"))
    replace(BOUNDS(nameOfTest), toReplace, '_');
  BOOST_TEST_MESSAGE("Running " + nameOfTest);

  Settings s(ms);
  ::Controller c(s);
  IControlPanelActions& cpa = c.getControlPanelActions();
  ::MatchEngine& me = dynamic_cast< ::MatchEngine&>(c.getMatchEngine(s));

  cpa.newFontFamily(fontFamily);
  cpa.newFontEncoding(encoding);

  // default font size is 10 - normally at most 10% errors
  // c.newFontSize(40U); // apparently larger fonts produce less errors (for 40
  // rarely any)
  me.getReady();

  // Recognizing the glyphs from current cmap
  VSymData::const_iterator it = cbegin(me.symsSet), itEnd = cend(me.symsSet);
  const unsigned symsCount = (unsigned)std::distance(it, itEnd),
                 SymsBatchSz = 25U;
  vector<unique_ptr<BestMatch> > mismatches;
  const auto& assessor = me.assessor();
  for (unsigned idx = 0U; it != itEnd; ++idx, ++it) {
    const Mat& negGlyph = (*it)->getNegSym();  // byte 0..255
    Mat patchD255, blurredPatch;
    std::shared_ptr<Patch> thePatch;
    int attempts = 10;
    do {
      negGlyph.convertTo(patchD255, CV_64FC1, -1., 255.);
      alterFgBg(patchD255, (*it)->getMinVal(), (*it)->getDiffMinMax());
      addWhiteNoise(patchD255, .2, 10U);  // affected % and noise amplitude
      thePatch = std::make_shared<Patch>(patchD255);
    } while (--attempts >= 0 && !thePatch->nonUniform());

    if (!thePatch->nonUniform())
      continue;  // couldn't generate a noisy non-uniform glyph

    auto best = make_unique<BestMatch>(*thePatch);

    if (PreselectionByTinySyms) {
      Mat tinyPatchMat, blurredTinyPatch;
      resize(thePatch->getOrig(), tinyPatchMat,
             Size(TinySymsSize, TinySymsSize), 0., 0., INTER_AREA);
      resize(thePatch->getBlurred(), blurredTinyPatch,
             Size(TinySymsSize, TinySymsSize), 0., 0., INTER_AREA);
      Patch tinyPatch(tinyPatchMat, blurredTinyPatch,
                      tinyPatchMat.channels() > 1);
      if (!tinyPatch.nonUniform()) {
        tinyPatch.forceApproximation();  // forcing the approximation for the
                                         // tiny patch

        const Mat& patch2Process = Transform_BlurredPatches_InsteadOf_Originals
                                       ? blurredTinyPatch
                                       : tinyPatchMat;
        Mat grayD;
        patch2Process.clone().convertTo(grayD, CV_64FC1);
        tinyPatch.setMatrixToApprox(grayD);
      }
      BestMatch bestTiny(tinyPatch);

      // Below using 3 instead of ShortListLength, to avoid surprises caused by
      // altered configurations
      TopCandidateMatches tcm(3U);
      MatchProgressWithPreselection mpwp(tcm);
      MatchSupportWithPreselection mswp(me.cachedData, me.symsSet,
                                        me.matchAssessor, ms);

      for (unsigned si = 0U; si < symsCount; si += SymsBatchSz) {
        // Value from below is 0.8 instead of
        // AdmitOnShortListEvenForInferiorScoreFactor to avoid surprises caused
        // by altered configurations
        tcm.reset(bestTiny.setScore(best->getScore() * 0.8).getScore());
        if (me.improvesBasedOnBatch(si, min(si + SymsBatchSz, symsCount),
                                    bestTiny, mpwp)) {
          tcm.prepareReport();
          CandidatesShortList shortList;
          tcm.moveShortList(shortList);
          mswp.improvesBasedOnBatchShortList(std::move(shortList), *best);
        }
      }

    } else {  // PreselectionByTinySyms == false here
      MatchProgress dummy;
      for (unsigned si = 0U; si < symsCount; si += SymsBatchSz)
        me.improvesBasedOnBatch(si, min(si + SymsBatchSz, symsCount), *best,
                                dummy);
    }

    if (best->getSymIdx() != idx) {
      MatchParams mp;
      double score;
      ScoreThresholds scoresToBeat;
      assessor.scoresToBeat(best->getScore(), scoresToBeat);
      assessor.isBetterMatch(patchD255, **it, me.cachedData, scoresToBeat, mp,
                             score);
      cerr << "Expecting symbol index " << idx << " while approximated as "
           << best->getSymIdx() << "\nApproximation achieved score=" << fixed
           << setprecision(17) << best->getScore()
           << " while the score for the expected symbol is " << fixed
           << setprecision(17) << score << endl;
      wcerr << "Params from approximated symbol: " << **best->getParams()
            << "\nParams from expected comparison: " << mp << '\n'
            << endl;
      mismatches.push_back(std::move(best));
    }
  }

  if (PreselectionByTinySyms) {  // Preselection with tiny symbols - less
                                 // accurate approximation allowed
    // Normally, less than 55% of the altered symbols are not identified
    // correctly.
    BOOST_CHECK((double)mismatches.size() < .55 * symsCount);
  } else {  // No Preselection - much more accurate approximation expected
    // Normally, less than 3% of the altered symbols are not identified
    // correctly.
    BOOST_CHECK((double)mismatches.size() < .03 * symsCount);
  }

  if (!mismatches.empty()) {
    extern const wstring MatchParams_HEADER;
    wcerr << "The parameters were displayed in this order:\n"
          << MatchParams_HEADER << '\n'
          << endl;

    showMismatches(nameOfTest, mismatches);
  }
}

FIXTURE_TEST_SUITE_SUFFIX(MatchParamsFixt<UsePreselection>,
                          SumSquareNormMeanSdevMassCenterComputation_Tests,
                          SuiteSuffix)
// Patches have pixels with double values 0..255.
// Glyphs have pixels with double values 0..1.
// Masks have pixels with byte values 0..255.

TITLED_AUTO_TEST_CASE_(
    CheckSdevComputation_PrecededOrNotByMeanComputation_SameSdevExpected,
    SuiteSuffix) {
  std::optional<double> miu1, sdev1;

  // Check that computeSdev performs the same when preceded or not by
  // computeMean
  MatchParams::computeMean(getRandD255(), getFullUc(), miu);
  MatchParams::computeSdev(getRandD255(), getFullUc(), miu,
                           sdev);  // miu already computed

  MatchParams::computeSdev(getRandD255(), getFullUc(), miu1,
                           sdev1);  // miu1 not computed yet
  BOOST_REQUIRE(miu && sdev && miu1 && sdev1);
  BOOST_TEST(*miu == *miu1, tol(1e-4));
  BOOST_TEST(*sdev == *sdev1, tol(1e-4));
  TITLED_AUTO_TEST_CASE_END
}

TITLED_AUTO_TEST_CASE_(
    CheckMeanAndSdev_RandomPatchWithEmptyMask_ZeroMeanAndSdev,
    SuiteSuffix) {
  MatchParams::computeSdev(getRandD255(), getEmptyUc(), miu, sdev);
  BOOST_REQUIRE(miu && sdev);
  BOOST_TEST(*miu == 0., tol(1e-4));
  BOOST_TEST(*sdev == 0., tol(1e-4));
  TITLED_AUTO_TEST_CASE_END
}

TITLED_AUTO_TEST_CASE_(
    CheckMeanAndSdev_UniformPatchWithRandomMask_PixelsValueAsMeanAndZeroSdev,
    SuiteSuffix) {
  const auto valRand = randUnsignedChar();

  MatchParams::computeSdev(Mat(getSz(), getSz(), CV_64FC1, Scalar(valRand)),
                           getRandUc() != 0U, miu, sdev);
  BOOST_REQUIRE(miu && sdev);
  BOOST_TEST(*miu == (double)valRand, tol(1e-4));
  BOOST_TEST(*sdev == 0., tol(1e-4));
  TITLED_AUTO_TEST_CASE_END
}

TITLED_AUTO_TEST_CASE_(
    CheckMeanSdevMassCenter_Half0Half255AndNoMask_127dot5MeanAndSdevAndPredictedMassCenter,
    SuiteSuffix) {
  const auto valRand = randUnsignedChar();
  Mat halfD255 = getEmptyD().clone();
  halfD255.rowRange(0, getSz() / 2) = 255.;

  MatchParams::computeSdev(halfD255, getFullUc(), miu, sdev);
  BOOST_REQUIRE(miu && sdev);
  BOOST_TEST(*miu == 127.5, tol(1e-4));
  BOOST_TEST(*sdev == CachedData::MaxSdev::forFgOrBg, tol(1e-4));
  mp.computeMcPatch(halfD255, getCd());
  BOOST_REQUIRE(mp.getMcPatch());
  // mass-center = ( (sz-1)/2 ,  255*((sz/2-1)*(sz/2)/2)/(255*sz/2) =
  // (sz/2-1)/2 ) all downsized by (sz-1)
  BOOST_TEST(mp.getMcPatch()->x == .5, tol(1e-4));
  BOOST_TEST(mp.getMcPatch()->y == (getSz() / 2. - 1.) / (2. * (getSz() - 1)),
             tol(1e-4));
  TITLED_AUTO_TEST_CASE_END
}

TITLED_AUTO_TEST_CASE_(ComputeSymDensity_SuperiorHalfOfPatchFull_0dot5,
                       SuiteSuffix) {
  mp.computeSymDensity(getSdWithHorizEdgeMask());
  BOOST_REQUIRE(mp.getSymDensity());
  BOOST_TEST(*mp.getSymDensity() == 0.5, tol(1e-4));
  TITLED_AUTO_TEST_CASE_END
}

TITLED_AUTO_TEST_CASE_(
    CheckParams_UniformPatchHorizontalEdge_GlyphConvergesToPatch,
    SuiteSuffix) {
  checkParams_UniformPatch_GlyphConvergesToPatch(getSz(), getCd(),
                                                 getSdWithHorizEdgeMask());
  TITLED_AUTO_TEST_CASE_END
}

TITLED_AUTO_TEST_CASE_(
    CheckParams_UniformPatchVerticalEdge_GlyphConvergesToPatch,
    SuiteSuffix) {
  checkParams_UniformPatch_GlyphConvergesToPatch(getSz(), getCd(),
                                                 getSdWithVertEdgeMask());
  TITLED_AUTO_TEST_CASE_END
}

TITLED_AUTO_TEST_CASE_(
    CheckParams_TestedGlyphWithHorizontalEdgeIsInverseOfPatch_GlyphConvergesToPatch,
    SuiteSuffix) {
  checkParams_TestedGlyphIsInverseOfPatch_GlyphConvergesToPatch(
      getSz(), getInvHalfD255(), getCd(), getSdWithHorizEdgeMask());
  TITLED_AUTO_TEST_CASE_END
}

TITLED_AUTO_TEST_CASE_(
    CheckParams_TestedGlyphWithVerticalEdgeIsInverseOfPatch_GlyphConvergesToPatch,
    SuiteSuffix) {
  checkParams_TestedGlyphIsInverseOfPatch_GlyphConvergesToPatch(
      getSz(), getInvHalfD255().t(), getCd(), getSdWithVertEdgeMask(), true);
  TITLED_AUTO_TEST_CASE_END
}

TITLED_AUTO_TEST_CASE_(
    CheckParams_TestHalfFullGlypWithHorizontalEdgehOnDimmerPatch_GlyphLoosesContrast,
    SuiteSuffix) {
  checkParams_TestHalfFullGlyphOnDimmerPatch_GlyphLoosesContrast(
      getSz(), getEmptyD(), getCd(), getSdWithHorizEdgeMask());
  TITLED_AUTO_TEST_CASE_END
}

TITLED_AUTO_TEST_CASE_(
    CheckParams_TestHalfFullGlypWithVerticalEdgehOnDimmerPatch_GlyphLoosesContrast,
    SuiteSuffix) {
  checkParams_TestHalfFullGlyphOnDimmerPatch_GlyphLoosesContrast(
      getSz(), getEmptyD(), getCd(), getSdWithVertEdgeMask(), true);
  TITLED_AUTO_TEST_CASE_END
}

TITLED_AUTO_TEST_CASE_(CheckParams_RowValuesSameAsRowIndices_PredictedParams,
                       SuiteSuffix) {
  // Testing on a patch with uniform rows of values gradually growing from 0
  // to sz-1
  Mat szBandsD255 = getEmptyD().clone();
  for (unsigned i = 0U; i < getSz(); ++i)
    szBandsD255.row(i) = i;
  const double expectedFgAndSdevHorEdge = (getSz() - 2) / 4.,
               expectedBg = (3. * getSz() - 2.) / 4.,
               expectedMiu = (getSz() - 1U) / 2.;
  double expectedSdev = 0., expectedNormMiu0 = 0., expectedCorrNominator = 0.;
  for (unsigned i = 0U; i < getSz() / 2; ++i) {
    double diff = i - expectedFgAndSdevHorEdge;
    expectedSdev += diff * diff;
    diff = i - expectedMiu;
    expectedNormMiu0 += diff * diff;

    // Avoid an unsigned parenthesis result below (overflow if not carefully)!
    expectedCorrNominator +=
        0.5 * (i - (getSz() - i - 1.));  // i*0.5 + (n-i-1)*(-0.5)
  }
  expectedSdev = sqrt(expectedSdev / (getSz() / 2));
  const double expectedSdevVertSym =
      sqrt(4. * expectedNormMiu0 / (2. * getSz()));
  expectedNormMiu0 = sqrt(2. * getSz() * expectedNormMiu0);
  expectedCorrNominator *= getSz();
  Mat expectedPatchApprox(getSz(), getSz(), CV_64FC1, Scalar(expectedBg));
  expectedPatchApprox.rowRange(0, getSz() / 2) = expectedFgAndSdevHorEdge;
  mp.computePatchSum(szBandsD255);
  BOOST_REQUIRE(mp.getPatchSum());
  BOOST_TEST(*mp.getPatchSum() == (getSz() - 1.) * getSzSq() / 2., tol(1e-4));
  mp.computePatchSq(szBandsD255);
  BOOST_REQUIRE(mp.getPatchSq());
  minMaxIdx(mp.getPatchSq().value() - szBandsD255.mul(szBandsD255), &minV,
            &maxV);
  BOOST_TEST(minV == 0., tol(1e-4));
  BOOST_TEST(maxV == 0., tol(1e-4));
  mp.computeNormPatchMinMiu(szBandsD255, cd);
  BOOST_REQUIRE(mp.getNormPatchMinMiu());
  BOOST_TEST(*mp.getNormPatchMinMiu() == expectedNormMiu0, tol(1e-4));
  mp.computeAbsCorr(szBandsD255, getSdWithHorizEdgeMask(), cd);
  BOOST_REQUIRE(mp.getAbsCorr());
  BOOST_TEST(
      *mp.getAbsCorr() ==
          abs(expectedCorrNominator) /
              (expectedNormMiu0 * getSdWithHorizEdgeMask().getNormSymMiu0()),
      tol(1e-4));
  mp.computePatchApprox(szBandsD255, getSdWithHorizEdgeMask());
  BOOST_REQUIRE(mp.getPatchApprox());
  minMaxIdx(mp.getPatchApprox().value() - expectedPatchApprox, &minV, &maxV);
  BOOST_TEST(minV == 0., tol(1e-4));
  BOOST_TEST(maxV == 0., tol(1e-4));
  mp.computeMcPatch(szBandsD255, getCd());
  BOOST_REQUIRE(mp.getMcPatch());
  BOOST_TEST(mp.getMcPatch()->x == .5, tol(1e-4));
  BOOST_TEST(mp.getMcPatch()->y == (2. * getSz() - 1.) / (3. * (getSz() - 1.)),
             tol(1e-4));
  mp.computeFg(szBandsD255, getSdWithHorizEdgeMask());
  BOOST_REQUIRE(mp.getFg());
  BOOST_TEST(*mp.getFg() == expectedFgAndSdevHorEdge, tol(1e-4));
  mp.computeBg(szBandsD255, getSdWithHorizEdgeMask());
  BOOST_REQUIRE(mp.getBg());
  BOOST_TEST(*mp.getBg() == expectedBg, tol(1e-4));
  mp.computeSdevFg(szBandsD255, getSdWithHorizEdgeMask());
  BOOST_REQUIRE(mp.getSdevFg());
  BOOST_TEST(*mp.getSdevFg() == expectedSdev, tol(1e-4));
  mp.computeSdevBg(szBandsD255, getSdWithHorizEdgeMask());
  BOOST_REQUIRE(mp.getSdevBg());
  BOOST_TEST(*mp.getSdevBg() == expectedSdev, tol(1e-4));
  mp.computeMcPatchApprox(szBandsD255, getSdWithHorizEdgeMask(), getCd());
  BOOST_REQUIRE(mp.getMcPatchApprox());
  BOOST_TEST(mp.getMcPatchApprox()->x == .5, tol(1e-4));
  BOOST_TEST(mp.getMcPatchApprox()->y ==
                 (*mp.getFg() * *mp.getFg() + *mp.getBg() * *mp.getBg()) /
                     pow(getSz() - 1U, 2),
             tol(1e-4));
  mp.computeSdevEdge(szBandsD255, getSdWithHorizEdgeMask());
  BOOST_REQUIRE(mp.getSdevEdge());
  BOOST_TEST(*mp.getSdevEdge() == expectedFgAndSdevHorEdge, tol(1e-4));

  // Do some tests also for a glyph with a vertical edge (keep the patch with
  // horizontal bands)
  mp.reset();
  mp.computeAbsCorr(szBandsD255, getSdWithVertEdgeMask(), cd);
  BOOST_REQUIRE(mp.getAbsCorr());
  BOOST_TEST(*mp.getAbsCorr() == 0., tol(1e-4));
  mp.computePatchApprox(szBandsD255, getSdWithVertEdgeMask());
  BOOST_REQUIRE(mp.getPatchApprox());
  minMaxIdx(mp.getPatchApprox().value(), &minV, &maxV);
  BOOST_TEST(minV == expectedMiu, tol(1e-4));
  BOOST_TEST(maxV == expectedMiu, tol(1e-4));
  mp.computeSdevEdge(szBandsD255, getSdWithVertEdgeMask());
  BOOST_REQUIRE(mp.getSdevEdge());
  BOOST_TEST(*mp.getSdevEdge() == expectedSdevVertSym, tol(1e-4));
  TITLED_AUTO_TEST_CASE_END
}
BOOST_AUTO_TEST_SUITE_END()  // CheckMatchParams

FIXTURE_TEST_SUITE_SUFFIX(MatchAspectsFixt<UsePreselection>,
                          MatchAspects_Tests,
                          SuiteSuffix)
TITLED_AUTO_TEST_CASE_(
    CheckCorrelationAspect_UniformPatchAndDiagGlyph_GlyphBecomesPatch,
    SuiteSuffix) {
  ms.set_kCorrel(1.);
  cd.useNewSymSize(getSz());

  const CorrelationAspect corr(ms);
  const auto valRand = randUnsignedChar(1U);

  // Using a symbol with a diagonal foreground mask
  const Mat diagFgMask = Mat::diag(Mat(1, getSz(), CV_8UC1, Scalar(255U))),
            diagSymD1 = Mat::diag(Mat(1, getSz(), CV_64FC1, Scalar(1.)));
  Mat allButMainDiagBgMask = Mat(getSz(), getSz(), CV_8UC1, Scalar(255U));
  allButMainDiagBgMask.diag() = 0U;
  const unsigned cnz = getArea() - getSz();
  BOOST_REQUIRE((unsigned)countNonZero(allButMainDiagBgMask) == cnz);

  SymData sd(
      NOT_RELEVANT_UL,     // symbol code (not relevant here)
      NOT_RELEVANT_SZ,     // symbol index (not relevant here)
      NOT_RELEVANT_D,      // min in range 0..1 (not relevant here)
      1.,                  // diff between min..max, each in range 0..1
      NOT_RELEVANT_D,      // avgPixVal (not relevant here)
      sqrt(getSz() - 1U),  // normSymMiu0
      NOT_RELEVANT_POINT,  // mc sym for original fg & bg (not relevant here)
      SymData::IdxMatMap{
          {(int)ISymData::MaskType::Fg, diagFgMask},
          {(int)ISymData::MaskType::Bg, allButMainDiagBgMask},
          {(int)ISymData::MaskType::GroundedSym,
           diagSymD1},  // same as the glyph
          {(int)ISymData::MaskType::BlurredGrSym, NOT_RELEVANT_MAT},
          {(int)ISymData::MaskType::VarianceGrSym, NOT_RELEVANT_MAT}},
      -diagSymD1 + 1.,            // negSym
      diagSymD1 - 1. / getSz());  // symMiu0

  // Testing on an uniform patch
  patchD255 = Mat(getSz(), getSz(), CV_64FC1, Scalar((double)valRand));
  res = corr.assessMatch(patchD255, sd, cd, mp);
  BOOST_REQUIRE(mp.getPatchSq() && mp.getAbsCorr());
  minMaxIdx(mp.getPatchSq().value() - patchD255.mul(patchD255), &minV, &maxV);
  BOOST_TEST(minV == 0., tol(1e-4));
  BOOST_TEST(maxV == 0., tol(1e-4));
  BOOST_TEST(*mp.getAbsCorr() == 1., tol(1e-4));
  BOOST_TEST(res == 1., tol(1e-4));
  TITLED_AUTO_TEST_CASE_END
}

TITLED_AUTO_TEST_CASE_(
    CheckCorrelationAspect_PatchIsDimmedGlyph_GlyphBecomesPatch,
    SuiteSuffix) {
  ms.set_kCorrel(1.);
  cd.useNewSymSize(getSz());

  const CorrelationAspect corr(ms);
  const auto valRand = randUnsignedChar(1U);

  // Using a symbol with a diagonal foreground mask
  const Mat diagFgMask = Mat::diag(Mat(1, getSz(), CV_8UC1, Scalar(255U))),
            diagSymD1 = Mat::diag(Mat(1, getSz(), CV_64FC1, Scalar(1.)));
  Mat allButMainDiagBgMask = Mat(getSz(), getSz(), CV_8UC1, Scalar(255U));
  allButMainDiagBgMask.diag() = 0U;
  const unsigned cnz = getArea() - getSz();
  BOOST_REQUIRE((unsigned)countNonZero(allButMainDiagBgMask) == cnz);

  SymData sd(
      NOT_RELEVANT_UL,     // symbol code (not relevant here)
      NOT_RELEVANT_SZ,     // symbol index (not relevant here)
      NOT_RELEVANT_D,      // min in range 0..1 (not relevant here)
      1.,                  // diff between min..max, each in range 0..1
      NOT_RELEVANT_D,      // avgPixVal (not relevant here)
      sqrt(getSz() - 1U),  // normSymMiu0
      NOT_RELEVANT_POINT,  // mc sym for original fg & bg (not relevant here)
      SymData::IdxMatMap{
          {(int)ISymData::MaskType::Fg, diagFgMask},
          {(int)ISymData::MaskType::Bg, allButMainDiagBgMask},
          {(int)ISymData::MaskType::GroundedSym,
           diagSymD1},  // same as the glyph
          {(int)ISymData::MaskType::BlurredGrSym, NOT_RELEVANT_MAT},
          {(int)ISymData::MaskType::VarianceGrSym, NOT_RELEVANT_MAT}},
      -diagSymD1 + 1.,            // negSym
      diagSymD1 - 1. / getSz());  // symMiu0

  // Testing on a patch with the background = valRand and diagonal = valRand's
  // complementary value
  const double complVal = (double)((128U + valRand) & 0xFFU);
  patchD255 = Mat(getSz(), getSz(), CV_64FC1, Scalar((double)valRand));
  patchD255.diag() = complVal;
  res = corr.assessMatch(patchD255, sd, cd, mp);
  BOOST_REQUIRE(mp.getPatchSq() && mp.getAbsCorr());
  minMaxIdx(mp.getPatchSq().value() - patchD255.mul(patchD255), &minV, &maxV);
  BOOST_TEST(minV == 0., tol(1e-4));
  BOOST_TEST(maxV == 0., tol(1e-4));
  BOOST_TEST(*mp.getAbsCorr() == 1., tol(1e-4));
  BOOST_TEST(res == 1., tol(1e-4));
  TITLED_AUTO_TEST_CASE_END
}

TITLED_AUTO_TEST_CASE_(
    CheckStructuralSimilarity_UniformPatchAndDiagGlyph_GlyphBecomesPatch,
    SuiteSuffix) {
  ms.set_kSsim(1.);
  const StructuralSimilarity strSim(ms);
  const auto valRand = randUnsignedChar(1U);

  // Using a symbol with a diagonal foreground mask
  const Mat diagFgMask = Mat::diag(Mat(1, getSz(), CV_8UC1, Scalar(255U))),
            diagSymD1 = Mat::diag(Mat(1, getSz(), CV_64FC1, Scalar(1.))),
            diagSymD255 = Mat::diag(Mat(1, getSz(), CV_64FC1, Scalar(255.)));
  Mat blurOfGroundedGlyph, varOfGroundedGlyph,
      allButMainDiagBgMask = Mat(getSz(), getSz(), CV_8UC1, Scalar(255U));
  allButMainDiagBgMask.diag() = 0U;
  const unsigned cnz = getArea() - getSz();
  BOOST_REQUIRE((unsigned)countNonZero(allButMainDiagBgMask) == cnz);
  blurSupport.process(diagSymD1, blurOfGroundedGlyph, PreselectionByTinySyms);
  blurSupport.process(diagSymD1.mul(diagSymD1), varOfGroundedGlyph,
                      PreselectionByTinySyms);
  varOfGroundedGlyph -= blurOfGroundedGlyph.mul(blurOfGroundedGlyph);

  SymData sd(
      NOT_RELEVANT_UL,     // symbol code (not relevant here)
      NOT_RELEVANT_SZ,     // symbol index (not relevant here)
      NOT_RELEVANT_D,      // min in range 0..1 (not relevant here)
      1.,                  // diff between min..max, each in range 0..1
      NOT_RELEVANT_D,      // avgPixVal (not relevant here)
      NOT_RELEVANT_D,      // normSymMiu0 (not relevant here)
      NOT_RELEVANT_POINT,  // mc sym for original fg & bg (not relevant here)
      SymData::IdxMatMap{
          {(int)ISymData::MaskType::Fg, diagFgMask},
          {(int)ISymData::MaskType::Bg, allButMainDiagBgMask},
          {(int)ISymData::MaskType::GroundedSym,
           diagSymD1},  // same as the glyph
          {(int)ISymData::MaskType::BlurredGrSym, blurOfGroundedGlyph},
          {(int)ISymData::MaskType::VarianceGrSym, varOfGroundedGlyph}});

  // Testing on an uniform patch
  patchD255 = Mat(getSz(), getSz(), CV_64FC1, Scalar((double)valRand));
  res = strSim.assessMatch(patchD255, sd, cd, mp);
  BOOST_REQUIRE(mp.getFg() && mp.getBg() && mp.getPatchApprox() &&
                mp.getSsim() && mp.getPatchSq() && mp.getBlurredPatch() &&
                mp.getBlurredPatchSq() && mp.getVariancePatch());
  BOOST_TEST(*mp.getFg() == (double)valRand, tol(1e-4));
  BOOST_TEST(*mp.getBg() == (double)valRand, tol(1e-4));
  minMaxIdx(mp.getPatchApprox().value(), &minV, &maxV);
  BOOST_TEST(minV == (double)valRand, tol(1e-4));
  BOOST_TEST(maxV == (double)valRand, tol(1e-4));
  minMaxIdx(mp.getPatchSq().value() - patchD255.mul(patchD255), &minV, &maxV);
  BOOST_TEST(minV == 0., tol(1e-4));
  BOOST_TEST(maxV == 0., tol(1e-4));
  BOOST_TEST(*mp.getSsim() == 1., tol(1e-4));
  BOOST_TEST(res == 1., tol(1e-4));
  TITLED_AUTO_TEST_CASE_END
}

TITLED_AUTO_TEST_CASE_(
    CheckStructuralSimilarity_PatchIsDimmedGlyph_GlyphBecomesPatch,
    SuiteSuffix) {
  ms.set_kSsim(1.);
  const StructuralSimilarity strSim(ms);
  const auto valRand = randUnsignedChar(1U);

  // Using a symbol with a diagonal foreground mask
  const Mat diagFgMask = Mat::diag(Mat(1, getSz(), CV_8UC1, Scalar(255U))),
            diagSymD1 = Mat::diag(Mat(1, getSz(), CV_64FC1, Scalar(1.))),
            diagSymD255 = Mat::diag(Mat(1, getSz(), CV_64FC1, Scalar(255.)));
  Mat blurOfGroundedGlyph, varOfGroundedGlyph,
      allButMainDiagBgMask = Mat(getSz(), getSz(), CV_8UC1, Scalar(255U));
  allButMainDiagBgMask.diag() = 0U;
  const unsigned cnz = getArea() - getSz();
  BOOST_REQUIRE((unsigned)countNonZero(allButMainDiagBgMask) == cnz);
  blurSupport.process(diagSymD1, blurOfGroundedGlyph, PreselectionByTinySyms);
  blurSupport.process(diagSymD1.mul(diagSymD1), varOfGroundedGlyph,
                      PreselectionByTinySyms);
  varOfGroundedGlyph -= blurOfGroundedGlyph.mul(blurOfGroundedGlyph);

  SymData sd(
      NOT_RELEVANT_UL,     // symbol code (not relevant here)
      NOT_RELEVANT_SZ,     // symbol index (not relevant here)
      NOT_RELEVANT_D,      // min in range 0..1 (not relevant here)
      1.,                  // diff between min..max, each in range 0..1
      NOT_RELEVANT_D,      // avgPixVal (not relevant here)
      NOT_RELEVANT_D,      // normSymMiu0 (not relevant here)
      NOT_RELEVANT_POINT,  // mc sym for original fg & bg (not relevant here)
      SymData::IdxMatMap{
          {(int)ISymData::MaskType::Fg, diagFgMask},
          {(int)ISymData::MaskType::Bg, allButMainDiagBgMask},
          {(int)ISymData::MaskType::GroundedSym,
           diagSymD1},  // same as the glyph
          {(int)ISymData::MaskType::BlurredGrSym, blurOfGroundedGlyph},
          {(int)ISymData::MaskType::VarianceGrSym, varOfGroundedGlyph}});

  // Testing on a patch with the background = valRand and diagonal = valRand's
  // complementary value
  const double complVal = (double)((128U + valRand) & 0xFFU);
  patchD255 = Mat(getSz(), getSz(), CV_64FC1, Scalar((double)valRand));
  patchD255.diag() = complVal;
  res = strSim.assessMatch(patchD255, sd, cd, mp);
  BOOST_REQUIRE(mp.getFg() && mp.getBg() && mp.getPatchApprox() &&
                mp.getSsim() && mp.getPatchSq() && mp.getBlurredPatch() &&
                mp.getBlurredPatchSq() && mp.getVariancePatch());
  minMaxIdx(mp.getPatchApprox().value() - patchD255, &minV, &maxV);
  BOOST_TEST(minV == 0., tol(1e-4));
  BOOST_TEST(maxV == 0., tol(1e-4));
  BOOST_TEST(*mp.getFg() == complVal, tol(1e-4));
  BOOST_TEST(*mp.getBg() == (double)valRand, tol(1e-4));
  minMaxIdx(mp.getPatchSq().value() - patchD255.mul(patchD255), &minV, &maxV);
  BOOST_TEST(minV == 0., tol(1e-4));
  BOOST_TEST(maxV == 0., tol(1e-4));
  BOOST_TEST(*mp.getSsim() == 1., tol(1e-4));
  BOOST_TEST(res == 1., tol(1e-4));
  TITLED_AUTO_TEST_CASE_END
}

TITLED_AUTO_TEST_CASE_(
    CheckFgAspect_UniformPatchAndDiagGlyph_CompleteMatchAndFgBecomesPatchValue,
    SuiteSuffix) {
  ms.set_kSdevFg(1.);
  const FgMatch fm(ms);
  const auto valRand = randUnsignedChar(1U);

  // Using a symbol with a diagonal foreground mask
  const Mat diagFgMask = Mat::diag(Mat(1, getSz(), CV_8UC1, Scalar(255U)));

  SymData sd(
      NOT_RELEVANT_UL,  // symbol code (not relevant here)
      NOT_RELEVANT_SZ,  // symbol index (not relevant here)
      NOT_RELEVANT_D,
      NOT_RELEVANT_D,      // min and diff between min..max, each in range 0..1
                           // (not relevant here)
      NOT_RELEVANT_D,      // avgPixVal (not relevant here)
      NOT_RELEVANT_D,      // normSymMiu0 (not relevant here)
      NOT_RELEVANT_POINT,  // mc sym for original fg & bg (not relevant here)
      SymData::IdxMatMap{{(int)ISymData::MaskType::Fg, diagFgMask}});

  // Testing on a uniform patch
  patchD255 = Mat(getSz(), getSz(), CV_64FC1, Scalar((double)valRand));
  res = fm.assessMatch(patchD255, sd, cd, mp);
  BOOST_REQUIRE(mp.getFg() && mp.getSdevFg());
  BOOST_TEST(*mp.getFg() == (double)valRand, tol(1e-4));
  BOOST_TEST(*mp.getSdevFg() == 0., tol(1e-4));
  BOOST_TEST(res == 1., tol(1e-4));
  TITLED_AUTO_TEST_CASE_END
}

TITLED_AUTO_TEST_CASE_(
    CheckFgAspect_DiagGlyphAndPatchWithUpperHalfEmptyAndUniformLowerHalf_ImperfectMatch,
    SuiteSuffix) {
  ms.set_kSdevFg(1.);
  const FgMatch fm(ms);
  const auto valRand = randUnsignedChar(1U);

  // Using a symbol with a diagonal foreground mask
  const Mat diagFgMask = Mat::diag(Mat(1, getSz(), CV_8UC1, Scalar(255U)));

  SymData sd(
      NOT_RELEVANT_UL,  // symbol code (not relevant here)
      NOT_RELEVANT_SZ,  // symbol index (not relevant here)
      NOT_RELEVANT_D,
      NOT_RELEVANT_D,      // min and diff between min..max, each in range 0..1
                           // (not relevant here)
      NOT_RELEVANT_D,      // avgPixVal (not relevant here)
      NOT_RELEVANT_D,      // normSymMiu0 (not relevant here)
      NOT_RELEVANT_POINT,  // mc sym for original fg & bg (not relevant here)
      SymData::IdxMatMap{{(int)ISymData::MaskType::Fg, diagFgMask}});

  // Testing on a patch with upper half empty and an uniform lower half
  // (valRand)
  patchD255 = Mat(getSz(), getSz(), CV_64FC1, Scalar((double)valRand));
  patchD255.rowRange(0, getSz() / 2) = 0.;
  res = fm.assessMatch(patchD255, sd, cd, mp);
  BOOST_REQUIRE(mp.getFg() && mp.getSdevFg());
  BOOST_TEST(*mp.getFg() == valRand / 2., tol(1e-4));
  BOOST_TEST(*mp.getSdevFg() == valRand / 2., tol(1e-4));
  BOOST_TEST(res == 1. - valRand / 255., tol(1e-4));
  TITLED_AUTO_TEST_CASE_END
}

TITLED_AUTO_TEST_CASE_(
    CheckFgAspect_DiagGlyphAndPachRowValuesSameAsRowIndices_ImperfectMatch,
    SuiteSuffix) {
  ms.set_kSdevFg(1.);
  const FgMatch fm(ms);
  const auto valRand = randUnsignedChar(1U);

  // Using a symbol with a diagonal foreground mask
  const Mat diagFgMask = Mat::diag(Mat(1, getSz(), CV_8UC1, Scalar(255U)));

  SymData sd(
      NOT_RELEVANT_UL,  // symbol code (not relevant here)
      NOT_RELEVANT_SZ,  // symbol index (not relevant here)
      NOT_RELEVANT_D,
      NOT_RELEVANT_D,      // min and diff between min..max, each in range 0..1
                           // (not relevant here)
      NOT_RELEVANT_D,      // avgPixVal (not relevant here)
      NOT_RELEVANT_D,      // normSymMiu0 (not relevant here)
      NOT_RELEVANT_POINT,  // mc sym for original fg & bg (not relevant here)
      SymData::IdxMatMap{{(int)ISymData::MaskType::Fg, diagFgMask}});

  double expectedMiu = (getSz() - 1U) / 2., expectedSdev = 0., aux;
  patchD255 = Mat::zeros(getSz(), getSz(), CV_64FC1);
  for (unsigned i = 0U; i < getSz(); ++i) {
    patchD255.row(i) = (double)i;
    aux = i - expectedMiu;
    expectedSdev += aux * aux;
  }
  expectedSdev = sqrt(expectedSdev / getSz());
  res = fm.assessMatch(patchD255, sd, cd, mp);
  BOOST_REQUIRE(mp.getFg() && mp.getSdevFg());
  BOOST_TEST(*mp.getFg() == expectedMiu, tol(1e-4));
  BOOST_TEST(*mp.getSdevFg() == expectedSdev, tol(1e-4));
  BOOST_TEST(res == 1. - expectedSdev / 127.5, tol(1e-4));
  TITLED_AUTO_TEST_CASE_END
}

TITLED_AUTO_TEST_CASE_(CheckEdgeAspect_UniformPatch_PerfectMatch, SuiteSuffix) {
  ms.set_kSdevEdge(1);
  const EdgeMatch em(ms);
  const auto valRand = randUnsignedChar(1U);

  // Using a symbol with a diagonal foreground mask
  const Mat diagFgMask = Mat::diag(Mat(1, getSz(), CV_8UC1, Scalar(255U)));

  // Using a symbol with an edge mask formed by 2 neighbor diagonals of the
  // main diagonal
  Mat sideDiagsEdgeMask = Mat::zeros(getSz(), getSz(), CV_8UC1);
  sideDiagsEdgeMask.diag(1) = 255U;   // 2nd diagonal lower half
  sideDiagsEdgeMask.diag(-1) = 255U;  // 2nd diagonal upper half
  const unsigned cnzEdge = 2U * (getSz() - 1U);
  BOOST_REQUIRE((unsigned)countNonZero(sideDiagsEdgeMask) == cnzEdge);

  // Using a symbol with a background mask full except the main 3 diagonals
  Mat allBut3DiagsBgMask = Mat(getSz(), getSz(), CV_8UC1, Scalar(255U));
  allBut3DiagsBgMask.diag() = 0U;
  allBut3DiagsBgMask.diag(1) = 0U;   // 2nd diagonal lower half
  allBut3DiagsBgMask.diag(-1) = 0U;  // 2nd diagonal upper half
  const unsigned cnzBg = getArea() - 3U * getSz() + 2U;
  BOOST_REQUIRE((unsigned)countNonZero(allBut3DiagsBgMask) == cnzBg);

  Mat groundedGlyph = Mat::zeros(getSz(), getSz(), CV_8UC1);  // bg will stay 0
  const unsigned char edgeLevel = 125U,
                      maxGlyph = edgeLevel << 1;            // 250
  add(groundedGlyph, maxGlyph, groundedGlyph, diagFgMask);  // set fg
  add(groundedGlyph, edgeLevel, groundedGlyph,
      sideDiagsEdgeMask);  // set edges
  groundedGlyph.convertTo(groundedGlyph, CV_64FC1, 1. / 255);

  SymData sd(
      NOT_RELEVANT_UL,  // symbol code (not relevant here)
      NOT_RELEVANT_SZ,  // symbol index (not relevant here)
      NOT_RELEVANT_D,   // min brightness value 0..1 range (not relevant here)
      maxGlyph / 255.,  // diff between min..max, each in range 0..1
      NOT_RELEVANT_D,   // avgPixVal (not relevant here)
      NOT_RELEVANT_D,   // normSymMiu0 (not relevant here)
      NOT_RELEVANT_POINT,  // mc sym for original fg & bg (not relevant here)
      SymData::IdxMatMap{
          {(int)ISymData::MaskType::Fg, diagFgMask},
          {(int)ISymData::MaskType::Bg, allBut3DiagsBgMask},
          {(int)ISymData::MaskType::Edge, sideDiagsEdgeMask},
          {(int)ISymData::MaskType::GroundedSym, groundedGlyph}});

  // Testing on a uniform patch
  patchD255 = Mat(getSz(), getSz(), CV_64FC1, Scalar((double)valRand));
  res = em.assessMatch(patchD255, sd, cd, mp);
  BOOST_REQUIRE(mp.getFg() && mp.getBg() && mp.getSdevEdge());
  BOOST_TEST(*mp.getFg() == (double)valRand, tol(1e-4));
  BOOST_TEST(*mp.getBg() == (double)valRand, tol(1e-4));
  BOOST_TEST(*mp.getSdevEdge() == 0., tol(1e-4));
  BOOST_TEST(res == 1., tol(1e-4));
  TITLED_AUTO_TEST_CASE_END
}

TITLED_AUTO_TEST_CASE_(
    CheckEdgeAspect_EdgyDiagGlyphAndPatchWithUpperHalfEmptyAndUniformLowerPart_ImperfectMatch,
    SuiteSuffix) {
  ms.set_kSdevEdge(1);
  const EdgeMatch em(ms);
  const auto valRand = randUnsignedChar(1U);

  // Using a symbol with a diagonal foreground mask
  const Mat diagFgMask = Mat::diag(Mat(1, getSz(), CV_8UC1, Scalar(255U)));

  // Using a symbol with an edge mask formed by 2 neighbor diagonals of the
  // main diagonal
  Mat sideDiagsEdgeMask = Mat::zeros(getSz(), getSz(), CV_8UC1);
  sideDiagsEdgeMask.diag(1) = 255U;   // 2nd diagonal lower half
  sideDiagsEdgeMask.diag(-1) = 255U;  // 2nd diagonal upper half
  const unsigned cnzEdge = 2U * (getSz() - 1U);
  BOOST_REQUIRE((unsigned)countNonZero(sideDiagsEdgeMask) == cnzEdge);

  // Using a symbol with a background mask full except the main 3 diagonals
  Mat allBut3DiagsBgMask = Mat(getSz(), getSz(), CV_8UC1, Scalar(255U));
  allBut3DiagsBgMask.diag() = 0U;
  allBut3DiagsBgMask.diag(1) = 0U;   // 2nd diagonal lower half
  allBut3DiagsBgMask.diag(-1) = 0U;  // 2nd diagonal upper half
  const unsigned cnzBg = getArea() - 3U * getSz() + 2U;
  BOOST_REQUIRE((unsigned)countNonZero(allBut3DiagsBgMask) == cnzBg);

  Mat groundedGlyph = Mat::zeros(getSz(), getSz(), CV_8UC1);  // bg will stay 0
  const unsigned char edgeLevel = 125U,
                      maxGlyph = edgeLevel << 1;            // 250
  add(groundedGlyph, maxGlyph, groundedGlyph, diagFgMask);  // set fg
  add(groundedGlyph, edgeLevel, groundedGlyph,
      sideDiagsEdgeMask);  // set edges
  groundedGlyph.convertTo(groundedGlyph, CV_64FC1, 1. / 255);

  SymData sd(
      NOT_RELEVANT_UL,  // symbol code (not relevant here)
      NOT_RELEVANT_SZ,  // symbol index (not relevant here)
      NOT_RELEVANT_D,   // min brightness value 0..1 range (not relevant here)
      maxGlyph / 255.,  // diff between min..max, each in range 0..1
      NOT_RELEVANT_D,   // avgPixVal (not relevant here)
      NOT_RELEVANT_D,   // normSymMiu0 (not relevant here)
      NOT_RELEVANT_POINT,  // mc sym for original fg & bg (not relevant here)
      SymData::IdxMatMap{
          {(int)ISymData::MaskType::Fg, diagFgMask},
          {(int)ISymData::MaskType::Bg, allBut3DiagsBgMask},
          {(int)ISymData::MaskType::Edge, sideDiagsEdgeMask},
          {(int)ISymData::MaskType::GroundedSym, groundedGlyph}});

  // Testing on a patch with upper half empty and an uniform lower half
  // (valRand)
  patchD255 = Mat(getSz(), getSz(), CV_64FC1, Scalar((double)valRand));
  patchD255.rowRange(0, getSz() / 2) = 0.;
  res = em.assessMatch(patchD255, sd, cd, mp);
  BOOST_REQUIRE(mp.getFg() && mp.getBg() && mp.getSdevEdge());
  BOOST_TEST(*mp.getFg() == valRand / 2., tol(1e-4));
  BOOST_TEST(*mp.getBg() == valRand / 2., tol(1e-4));
  BOOST_TEST(*mp.getSdevEdge() == valRand / 2., tol(1e-4));
  BOOST_TEST(res == 1. - valRand / 510., tol(1e-4));
  TITLED_AUTO_TEST_CASE_END
}

TITLED_AUTO_TEST_CASE_(
    CheckEdgeAspect_EdgyDiagGlyphAndPatchRowValuesSameAsRowIndices_ImperfectMatch,
    SuiteSuffix) {
  ms.set_kSdevEdge(1);
  const EdgeMatch em(ms);
  const auto valRand = randUnsignedChar(1U);

  // Using a symbol with a diagonal foreground mask
  const Mat diagFgMask = Mat::diag(Mat(1, getSz(), CV_8UC1, Scalar(255U)));

  // Using a symbol with an edge mask formed by 2 neighbor diagonals of the
  // main diagonal
  Mat sideDiagsEdgeMask = Mat::zeros(getSz(), getSz(), CV_8UC1);
  sideDiagsEdgeMask.diag(1) = 255U;   // 2nd diagonal lower half
  sideDiagsEdgeMask.diag(-1) = 255U;  // 2nd diagonal upper half
  const unsigned cnzEdge = 2U * (getSz() - 1U);
  BOOST_REQUIRE((unsigned)countNonZero(sideDiagsEdgeMask) == cnzEdge);

  // Using a symbol with a background mask full except the main 3 diagonals
  Mat allBut3DiagsBgMask = Mat(getSz(), getSz(), CV_8UC1, Scalar(255U));
  allBut3DiagsBgMask.diag() = 0U;
  allBut3DiagsBgMask.diag(1) = 0U;   // 2nd diagonal lower half
  allBut3DiagsBgMask.diag(-1) = 0U;  // 2nd diagonal upper half
  const unsigned cnzBg = getArea() - 3U * getSz() + 2U;
  BOOST_REQUIRE((unsigned)countNonZero(allBut3DiagsBgMask) == cnzBg);

  Mat groundedGlyph = Mat::zeros(getSz(), getSz(), CV_8UC1);  // bg will stay 0
  const unsigned char edgeLevel = 125U,
                      maxGlyph = edgeLevel << 1;            // 250
  add(groundedGlyph, maxGlyph, groundedGlyph, diagFgMask);  // set fg
  add(groundedGlyph, edgeLevel, groundedGlyph,
      sideDiagsEdgeMask);  // set edges
  groundedGlyph.convertTo(groundedGlyph, CV_64FC1, 1. / 255);

  SymData sd(
      NOT_RELEVANT_UL,  // symbol code (not relevant here)
      NOT_RELEVANT_SZ,  // symbol index (not relevant here)
      NOT_RELEVANT_D,   // min brightness value 0..1 range (not relevant here)
      maxGlyph / 255.,  // diff between min..max, each in range 0..1
      NOT_RELEVANT_D,   // avgPixVal (not relevant here)
      NOT_RELEVANT_D,   // normSymMiu0 (not relevant here)
      NOT_RELEVANT_POINT,  // mc sym for original fg & bg (not relevant here)
      SymData::IdxMatMap{
          {(int)ISymData::MaskType::Fg, diagFgMask},
          {(int)ISymData::MaskType::Bg, allBut3DiagsBgMask},
          {(int)ISymData::MaskType::Edge, sideDiagsEdgeMask},
          {(int)ISymData::MaskType::GroundedSym, groundedGlyph}});

  // Testing on a patch with uniform rows, but gradually brighter, from top to
  // bottom
  patchD255 = Mat::zeros(getSz(), getSz(), CV_64FC1);
  double expectedMiu = (getSz() - 1U) / 2., expectedSdev = 0., aux;
  for (unsigned i = 0U; i < getSz(); ++i) {
    patchD255.row(i) = (double)i;
    aux = i - expectedMiu;
    expectedSdev += aux * aux;
  }
  expectedSdev *= 2.;
  expectedSdev -= 2 * expectedMiu * expectedMiu;
  expectedSdev = sqrt(expectedSdev / cnzEdge);
  res = em.assessMatch(patchD255, sd, cd, mp);
  BOOST_REQUIRE(mp.getFg() && mp.getBg() && mp.getSdevEdge());
  BOOST_TEST(*mp.getFg() == expectedMiu, tol(1e-4));
  BOOST_TEST(*mp.getBg() == expectedMiu, tol(1e-4));
  BOOST_TEST(*mp.getSdevEdge() == expectedSdev, tol(1e-4));
  BOOST_TEST(res == 1. - expectedSdev / 255, tol(1e-4));
  TITLED_AUTO_TEST_CASE_END
}

TITLED_AUTO_TEST_CASE_(
    CheckEdgeAspect_EdgyDiagGlyphAndUniformLowerTriangularPatch_ImperfectMatch,
    SuiteSuffix) {
  ms.set_kSdevEdge(1);
  const EdgeMatch em(ms);
  const auto valRand = randUnsignedChar(1U);

  // Using a symbol with a diagonal foreground mask
  const Mat diagFgMask = Mat::diag(Mat(1, getSz(), CV_8UC1, Scalar(255U)));

  // Using a symbol with an edge mask formed by 2 neighbor diagonals of the
  // main diagonal
  Mat sideDiagsEdgeMask = Mat::zeros(getSz(), getSz(), CV_8UC1);
  sideDiagsEdgeMask.diag(1) = 255U;   // 2nd diagonal lower half
  sideDiagsEdgeMask.diag(-1) = 255U;  // 2nd diagonal upper half
  const unsigned cnzEdge = 2U * (getSz() - 1U);
  BOOST_REQUIRE((unsigned)countNonZero(sideDiagsEdgeMask) == cnzEdge);

  // Using a symbol with a background mask full except the main 3 diagonals
  Mat allBut3DiagsBgMask = Mat(getSz(), getSz(), CV_8UC1, Scalar(255U));
  allBut3DiagsBgMask.diag() = 0U;
  allBut3DiagsBgMask.diag(1) = 0U;   // 2nd diagonal lower half
  allBut3DiagsBgMask.diag(-1) = 0U;  // 2nd diagonal upper half
  const unsigned cnzBg = getArea() - 3U * getSz() + 2U;
  BOOST_REQUIRE((unsigned)countNonZero(allBut3DiagsBgMask) == cnzBg);

  Mat groundedGlyph = Mat::zeros(getSz(), getSz(), CV_8UC1);  // bg will stay 0
  const unsigned char edgeLevel = 125U,
                      maxGlyph = edgeLevel << 1;            // 250
  add(groundedGlyph, maxGlyph, groundedGlyph, diagFgMask);  // set fg
  add(groundedGlyph, edgeLevel, groundedGlyph,
      sideDiagsEdgeMask);  // set edges
  groundedGlyph.convertTo(groundedGlyph, CV_64FC1, 1. / 255);

  SymData sd(
      NOT_RELEVANT_UL,  // symbol code (not relevant here)
      NOT_RELEVANT_SZ,  // symbol index (not relevant here)
      NOT_RELEVANT_D,   // min brightness value 0..1 range (not relevant here)
      maxGlyph / 255.,  // diff between min..max, each in range 0..1
      NOT_RELEVANT_D,   // avgPixVal (not relevant here)
      NOT_RELEVANT_D,   // normSymMiu0 (not relevant here)
      NOT_RELEVANT_POINT,  // mc sym for original fg & bg (not relevant here)
      SymData::IdxMatMap{
          {(int)ISymData::MaskType::Fg, diagFgMask},
          {(int)ISymData::MaskType::Bg, allBut3DiagsBgMask},
          {(int)ISymData::MaskType::Edge, sideDiagsEdgeMask},
          {(int)ISymData::MaskType::GroundedSym, groundedGlyph}});

  // Testing on an uniform lower triangular patch
  patchD255 = Mat::zeros(getSz(), getSz(), CV_64FC1);
  double expectedFg = valRand, expectedBg = valRand / 2.,
         expectedSdevEdge = valRand * sqrt(5) / 4.;
  for (int i = 0; i < (int)getSz(); ++i)
    patchD255.diag(-i) = (double)valRand;  // i-th lower diagonal set on valRand
  BOOST_REQUIRE((unsigned)countNonZero(patchD255) == (getArea() + getSz()) / 2);
  res = em.assessMatch(patchD255, sd, cd, mp);
  BOOST_REQUIRE(mp.getFg() && mp.getBg() && mp.getSdevEdge());
  BOOST_TEST(*mp.getFg() == expectedFg, tol(1e-4));
  BOOST_TEST(*mp.getBg() == expectedBg, tol(1e-4));
  BOOST_TEST(*mp.getSdevEdge() == expectedSdevEdge, tol(1e-4));
  BOOST_TEST(res == 1. - expectedSdevEdge / 255, tol(1e-4));
  TITLED_AUTO_TEST_CASE_END
}

TITLED_AUTO_TEST_CASE_(CheckBgAspect_UniformPatch_PerfectMatch, SuiteSuffix) {
  ms.set_kSdevBg(1.);
  const BgMatch bm(ms);
  const auto valRand = randUnsignedChar(1U);

  // Using a symbol with a background mask full except the main 3 diagonals
  Mat allBut3DiagsBgMask = Mat(getSz(), getSz(), CV_8UC1, Scalar(255U));
  allBut3DiagsBgMask.diag() = 0U;
  allBut3DiagsBgMask.diag(1) = 0U;   // 2nd diagonal lower half
  allBut3DiagsBgMask.diag(-1) = 0U;  // 2nd diagonal upper half
  const unsigned cnz = getArea() - 3U * getSz() + 2U;
  BOOST_REQUIRE((unsigned)countNonZero(allBut3DiagsBgMask) == cnz);
  SymData sd(
      NOT_RELEVANT_UL,  // symbol code (not relevant here)
      NOT_RELEVANT_SZ,  // symbol index (not relevant here)
      NOT_RELEVANT_D,
      NOT_RELEVANT_D,      // min and diff between min..max, each in range 0..1
                           // (not relevant here)
      NOT_RELEVANT_D,      // avgPixVal (not relevant here)
      NOT_RELEVANT_D,      // normSymMiu0 (not relevant here)
      NOT_RELEVANT_POINT,  // mc sym for original fg & bg (not relevant here)
      SymData::IdxMatMap{{(int)ISymData::MaskType::Bg, allBut3DiagsBgMask}});

  patchD255 = Mat(getSz(), getSz(), CV_64FC1, Scalar((double)valRand));
  res = bm.assessMatch(patchD255, sd, cd, mp);
  BOOST_REQUIRE(mp.getBg() && mp.getSdevBg());
  BOOST_TEST(*mp.getBg() == (double)valRand, tol(1e-4));
  BOOST_TEST(*mp.getSdevBg() == 0., tol(1e-4));
  BOOST_TEST(res == 1., tol(1e-4));
  TITLED_AUTO_TEST_CASE_END
}

TITLED_AUTO_TEST_CASE_(
    CheckBgAspect_GlyphWith3MainDiagsOn0AndPatchWithUpperHalfEmptyAndUniformLowerPart_ImperfectMatch,
    SuiteSuffix) {
  ms.set_kSdevBg(1.);
  const BgMatch bm(ms);
  const auto valRand = randUnsignedChar(1U);

  // Using a symbol with a background mask full except the main 3 diagonals
  Mat allBut3DiagsBgMask = Mat(getSz(), getSz(), CV_8UC1, Scalar(255U));
  allBut3DiagsBgMask.diag() = 0U;
  allBut3DiagsBgMask.diag(1) = 0U;   // 2nd diagonal lower half
  allBut3DiagsBgMask.diag(-1) = 0U;  // 2nd diagonal upper half
  const unsigned cnz = getArea() - 3U * getSz() + 2U;
  BOOST_REQUIRE((unsigned)countNonZero(allBut3DiagsBgMask) == cnz);
  SymData sd(
      NOT_RELEVANT_UL,  // symbol code (not relevant here)
      NOT_RELEVANT_SZ,  // symbol index (not relevant here)
      NOT_RELEVANT_D,
      NOT_RELEVANT_D,      // min and diff between min..max, each in range 0..1
                           // (not relevant here)
      NOT_RELEVANT_D,      // avgPixVal (not relevant here)
      NOT_RELEVANT_D,      // normSymMiu0 (not relevant here)
      NOT_RELEVANT_POINT,  // mc sym for original fg & bg (not relevant here)
      SymData::IdxMatMap{{(int)ISymData::MaskType::Bg, allBut3DiagsBgMask}});

  patchD255 = Mat(getSz(), getSz(), CV_64FC1, Scalar((double)valRand));
  patchD255.rowRange(0, getSz() / 2) = 0.;
  res = bm.assessMatch(patchD255, sd, cd, mp);
  BOOST_REQUIRE(mp.getBg() && mp.getSdevBg());
  BOOST_TEST(*mp.getBg() == valRand / 2., tol(1e-4));
  BOOST_TEST(*mp.getSdevBg() == valRand / 2., tol(1e-4));
  BOOST_TEST(res == 1. - valRand / 255., tol(1e-4));
  TITLED_AUTO_TEST_CASE_END
}

TITLED_AUTO_TEST_CASE_(
    CheckBgAspect_GlyphWith3MainDiagsOn0AndPatchRowValuesSameAsRowIndices_ImperfectMatch,
    SuiteSuffix) {
  ms.set_kSdevBg(1.);
  const BgMatch bm(ms);
  const auto valRand = randUnsignedChar(1U);

  // Using a symbol with a background mask full except the main 3 diagonals
  Mat allBut3DiagsBgMask = Mat(getSz(), getSz(), CV_8UC1, Scalar(255U));
  allBut3DiagsBgMask.diag() = 0U;
  allBut3DiagsBgMask.diag(1) = 0U;   // 2nd diagonal lower half
  allBut3DiagsBgMask.diag(-1) = 0U;  // 2nd diagonal upper half
  const unsigned cnz = getArea() - 3U * getSz() + 2U;
  BOOST_REQUIRE((unsigned)countNonZero(allBut3DiagsBgMask) == cnz);
  SymData sd(
      NOT_RELEVANT_UL,  // symbol code (not relevant here)
      NOT_RELEVANT_SZ,  // symbol index (not relevant here)
      NOT_RELEVANT_D,
      NOT_RELEVANT_D,      // min and diff between min..max, each in range 0..1
                           // (not relevant here)
      NOT_RELEVANT_D,      // avgPixVal (not relevant here)
      NOT_RELEVANT_D,      // normSymMiu0 (not relevant here)
      NOT_RELEVANT_POINT,  // mc sym for original fg & bg (not relevant here)
      SymData::IdxMatMap{{(int)ISymData::MaskType::Bg, allBut3DiagsBgMask}});

  patchD255 = Mat::zeros(getSz(), getSz(), CV_64FC1);
  double expectedMiu = (getSz() - 1U) / 2., expectedSdev = 0., aux;
  for (unsigned i = 0U; i < getSz(); ++i) {
    patchD255.row(i) = (double)i;
    aux = i - expectedMiu;
    expectedSdev += aux * aux;
  }
  expectedSdev *= getSz() - 3.;
  expectedSdev += 2 * expectedMiu * expectedMiu;
  expectedSdev = sqrt(expectedSdev / cnz);
  res = bm.assessMatch(patchD255, sd, cd, mp);
  BOOST_REQUIRE(mp.getBg() && mp.getSdevBg());
  BOOST_TEST(*mp.getBg() == expectedMiu, tol(1e-4));
  BOOST_TEST(*mp.getSdevBg() == expectedSdev, tol(1e-4));
  BOOST_TEST(res == 1. - expectedSdev / 127.5, tol(1e-4));
  TITLED_AUTO_TEST_CASE_END
}

TITLED_AUTO_TEST_CASE_(CheckContrastAspect_UniformPatch_0Contrast,
                       SuiteSuffix) {
  ms.set_kContrast(1.);
  const BetterContrast bc(ms);
  const auto valRand = randUnsignedChar(1U);

  // Using a symbol with a diagonal foreground mask
  const Mat diagFgMask = Mat::diag(Mat(1, getSz(), CV_8UC1, Scalar(255U)));

  // Using a symbol with a background mask full except the main 3 diagonals
  Mat allBut3DiagsBgMask = Mat(getSz(), getSz(), CV_8UC1, Scalar(255U));
  allBut3DiagsBgMask.diag() = 0U;
  allBut3DiagsBgMask.diag(1) = 0U;   // 2nd diagonal lower half
  allBut3DiagsBgMask.diag(-1) = 0U;  // 2nd diagonal upper half
  const unsigned cnz = getArea() - 3U * getSz() + 2U;
  BOOST_REQUIRE((unsigned)countNonZero(allBut3DiagsBgMask) == cnz);
  SymData sd(
      NOT_RELEVANT_UL,  // symbol code (not relevant here)
      NOT_RELEVANT_SZ,  // symbol index (not relevant here)
      NOT_RELEVANT_D,
      NOT_RELEVANT_D,      // min and diff between min..max, each in range 0..1
                           // (not relevant here)
      NOT_RELEVANT_D,      // avgPixVal (not relevant here)
      NOT_RELEVANT_D,      // normSymMiu0 (not relevant here)
      NOT_RELEVANT_POINT,  // mc sym for original fg & bg (not relevant here)
      SymData::IdxMatMap{{(int)ISymData::MaskType::Fg, diagFgMask},
                         {(int)ISymData::MaskType::Bg, allBut3DiagsBgMask}});

  // Testing on a uniform patch
  patchD255 = Mat(getSz(), getSz(), CV_64FC1, Scalar((double)valRand));
  res = bc.assessMatch(patchD255, sd, cd, mp);
  BOOST_REQUIRE(mp.getFg() && mp.getBg());
  BOOST_TEST(*mp.getFg() == (double)valRand, tol(1e-4));
  BOOST_TEST(*mp.getBg() == (double)valRand, tol(1e-4));
  BOOST_TEST(res == 0., tol(1e-4));
  TITLED_AUTO_TEST_CASE_END
}

TITLED_AUTO_TEST_CASE_(
    CheckContrastAspect_EdgyDiagGlyphAndDiagPatchWithMaxContrast_ContrastFromPatch,
    SuiteSuffix) {
  ms.set_kContrast(1.);
  const BetterContrast bc(ms);
  const auto valRand = randUnsignedChar(1U);

  // Using a symbol with a diagonal foreground mask
  const Mat diagFgMask = Mat::diag(Mat(1, getSz(), CV_8UC1, Scalar(255U)));

  // Using a symbol with a background mask full except the main 3 diagonals
  Mat allBut3DiagsBgMask = Mat(getSz(), getSz(), CV_8UC1, Scalar(255U));
  allBut3DiagsBgMask.diag() = 0U;
  allBut3DiagsBgMask.diag(1) = 0U;   // 2nd diagonal lower half
  allBut3DiagsBgMask.diag(-1) = 0U;  // 2nd diagonal upper half
  const unsigned cnz = getArea() - 3U * getSz() + 2U;
  BOOST_REQUIRE((unsigned)countNonZero(allBut3DiagsBgMask) == cnz);
  SymData sd(
      NOT_RELEVANT_UL,  // symbol code (not relevant here)
      NOT_RELEVANT_SZ,  // symbol index (not relevant here)
      NOT_RELEVANT_D,
      NOT_RELEVANT_D,      // min and diff between min..max, each in range 0..1
                           // (not relevant here)
      NOT_RELEVANT_D,      // avgPixVal (not relevant here)
      NOT_RELEVANT_D,      // normSymMiu0 (not relevant here)
      NOT_RELEVANT_POINT,  // mc sym for original fg & bg (not relevant here)
      SymData::IdxMatMap{{(int)ISymData::MaskType::Fg, diagFgMask},
                         {(int)ISymData::MaskType::Bg, allBut3DiagsBgMask}});

  patchD255 = Mat::diag(Mat(1, getSz(), CV_64FC1, Scalar(255.)));
  res = bc.assessMatch(patchD255, sd, cd, mp);
  BOOST_REQUIRE(mp.getFg() && mp.getBg());
  BOOST_TEST(*mp.getFg() == 255., tol(1e-4));
  BOOST_TEST(*mp.getBg() == 0., tol(1e-4));
  BOOST_TEST(res == 1., tol(1e-4));
  TITLED_AUTO_TEST_CASE_END
}

TITLED_AUTO_TEST_CASE_(
    CheckContrastAspect_EdgyDiagGlyphAndDiagPatchWithHalfContrast_ContrastFromPatch,
    SuiteSuffix) {
  ms.set_kContrast(1.);
  const BetterContrast bc(ms);
  const auto valRand = randUnsignedChar(1U);

  // Using a symbol with a diagonal foreground mask
  const Mat diagFgMask = Mat::diag(Mat(1, getSz(), CV_8UC1, Scalar(255U)));

  // Using a symbol with a background mask full except the main 3 diagonals
  Mat allBut3DiagsBgMask = Mat(getSz(), getSz(), CV_8UC1, Scalar(255U));
  allBut3DiagsBgMask.diag() = 0U;
  allBut3DiagsBgMask.diag(1) = 0U;   // 2nd diagonal lower half
  allBut3DiagsBgMask.diag(-1) = 0U;  // 2nd diagonal upper half
  const unsigned cnz = getArea() - 3U * getSz() + 2U;
  BOOST_REQUIRE((unsigned)countNonZero(allBut3DiagsBgMask) == cnz);
  SymData sd(
      NOT_RELEVANT_UL,  // symbol code (not relevant here)
      NOT_RELEVANT_SZ,  // symbol index (not relevant here)
      NOT_RELEVANT_D,
      NOT_RELEVANT_D,      // min and diff between min..max, each in range 0..1
                           // (not relevant here)
      NOT_RELEVANT_D,      // avgPixVal (not relevant here)
      NOT_RELEVANT_D,      // normSymMiu0 (not relevant here)
      NOT_RELEVANT_POINT,  // mc sym for original fg & bg (not relevant here)
      SymData::IdxMatMap{{(int)ISymData::MaskType::Fg, diagFgMask},
                         {(int)ISymData::MaskType::Bg, allBut3DiagsBgMask}});

  patchD255 = Mat::diag(Mat(1, getSz(), CV_64FC1, Scalar(127.5)));
  res = bc.assessMatch(patchD255, sd, cd, mp);
  BOOST_REQUIRE(mp.getFg() && mp.getBg());
  BOOST_TEST(*mp.getFg() == 127.5, tol(1e-4));
  BOOST_TEST(*mp.getBg() == 0., tol(1e-4));
  BOOST_TEST(res == .5, tol(1e-4));
  TITLED_AUTO_TEST_CASE_END
}

TITLED_AUTO_TEST_CASE_(
    CheckGravitationalSmoothness_PatchAndGlyphArePixelsOnOppositeCorners_ImperfectMatch,
    SuiteSuffix) {
  ms.set_kMCsOffset(1.);
  const unsigned sz_1 = getSz() - 1U;
  cd.useNewSymSize(getSz());
  const GravitationalSmoothness gs(ms);

  // Checking a symbol that has a single 255 pixel in bottom right corner
  double avgPixVal = 1. / getArea();  // a single pixel set to max
  Point2d origMcSym(1., 1.);
  Mat fgMask = Mat::zeros(getSz(), getSz(), CV_8UC1),
      bgMask(getSz(), getSz(), CV_8UC1, Scalar(255U));
  fgMask.at<unsigned char>(sz_1, sz_1) = 255U;
  bgMask.at<unsigned char>(sz_1, sz_1) = 0U;

  SymData sd(NOT_RELEVANT_UL,  // symbol code (not relevant here)
             NOT_RELEVANT_SZ,  // symbol index (not relevant here)
             NOT_RELEVANT_D,
             NOT_RELEVANT_D,  // min and diff between min..max, each in range
                              // 0..1 (not relevant here)
             avgPixVal,
             NOT_RELEVANT_D,  // normSymMiu0 (not relevant here)
             origMcSym,
             SymData::IdxMatMap{{(int)ISymData::MaskType::Fg, fgMask},
                                {(int)ISymData::MaskType::Bg, bgMask}});

  // Using a patch with a single 255 pixel in top left corner
  patchD255 = Mat::zeros(getSz(), getSz(), CV_64FC1);
  patchD255.at<double>(0, 0) = 255.;
  res = gs.assessMatch(patchD255, sd, cd, mp);
  BOOST_REQUIRE(mp.getFg() && mp.getBg() && mp.getSymDensity() &&
                mp.getMcPatchApprox() && mp.getMcPatch());
  BOOST_TEST(*mp.getFg() == 0., tol(1e-4));
  BOOST_TEST(*mp.getBg() == 255. / (getArea() - 1), tol(1e-4));
  BOOST_TEST(*mp.getSymDensity() == 1. / getArea(), tol(1e-8));

  // Symbol's mc migrated diagonally from bottom-right corner to above the
  // center of patch. Migration occurred due to the new fg & bg of the symbol
  BOOST_TEST(mp.getMcPatchApprox()->x == .5 - .5 / (pow(getSz(), 2) - 1.),
             tol(1e-8));
  BOOST_TEST(mp.getMcPatchApprox()->y == .5 - .5 / (pow(getSz(), 2) - 1.),
             tol(1e-8));
  BOOST_TEST(mp.getMcPatch()->x == 0., tol(1e-4));
  BOOST_TEST(mp.getMcPatch()->y == 0., tol(1e-4));
  BOOST_TEST(res == 1. + (CachedData::MassCenters::preferredMaxMcDist -
                          M_SQRT2 * .5 * (1. - 1. / (pow(getSz(), 2) - 1.))) *
                             CachedData::MassCenters::invComplPrefMaxMcDist,
             tol(1e-8));
  TITLED_AUTO_TEST_CASE_END
}

TITLED_AUTO_TEST_CASE_(
    CheckGravitationalSmoothness_CornerPixelAsGlyphAndCenterOfEdgeAsPatch_McGlyphCenters,
    SuiteSuffix) {
  ms.set_kMCsOffset(1.);
  const unsigned sz_1 = getSz() - 1U;
  cd.useNewSymSize(getSz());
  const GravitationalSmoothness gs(ms);

  // Checking a symbol that has a single 255 pixel in bottom right corner
  double avgPixVal = 1. / getArea();  // a single pixel set to max
  Point2d origMcSym(1., 1.);
  Mat fgMask = Mat::zeros(getSz(), getSz(), CV_8UC1),
      bgMask(getSz(), getSz(), CV_8UC1, Scalar(255U));
  fgMask.at<unsigned char>(sz_1, sz_1) = 255U;
  bgMask.at<unsigned char>(sz_1, sz_1) = 0U;

  SymData sd(NOT_RELEVANT_UL,  // symbol code (not relevant here)
             NOT_RELEVANT_SZ,  // symbol index (not relevant here)
             NOT_RELEVANT_D,
             NOT_RELEVANT_D,  // min and diff between min..max, each in range
                              // 0..1 (not relevant here)
             avgPixVal,
             NOT_RELEVANT_D,  // normSymMiu0 (not relevant here)
             origMcSym,
             SymData::IdxMatMap{{(int)ISymData::MaskType::Fg, fgMask},
                                {(int)ISymData::MaskType::Bg, bgMask}});

  // Using a patch with the middle pixels pair on the top row on 255
  // Patch mc is at half width on top row.
  patchD255 = Mat::zeros(getSz(), getSz(), CV_64FC1);
  Mat(patchD255, Rect(getSz() / 2U - 1U, 0, 2, 1)) = Scalar(255.);
  BOOST_REQUIRE(countNonZero(patchD255) == 2);
  res = gs.assessMatch(patchD255, sd, cd, mp);
  BOOST_REQUIRE(mp.getFg() && mp.getBg() && mp.getSymDensity() &&
                mp.getMcPatchApprox() && mp.getMcPatch());
  BOOST_TEST(*mp.getFg() == 0., tol(1e-4));
  BOOST_TEST(*mp.getBg() == 2 * 255. / (getArea() - 1), tol(1e-4));
  BOOST_TEST(*mp.getSymDensity() == 1. / getArea(), tol(1e-8));

  // Symbol's mc migrated diagonally from bottom-right corner to above the
  // center of patch. Migration occurred due to the new fg & bg of the symbol
  BOOST_TEST(mp.getMcPatchApprox()->x == .5 - .5 / (pow(getSz(), 2) - 1.),
             tol(1e-8));
  BOOST_TEST(mp.getMcPatchApprox()->y == .5 - .5 / (pow(getSz(), 2) - 1.),
             tol(1e-8));
  BOOST_TEST(mp.getMcPatch()->x == .5, tol(1e-4));
  BOOST_TEST(mp.getMcPatch()->y == 0., tol(1e-4));
  double dx = .5 / (pow(getSz(), 2) - 1.),
         dy = .5 * (1. - 1. / (pow(getSz(), 2) - 1.));
  BOOST_TEST(res == 1. + (CachedData::MassCenters::preferredMaxMcDist -
                          sqrt(dx * dx + dy * dy)) *
                             CachedData::MassCenters::invComplPrefMaxMcDist,
             tol(1e-8));
  TITLED_AUTO_TEST_CASE_END
}

TITLED_AUTO_TEST_CASE_(
    CheckGravitationalSmoothness_CornerPixelAsGlyphAndOtherCornerAsPatch_McGlyphCenters,
    SuiteSuffix) {
  ms.set_kMCsOffset(1.);
  const unsigned sz_1 = getSz() - 1U;
  cd.useNewSymSize(getSz());
  const GravitationalSmoothness gs(ms);

  // Checking a symbol that has a single 255 pixel in bottom right corner
  double avgPixVal = 1. / getArea();  // a single pixel set to max
  Point2d origMcSym(1., 1.);
  Mat fgMask = Mat::zeros(getSz(), getSz(), CV_8UC1),
      bgMask(getSz(), getSz(), CV_8UC1, Scalar(255U));
  fgMask.at<unsigned char>(sz_1, sz_1) = 255U;
  bgMask.at<unsigned char>(sz_1, sz_1) = 0U;

  SymData sd(NOT_RELEVANT_UL,  // symbol code (not relevant here)
             NOT_RELEVANT_SZ,  // symbol index (not relevant here)
             NOT_RELEVANT_D,
             NOT_RELEVANT_D,  // min and diff between min..max, each in range
                              // 0..1 (not relevant here)
             avgPixVal,
             NOT_RELEVANT_D,  // normSymMiu0 (not relevant here)
             origMcSym,
             SymData::IdxMatMap{{(int)ISymData::MaskType::Fg, fgMask},
                                {(int)ISymData::MaskType::Bg, bgMask}});

  // Using a patch with the last pixel on the top row on 255
  patchD255 = Mat::zeros(getSz(), getSz(), CV_64FC1);
  patchD255.at<double>(0, sz_1) = 255.;
  res = gs.assessMatch(patchD255, sd, cd, mp);
  BOOST_REQUIRE(mp.getFg() && mp.getBg() && mp.getSymDensity() &&
                mp.getMcPatchApprox() && mp.getMcPatch());
  BOOST_TEST(*mp.getFg() == 0., tol(1e-4));
  BOOST_TEST(*mp.getBg() == 255. / (getArea() - 1), tol(1e-4));
  BOOST_TEST(*mp.getSymDensity() == 1. / getArea(), tol(1e-8));

  // Symbol's mc migrated diagonally from bottom-right corner to above the
  // center of patch. Migration occurred due to the new fg & bg of the symbol
  BOOST_TEST(mp.getMcPatchApprox()->x == .5 - .5 / (pow(getSz(), 2) - 1.),
             tol(1e-8));
  BOOST_TEST(mp.getMcPatchApprox()->y == .5 - .5 / (pow(getSz(), 2) - 1.),
             tol(1e-8));
  BOOST_TEST(mp.getMcPatch()->x == 1., tol(1e-4));
  BOOST_TEST(mp.getMcPatch()->y == 0., tol(1e-4));
  double dx = .5 * (1. + 1 / (pow(getSz(), 2) - 1.)),
         dy = .5 * (1. - 1 / (pow(getSz(), 2) - 1.));
  BOOST_TEST(res == 1. + (CachedData::MassCenters::preferredMaxMcDist -
                          sqrt(dx * dx + dy * dy)) *
                             CachedData::MassCenters::invComplPrefMaxMcDist,
             tol(1e-8));
  TITLED_AUTO_TEST_CASE_END
}

TITLED_AUTO_TEST_CASE_(
    CheckDirectionalSmoothness_PatchAndGlyphArePixelsOnOppositeCorners_ImperfectMatch,
    SuiteSuffix) {
  ms.set_kCosAngleMCs(1.);
  const unsigned sz_1 = getSz() - 1U;
  cd.useNewSymSize(getSz());
  const DirectionalSmoothness ds(ms);

  // Checking a symbol that has a single 255 pixel in bottom right corner
  double avgPixVal = 1. / getArea();  // a single pixel set to max
  Point2d origMcSym(1., 1.);
  Mat fgMask = Mat::zeros(getSz(), getSz(), CV_8UC1),
      bgMask(getSz(), getSz(), CV_8UC1, Scalar(255U));
  fgMask.at<unsigned char>(sz_1, sz_1) = 255U;
  bgMask.at<unsigned char>(sz_1, sz_1) = 0U;

  SymData sd(NOT_RELEVANT_UL,  // symbol code (not relevant here)
             NOT_RELEVANT_SZ,  // symbol index (not relevant here)
             NOT_RELEVANT_D,
             NOT_RELEVANT_D,  // min and diff between min..max, each in range
                              // 0..1 (not relevant here)
             avgPixVal,
             NOT_RELEVANT_D,  // normSymMiu0 (not relevant here)
             origMcSym,
             SymData::IdxMatMap{{(int)ISymData::MaskType::Fg, fgMask},
                                {(int)ISymData::MaskType::Bg, bgMask}});

  // Using a patch with a single 255 pixel in top left corner
  // Same as 1st scenario from Gravitational Smoothness
  patchD255 = Mat::zeros(getSz(), getSz(), CV_64FC1);
  patchD255.at<double>(0, 0) = 255.;
  res = ds.assessMatch(patchD255, sd, cd, mp);
  BOOST_REQUIRE(mp.getFg() && mp.getBg() && mp.getSymDensity() &&
                mp.getMcPatchApprox() && mp.getMcPatch());
  // Symbol's mc migrated diagonally from bottom-right corner to above the
  // center of patch. Migration occurred due to the new fg & bg of the symbol
  BOOST_TEST(mp.getMcPatchApprox()->x == .5 - .5 / (pow(getSz(), 2) - 1.),
             tol(1e-8));
  BOOST_TEST(mp.getMcPatchApprox()->y == .5 - .5 / (pow(getSz(), 2) - 1.),
             tol(1e-8));
  BOOST_TEST(mp.getMcPatch()->x == 0., tol(1e-4));
  BOOST_TEST(mp.getMcPatch()->y == 0., tol(1e-4));
  BOOST_TEST(res == 2. * (2. - M_SQRT2) *
                        (CachedData::MassCenters::a_mcsOffsetFactor() *
                             *mp.getMcsOffset() +
                         CachedData::MassCenters::b_mcsOffsetFactor()),
             tol(1e-8));  // angle = 0 => cos = 1
  TITLED_AUTO_TEST_CASE_END
}

TITLED_AUTO_TEST_CASE_(
    CheckDirectionalSmoothness_CornerPixelAsGlyphAndCenterOfEdgeAsPatch_McGlyphCenters,
    SuiteSuffix) {
  ms.set_kCosAngleMCs(1.);
  const unsigned sz_1 = getSz() - 1U;
  cd.useNewSymSize(getSz());
  const DirectionalSmoothness ds(ms);

  // Checking a symbol that has a single 255 pixel in bottom right corner
  double avgPixVal = 1. / getArea();  // a single pixel set to max
  Point2d origMcSym(1., 1.);
  Mat fgMask = Mat::zeros(getSz(), getSz(), CV_8UC1),
      bgMask(getSz(), getSz(), CV_8UC1, Scalar(255U));
  fgMask.at<unsigned char>(sz_1, sz_1) = 255U;
  bgMask.at<unsigned char>(sz_1, sz_1) = 0U;

  SymData sd(NOT_RELEVANT_UL,  // symbol code (not relevant here)
             NOT_RELEVANT_SZ,  // symbol index (not relevant here)
             NOT_RELEVANT_D,
             NOT_RELEVANT_D,  // min and diff between min..max, each in range
                              // 0..1 (not relevant here)
             avgPixVal,
             NOT_RELEVANT_D,  // normSymMiu0 (not relevant here)
             origMcSym,
             SymData::IdxMatMap{{(int)ISymData::MaskType::Fg, fgMask},
                                {(int)ISymData::MaskType::Bg, bgMask}});

  // Using a patch with the middle pixels pair on the top row on 255
  // Patch mc is at half width on top row.
  // Same as 2nd scenario from Gravitational Smoothness
  patchD255 = Mat::zeros(getSz(), getSz(), CV_64FC1);
  Mat(patchD255, Rect(getSz() / 2U - 1U, 0, 2, 1)) = Scalar(255.);
  res = ds.assessMatch(patchD255, sd, cd, mp);
  BOOST_REQUIRE(mp.getFg() && mp.getBg() && mp.getSymDensity() &&
                mp.getMcPatchApprox() && mp.getMcPatch());
  // Symbol's mc migrated diagonally from bottom-right corner to above the
  // center of patch. Migration occurred due to the new fg & bg of the symbol
  BOOST_TEST(mp.getMcPatchApprox()->x == .5 - .5 / (pow(getSz(), 2) - 1.),
             tol(1e-8));
  BOOST_TEST(mp.getMcPatchApprox()->y == .5 - .5 / (pow(getSz(), 2) - 1.),
             tol(1e-8));
  BOOST_TEST(mp.getMcPatch()->x == .5, tol(1e-4));
  BOOST_TEST(mp.getMcPatch()->y == 0., tol(1e-4));
  BOOST_TEST(res == (CachedData::MassCenters::a_mcsOffsetFactor() *
                         *mp.getMcsOffset() +
                     CachedData::MassCenters::b_mcsOffsetFactor()),
             tol(1e-8));  // angle = 45 => cos = sqrt(2)/2
  TITLED_AUTO_TEST_CASE_END
}

TITLED_AUTO_TEST_CASE_(
    CheckDirectionalSmoothness_CornerPixelAsGlyphAndOtherCornerAsPatch_McGlyphCenters,
    SuiteSuffix) {
  ms.set_kCosAngleMCs(1.);
  const unsigned sz_1 = getSz() - 1U;
  cd.useNewSymSize(getSz());
  const DirectionalSmoothness ds(ms);

  // Checking a symbol that has a single 255 pixel in bottom right corner
  double avgPixVal = 1. / getArea();  // a single pixel set to max
  Point2d origMcSym(1., 1.);
  Mat fgMask = Mat::zeros(getSz(), getSz(), CV_8UC1),
      bgMask(getSz(), getSz(), CV_8UC1, Scalar(255U));
  fgMask.at<unsigned char>(sz_1, sz_1) = 255U;
  bgMask.at<unsigned char>(sz_1, sz_1) = 0U;

  SymData sd(NOT_RELEVANT_UL,  // symbol code (not relevant here)
             NOT_RELEVANT_SZ,  // symbol index (not relevant here)
             NOT_RELEVANT_D,
             NOT_RELEVANT_D,  // min and diff between min..max, each in range
                              // 0..1 (not relevant here)
             avgPixVal,
             NOT_RELEVANT_D,  // normSymMiu0 (not relevant here)
             origMcSym,
             SymData::IdxMatMap{{(int)ISymData::MaskType::Fg, fgMask},
                                {(int)ISymData::MaskType::Bg, bgMask}});

  // Using a patch with the last pixel on the top row on 255
  // Same as 3rd scenario from Gravitational Smoothness
  patchD255 = Mat::zeros(getSz(), getSz(), CV_64FC1);
  patchD255.at<double>(0, sz_1) = 255.;
  res = ds.assessMatch(patchD255, sd, cd, mp);
  BOOST_REQUIRE(mp.getFg() && mp.getBg() && mp.getSymDensity() &&
                mp.getMcPatchApprox() && mp.getMcPatch());
  // Symbol's mc migrated diagonally from bottom-right corner to above the
  // center of patch. Migration occurred due to the new fg & bg of the symbol
  BOOST_TEST(mp.getMcPatchApprox()->x == .5 - .5 / (pow(getSz(), 2) - 1.),
             tol(1e-8));
  BOOST_TEST(mp.getMcPatchApprox()->y == .5 - .5 / (pow(getSz(), 2) - 1.),
             tol(1e-8));
  BOOST_TEST(mp.getMcPatch()->x == 1., tol(1e-4));
  BOOST_TEST(mp.getMcPatch()->y == 0., tol(1e-4));
  BOOST_TEST(
      res == (2. - M_SQRT2) * (CachedData::MassCenters::a_mcsOffsetFactor() *
                                   *mp.getMcsOffset() +
                               CachedData::MassCenters::b_mcsOffsetFactor()),
      tol(1e-8));  // angle is 90 => cos = 0
  TITLED_AUTO_TEST_CASE_END
}

TITLED_AUTO_TEST_CASE_(CheckLargerSymAspect_EmptyGlyph_Density0, SuiteSuffix) {
  ms.set_kSymDensity(1.);
  cd.smallGlyphsCoverage =
      .1;  // large glyphs need to cover more than 10% of their box
  const LargerSym ls(ms);

  SymData sd(
      NOT_RELEVANT_UL,  // symbol code (not relevant here)
      NOT_RELEVANT_SZ,  // symbol index (not relevant here)
      NOT_RELEVANT_D,
      NOT_RELEVANT_D,  // min and diff between min..max, each in range 0..1
                       // (not relevant here)
      0.,              // avgPixVal (INITIALLY, AN EMPTY SYMBOL IS CONSIDERED)
      NOT_RELEVANT_D,  // normSymMiu0 (not relevant here)
      NOT_RELEVANT_POINT,  // mc sym for original fg & bg (not relevant here)
      SymData::IdxMatMap{});

  // Testing with an empty symbol (sd.avgPixVal == 0)
  res = ls.assessMatch(NOT_RELEVANT_MAT, sd, cd, mp);
  BOOST_REQUIRE(mp.getSymDensity());
  BOOST_TEST(*mp.getSymDensity() == 0., tol(1e-4));
  BOOST_TEST(res == 1. - cd.getSmallGlyphsCoverage(), tol(1e-4));
  TITLED_AUTO_TEST_CASE_END
}

TITLED_AUTO_TEST_CASE_(
    CheckLargerSymAspect_InferiorLimitOfLargeSymbols_QualifiesAsLarge,
    SuiteSuffix) {
  ms.set_kSymDensity(1.);
  cd.smallGlyphsCoverage =
      .1;  // large glyphs need to cover more than 10% of their box
  const LargerSym ls(ms);

  SymData sd(
      NOT_RELEVANT_UL,  // symbol code (not relevant here)
      NOT_RELEVANT_SZ,  // symbol index (not relevant here)
      NOT_RELEVANT_D,
      NOT_RELEVANT_D,  // min and diff between min..max, each in range 0..1
                       // (not relevant here)
      cd.smallGlyphsCoverage,  // avgPixVal (symbol that just enters the
                               // 'large symbols' category)
      NOT_RELEVANT_D,          // normSymMiu0 (not relevant here)
      NOT_RELEVANT_POINT,  // mc sym for original fg & bg (not relevant here)
      SymData::IdxMatMap{});

  res = ls.assessMatch(NOT_RELEVANT_MAT, sd, cd, mp);
  BOOST_REQUIRE(mp.getSymDensity());
  BOOST_TEST(*mp.getSymDensity() == cd.getSmallGlyphsCoverage(), tol(1e-4));
  BOOST_TEST(res == 1., tol(1e-4));
  TITLED_AUTO_TEST_CASE_END
}

TITLED_AUTO_TEST_CASE_(CheckLargerSymAspect_LargestPosibleSymbol_LargestScore,
                       SuiteSuffix) {
  ms.set_kSymDensity(1.);
  cd.smallGlyphsCoverage =
      .1;  // large glyphs need to cover more than 10% of their box
  const LargerSym ls(ms);

  SymData sd(
      NOT_RELEVANT_UL,  // symbol code (not relevant here)
      NOT_RELEVANT_SZ,  // symbol index (not relevant here)
      NOT_RELEVANT_D,
      NOT_RELEVANT_D,      // min and diff between min..max, each in range 0..1
                           // (not relevant here)
      1.,                  // avgPixVal (largest possible symbol)
      NOT_RELEVANT_D,      // normSymMiu0 (not relevant here)
      NOT_RELEVANT_POINT,  // mc sym for original fg & bg (not relevant here)
      SymData::IdxMatMap{});

  res = ls.assessMatch(NOT_RELEVANT_MAT, sd, cd, mp);
  BOOST_REQUIRE(mp.getSymDensity());
  BOOST_TEST(*mp.getSymDensity() == 1., tol(1e-4));
  BOOST_TEST(res == 2. - cd.getSmallGlyphsCoverage(), tol(1e-4));
  TITLED_AUTO_TEST_CASE_END
}
BOOST_AUTO_TEST_SUITE_END()  // MatchAspects_Tests

#endif  // BOOST_PP_IS_ITERATING
