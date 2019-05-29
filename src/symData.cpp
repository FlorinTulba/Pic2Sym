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

#include "blurBase.h"
#include "misc.h"
#include "pixMapSymBase.h"
#include "structuralSimilarity.h"
#include "symData.h"

using namespace std;
using namespace cv;

unsigned SymData::VERSION_FROM_LAST_IO_OP = UINT_MAX;

SymData::SymData(const Mat& negSym_,
                 const cv::Mat& symMiu0_,
                 unsigned long code_,
                 size_t symIdx_,
                 double minVal_,
                 double diffMinMax_,
                 double avgPixVal_,
                 double normSymMiu0_,
                 const Point2d& mc_,
                 const MatArray& masks_,
                 bool removable_ /* = false*/) noexcept
    : mc(mc_),
      negSym(negSym_),
      symMiu0(symMiu0_),
      masks(masks_),
      symIdx(symIdx_),
      minVal(minVal_),
      diffMinMax(diffMinMax_),
      avgPixVal(avgPixVal_),
      normSymMiu0(normSymMiu0_),
      code(code_),
      removable(removable_) {}

SymData::SymData(const IPixMapSym& pms, unsigned sz, bool forTinySym) noexcept
    : mc(pms.getMc()),
      negSym(pms.toMat(sz, true)),
      symIdx(pms.getSymIdx()),
      avgPixVal(pms.getAvgPixVal()),
      code(pms.getSymCode()),
      removable(pms.isRemovable()) {
  computeFields(pms.toMatD01(sz), *this, forTinySym);
}

SymData::SymData(unsigned long code_ /* = ULONG_MAX*/,
                 size_t symIdx_ /* = 0ULL*/,
                 double avgPixVal_ /* = 0.*/,
                 const Point2d& mc_ /* = Point2d(.5, .5)*/) noexcept
    : mc(mc_), symIdx(symIdx_), avgPixVal(avgPixVal_), code(code_) {}

SymData::SymData(const Point2d& mc_, double avgPixVal_) noexcept
    : mc(mc_), avgPixVal(avgPixVal_) {}

SymData::SymData(const SymData& other) noexcept
    : mc(other.mc),
      negSym(other.negSym),
      symMiu0(other.symMiu0),
      masks(other.masks),
      symIdx(other.symIdx),
      minVal(other.minVal),
      diffMinMax(other.diffMinMax),
      avgPixVal(other.avgPixVal),
      normSymMiu0(other.normSymMiu0),
      code(other.code),
      removable(other.removable) {}

SymData::SymData(SymData&& other) noexcept : SymData(other) {
  other.negSym.release();
  other.symMiu0.release();
  for (Mat& m : other.masks)
    m.release();
}

SymData& SymData::operator=(const SymData& other) noexcept {
  ISymData::operator=(other);

  // if (this != &other) { // Costly to always perform. Harmless & cheap if cut
#define REPLACE_FIELD(Field) Field = other.Field

  REPLACE_FIELD(code);
  REPLACE_FIELD(symIdx);
  REPLACE_FIELD(minVal);
  REPLACE_FIELD(diffMinMax);
  REPLACE_FIELD(avgPixVal);
  REPLACE_FIELD(normSymMiu0);
  REPLACE_FIELD(mc);
  REPLACE_FIELD(negSym);
  REPLACE_FIELD(symMiu0);
  REPLACE_FIELD(removable);

  for (int i = 0; i < (int)ISymData::MaskType::MATRICES_COUNT; ++i)
    REPLACE_FIELD(masks[(size_t)i]);

#undef REPLACE_FIELD
  //}

  return *this;
}

SymData& SymData::operator=(SymData&& other) noexcept {
  operator=(other);

  if (this != &other) {  // Mandatory this time, unlike for the copy assignment
    other.negSym.release();
    other.symMiu0.release();

    for (Mat& m : other.masks)
      m.release();
  }

  return *this;
}

const Point2d& SymData::getMc() const noexcept {
  return mc;
}

const Mat& SymData::getNegSym() const noexcept {
  return negSym;
}

const Mat& SymData::getSymMiu0() const noexcept {
  return symMiu0;
}

double SymData::getNormSymMiu0() const noexcept {
  return normSymMiu0;
}

const ISymData::MatArray& SymData::getMasks() const noexcept {
  return masks;
}

size_t SymData::getSymIdx() const noexcept {
  return symIdx;
}

#ifdef UNIT_TESTING
double SymData::getMinVal() const noexcept {
  return minVal;
}
#endif  // UNIT_TESTING defined

double SymData::getDiffMinMax() const noexcept {
  return diffMinMax;
}

double SymData::getAvgPixVal() const noexcept {
  return avgPixVal;
}

unsigned long SymData::getCode() const noexcept {
  return code;
}

bool SymData::isRemovable() const noexcept {
  return removable;
}

#pragma warning(disable : WARN_EXPR_ALWAYS_FALSE)
bool SymData::olderVersionDuringLastIO() noexcept {
  return VERSION_FROM_LAST_IO_OP < VERSION;
}
#pragma warning(default : WARN_EXPR_ALWAYS_FALSE)

void SymData::computeSymMiu0Related(const cv::Mat& glyph,
                                    double miu,
                                    SymData& sd) noexcept {
  sd.symMiu0 = glyph - miu;
  sd.normSymMiu0 = norm(sd.symMiu0, NORM_L2);
}

void SymData::computeFields(const cv::Mat& glyph,
                            SymData& sd,
                            bool forTinySym) noexcept {
  computeSymMiu0Related(glyph, sd.avgPixVal, sd);

  /*
  SymData_computeFields_STILL_BG and STILL_FG from below are constants for
  foreground / background thresholds.

  1/255 = 0.00392, so 0.004 tolerates pixels with 1 brightness unit less / more
  than ideal STILL_BG was set to 0, as there are font families with extremely
  similar glyphs. When Unit Testing shouldn't identify exactly each glyph,
  STILL_BG might be > 0. But testing on 'BPmonoBold.ttf' does tolerate such
  larger values (0.025, for instance).
  */
  extern const double SymData_computeFields_STILL_BG;  // darkest shades

  // brightest shades
  static const double STILL_FG = 1. - SymData_computeFields_STILL_BG;

  double minVal, maxVal;
  minMaxIdx(glyph, &minVal, &maxVal);
  assert(maxVal < EPSp1);
  // ensures diffMinMax, groundedGlyph and blurOfGroundedGlyph are within 0..1

  sd.minVal = minVal;
  const double diffMinMax = sd.diffMinMax = maxVal - minVal;
  const Mat groundedGlyph = sd.masks[(size_t)MaskType::GroundedSym] =
      (minVal == 0. ? glyph.clone() : (glyph - minVal));  // min val on 0

  sd.masks[(size_t)MaskType::Fg] = (glyph >= (minVal + STILL_FG * diffMinMax));
  sd.masks[(size_t)MaskType::Bg] =
      (glyph <= (minVal + SymData_computeFields_STILL_BG * diffMinMax));

  // Storing a blurred version of the grounded glyph for structural similarity
  // match aspect
  Mat blurOfGroundedGlyph;
  StructuralSimilarity::supportBlur.process(groundedGlyph, blurOfGroundedGlyph,
                                            forTinySym);
  sd.masks[(size_t)MaskType::BlurredGrSym] = blurOfGroundedGlyph;

  // edgeMask selects all pixels that are not minVal, nor maxVal
  inRange(glyph, minVal + EPS, maxVal - EPS, sd.masks[(size_t)MaskType::Edge]);

  // Storing also the variance of the grounded glyph for structural similarity
  // match aspect Actual varianceOfGroundedGlyph is obtained in the subtraction
  // after the blur
  Mat blurOfGroundedGlyphSquared;
  StructuralSimilarity::supportBlur.process(
      groundedGlyph.mul(groundedGlyph), blurOfGroundedGlyphSquared, forTinySym);

  sd.masks[(size_t)MaskType::VarianceGrSym] =
      blurOfGroundedGlyphSquared - blurOfGroundedGlyph.mul(blurOfGroundedGlyph);
}

#pragma warning(disable : WARN_THROWS_ALTHOUGH_NOEXCEPT)
void SymData::setNegSym(const cv::Mat& negSym_) noexcept {
  // Check and throw invalid_argument if invalid negSym_
  negSym = negSym_;
}
#pragma warning(default : WARN_THROWS_ALTHOUGH_NOEXCEPT)

#pragma warning(disable : WARN_THROWS_ALTHOUGH_NOEXCEPT)
void SymData::setMc(const cv::Point2d& mc_) noexcept {
  // Check and throw invalid_argument if invalid mc_
  mc = mc_;
}
#pragma warning(default : WARN_THROWS_ALTHOUGH_NOEXCEPT)

#pragma warning(disable : WARN_THROWS_ALTHOUGH_NOEXCEPT)
void SymData::setAvgPixVal(double avgPixVal_) noexcept {
  // Check and throw invalid_argument if invalid avgPixVal_
  avgPixVal = avgPixVal_;
}
#pragma warning(default : WARN_THROWS_ALTHOUGH_NOEXCEPT)
