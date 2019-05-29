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

#include "patch.h"
#include "warnings.h"

#pragma warning(push, 0)

#include <opencv2/imgproc/imgproc.hpp>

#pragma warning(pop)

using namespace std;
using namespace cv;

Patch::Patch(const Mat& orig_, const Mat& blurred_, bool isColor_) noexcept
    : orig(orig_), blurred(blurred_), isColor(isColor_) {
  // Don't approximate rather uniform patches
  Mat grayBlurred;
  if (isColor)
    cvtColor(blurred, grayBlurred, COLOR_RGB2GRAY);
  else
    grayBlurred = blurred;

  double minVal, maxVal;
  // assessed on blurred patch, to avoid outliers bias
  minMaxIdx(grayBlurred, &minVal, &maxVal);
  extern const double Transformer_run_THRESHOLD_CONTRAST_BLURRED;
  if (maxVal - minVal < Transformer_run_THRESHOLD_CONTRAST_BLURRED) {
    needsApproximation = false;
    return;
  }

  // Configurable source of transformation - either the patch itself, or its
  // blurred version:
  extern const bool Transform_BlurredPatches_InsteadOf_Originals;
  const Mat& patch2Process =
      Transform_BlurredPatches_InsteadOf_Originals ? blurred : orig;
  Mat gray;
  if (isColor)
    cvtColor(patch2Process, gray, COLOR_RGB2GRAY);
  else
    gray = patch2Process.clone();
  gray.convertTo(grayD, CV_64FC1);
}

const Mat& Patch::getOrig() const noexcept {
  return orig;
}

const Mat& Patch::getBlurred() const noexcept {
  return blurred;
}

bool Patch::isColored() const noexcept {
  return isColor;
}

bool Patch::nonUniform() const noexcept {
  return needsApproximation;
}

#pragma warning(disable : WARN_THROWS_ALTHOUGH_NOEXCEPT)
const Mat& Patch::matrixToApprox() const noexcept(!UT) {
  if (!needsApproximation)
    THROW_WITH_CONST_MSG(__FUNCTION__ " should be called only "
                         "for non-uniform patches!", logic_error);
  return grayD;
}
#pragma warning(default : WARN_THROWS_ALTHOUGH_NOEXCEPT)

unique_ptr<const IPatch> Patch::clone() const noexcept {
  return make_unique<const Patch>(*this);
}
