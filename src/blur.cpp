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

#include "blur.h"
#include "misc.h"
#include "warnings.h"

using namespace std;
using namespace cv;

BlurEngine::ConfiguredInstances& BlurEngine::configuredInstances() noexcept {
  static ConfiguredInstances configuredInstances_;
  return configuredInstances_;
}

BlurEngine::ConfInstRegistrator::ConfInstRegistrator(
    const string& blurType,
    const IBlurEngine& configuredInstance) noexcept {
  configuredInstances().emplace(blurType, &configuredInstance);
}

#pragma warning(disable : WARN_THROWS_ALTHOUGH_NOEXCEPT)
const IBlurEngine& BlurEngine::byName(const string& blurType) noexcept(!UT) {
  try {
    return *configuredInstances().at(blurType);
  } catch (const out_of_range&) {
    THROW_WITH_VAR_MSG("Unknown blur type: '" + blurType + "' in " __FUNCTION__,
                       invalid_argument);
  }
}
#pragma warning(default : WARN_THROWS_ALTHOUGH_NOEXCEPT)

#pragma warning(disable : WARN_THROWS_ALTHOUGH_NOEXCEPT)
void BlurEngine::process(const Mat& toBlur, Mat& blurred, bool forTinySym) const
    noexcept(!UT) {
  if (toBlur.empty() || toBlur.type() != CV_64FC1)
    THROW_WITH_CONST_MSG(__FUNCTION__ " needs toBlur to be non-empty "
                         "and to contain double-s!", invalid_argument);

  blurred = Mat(toBlur.size(), CV_64FC1, 0.);

  doProcess(toBlur, blurred, forTinySym);
}
#pragma warning(default : WARN_THROWS_ALTHOUGH_NOEXCEPT)
