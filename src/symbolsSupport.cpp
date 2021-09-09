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

#include "symbolsSupport.h"

using namespace std;
using namespace gsl;
using namespace cv;

namespace pic2sym::syms {

bool SymsSupport::usingTinySymbols() const noexcept {
  return false;
}

void SymsSupport::computeClusterRepresentative(
    const vector<not_null<const ISymData*>>& clusterSyms,
    int symSz,
    double invClusterSz,
    Mat& synthesizedSym,
    Mat& negSym) const noexcept {
  Mat negSynthesizedSym{symSz, symSz, CV_64FC1, Scalar{}};
  for (not_null<const ISymData*> pSymData : clusterSyms) {
    // normal-size symbol is guaranteed to be non-blank
    Expects(!pSymData->getNegSym().empty());
    Mat negSymD;
    pSymData->getNegSym().convertTo(negSymD, CV_64FC1);
    negSynthesizedSym += negSymD;
  }
  negSynthesizedSym *= invClusterSz;

  // cluster representatives for normal symbols have negSym of type byte
  negSynthesizedSym.convertTo(negSym, CV_8UC1);

  // providing a symbol in 0..1 range
  synthesizedSym = 1. - negSynthesizedSym * Inv255;
}

}  // namespace pic2sym::syms
