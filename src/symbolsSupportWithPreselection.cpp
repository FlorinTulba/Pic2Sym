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

#include "misc.h"
#include "symDataBase.h"
#include "symbolsSupportWithPreselection.h"

using namespace std;
using namespace cv;

bool SymsSupportWithPreselection::usingTinySymbols() const noexcept {
  return true;
}

void SymsSupportWithPreselection::computeClusterRepresentative(
    const vector<const ISymData*>& clusterSyms,
    int symSz,
    double invClusterSz,
    Mat& synthesizedSym,
    Mat& negSym) const noexcept {
  Mat negSynthesizedSym(symSz, symSz, CV_64FC1, Scalar(0.));
  for (const ISymData* pSymData : clusterSyms) {
    if (!pSymData->getNegSym().empty())
      // A non-blank normal-size symbol can become a blank when shrunken
      negSynthesizedSym += pSymData->getNegSym();
  }
  negSynthesizedSym *= invClusterSz;

  // cluster representatives for tiny symbols have negSym of type double
  negSym = negSynthesizedSym;

  // providing a symbol in 0..1 range
  synthesizedSym = 1. - negSynthesizedSym * INV_255;
}
