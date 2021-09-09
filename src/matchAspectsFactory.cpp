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

#include "matchAspectsFactory.h"

#include "correlationAspect.h"
#include "matchAspects.h"
#include "misc.h"
#include "structuralSimilarity.h"

using namespace std;
using namespace gsl;

namespace pic2sym::match {

owner<const MatchAspect*> MatchAspectsFactory::create(
    string_view aspectName,
    const p2s::cfg::IMatchSettings& ms) {
#define HANDLE_MATCH_ASPECT(Aspect) \
  if (aspectName == #Aspect)        \
    return owner<const Aspect*> {   \
      new Aspect { ms }             \
    }
  // return make_unique<Aspect>(ms) won't work above, because Aspect ctor are
  // protected and only MatchAspectsFactory is their friend, while make_unique
  // is not

  HANDLE_MATCH_ASPECT(StructuralSimilarity);
  HANDLE_MATCH_ASPECT(CorrelationAspect);
  HANDLE_MATCH_ASPECT(FgMatch);
  HANDLE_MATCH_ASPECT(BgMatch);
  HANDLE_MATCH_ASPECT(EdgeMatch);
  HANDLE_MATCH_ASPECT(BetterContrast);
  HANDLE_MATCH_ASPECT(GravitationalSmoothness);
  HANDLE_MATCH_ASPECT(DirectionalSmoothness);
  HANDLE_MATCH_ASPECT(LargerSym);

#undef HANDLE_ASPECT

  reportAndThrow<domain_error>(HERE.function_name() +
                               " not handling aspect named '"s +
                               string{aspectName} + "'!"s);
  return nullptr;  // avoids warning about code paths not returning
}

}  // namespace pic2sym::match
