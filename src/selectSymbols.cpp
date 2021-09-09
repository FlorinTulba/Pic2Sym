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

#include "selectSymbols.h"

#include "cmapPerspectiveBase.h"
#include "controllerBase.h"
#include "matchEngineBase.h"
#include "pixMapSymBase.h"
#include "settingsBase.h"
#include "symDataBase.h"
#include "symsSerialization.h"

using namespace std;
using namespace std::filesystem;

namespace pic2sym {

#pragma warning(disable : WARN_REF_TO_CONST_UNIQUE_PTR)
SelectSymbols::SelectSymbols(const IController& ctrler_,
                             const match::IMatchEngine& me_,
                             const ui::ICmapPerspective& cmP_,
                             function<ui::ICmapInspect&()>&& cmiFn_) noexcept
    : ctrler(&ctrler_), me(&me_), cmP(&cmP_), cmiFn(move(cmiFn_)) {}
#pragma warning(default : WARN_REF_TO_CONST_UNIQUE_PTR)

}  // namespace pic2sym

#ifndef UNIT_TESTING

#include "appStart.h"

namespace pic2sym {

const syms::ISymData* SelectSymbols::pointedSymbol(int x,
                                                   int y) const noexcept {
  const ui::ICmapInspect& cmi = cmiFn();
  if (!cmi.isBrowsable())
    return nullptr;

  const unsigned cellSide{cmi.getCellSide()};
  const unsigned r{(unsigned)y / cellSide};
  const unsigned c{(unsigned)x / cellSide};
  const unsigned symIdx{cmi.getPageIdx() * cmi.getSymsPerPage() +
                        r * cmi.getSymsPerRow() + c};

  if (symIdx >= me->getSymsCount())
    return nullptr;

  return *cmP->getSymsRange(symIdx, 1U).begin();
}

void SelectSymbols::displaySymCode(unsigned long symCode) const noexcept {
  ostringstream oss;
  oss << " [symbol " << symCode << ']';
  ctrler->updateStatusBarCmapInspect(0U, oss.str());  // synchronous update
}

void SelectSymbols::enlistSymbolForInvestigation(
    const syms::ISymData& sd) const noexcept {
  cout << "Appending symbol " << sd.getCode()
       << " to the list needed for further investigations" << endl;

  // Enlist actual symbol, not its negative
  symsToInvestigate.push_back(255U - sd.getNegSym());
}

void SelectSymbols::symbolsReadyToInvestigate() const noexcept {
  if (symsToInvestigate.empty()) {
    cout << "The list of symbols for further investigations was empty, "
            "so there's nothing to save."
         << endl;
    return;
  }

  path destFile{AppStart::dir()};
  if (!exists(destFile.append("SymsSelections")))
    create_directory(destFile);
  destFile.append(to_string(time(nullptr))).concat(".txt");
  cout << "The list of " << size(symsToInvestigate)
       << " symbols for further investigations will be saved to file "
       << destFile << " and then cleared." << endl;

  ut::saveSymsSelection(destFile.string(), symsToInvestigate);
  symsToInvestigate.clear();
}

}  // namespace pic2sym

#endif  // UNIT_TESTING
