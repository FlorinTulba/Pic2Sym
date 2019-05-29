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

#include "cmapPerspectiveBase.h"
#include "controllerBase.h"
#include "matchEngineBase.h"
#include "pixMapSymBase.h"
#include "selectSymbols.h"
#include "settingsBase.h"
#include "symDataBase.h"
#include "symsSerialization.h"
#include "views.h"

#pragma warning(push, 0)

#include <ctime>
#include <iostream>

#include <filesystem>

#pragma warning(pop)

using namespace std;
using namespace std::filesystem;

#pragma warning(disable : WARN_REF_TO_CONST_UNIQUE_PTR)
SelectSymbols::SelectSymbols(
    const IController& ctrler_,
    const IMatchEngine& me_,
    const ICmapPerspective& cmP_,
    const std::unique_ptr<ICmapInspect>& pCmi_) noexcept
    : ctrler(ctrler_), me(me_), cmP(cmP_), pCmi(pCmi_) {}
#pragma warning(default : WARN_REF_TO_CONST_UNIQUE_PTR)

#ifndef UNIT_TESTING

#include "appStart.h"

const ISymData* SelectSymbols::pointedSymbol(int x, int y) const noexcept {
  if (!pCmi->isBrowsable())
    return nullptr;

  const unsigned cellSide = pCmi->getCellSide(), r = (unsigned)y / cellSide,
                 c = (unsigned)x / cellSide,
                 symIdx = pCmi->getPageIdx() * pCmi->getSymsPerPage() +
                          r * pCmi->getSymsPerRow() + c;

  if (symIdx >= me.getSymsCount())
    return nullptr;

  return *cmP.getSymsRange(symIdx, 1U).first;
}

void SelectSymbols::displaySymCode(unsigned long symCode) const noexcept {
  ostringstream oss;
  oss << " [symbol " << symCode << ']';
  ctrler.updateStatusBarCmapInspect(0U, oss.str());  // synchronous update
}

void SelectSymbols::enlistSymbolForInvestigation(const ISymData& sd) const
    noexcept {
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

  path destFile = AppStart::dir();
  if (!exists(destFile.append("SymsSelections")))
    create_directory(destFile);
  destFile.append(to_string(time(nullptr))).concat(".txt");
  cout << "The list of " << symsToInvestigate.size()
       << " symbols for further investigations will be saved to file "
       << destFile << " and then cleared." << endl;

  ut::saveSymsSelection(destFile.string(), symsToInvestigate);
  symsToInvestigate.clear();
}

#endif  // UNIT_TESTING
