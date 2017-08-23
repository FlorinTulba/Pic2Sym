/************************************************************************************************
 The application Pic2Sym approximates images by a
 grid of colored symbols with colored backgrounds.

 Copyrights from the libraries used by the program:
 - (c) 2016 Boost (www.boost.org)
		License: <http://www.boost.org/LICENSE_1_0.txt>
			or doc/licenses/Boost.lic
 - (c) 2015 OpenCV (www.opencv.org)
		License: <http://opencv.org/license.html>
            or doc/licenses/OpenCV.lic
 - (c) 2015 The FreeType Project (www.freetype.org)
		License: <http://git.savannah.gnu.org/cgit/freetype/freetype2.git/plain/docs/FTL.TXT>
	        or doc/licenses/FTL.txt
 - (c) 1997-2002 OpenMP Architecture Review Board (www.openmp.org)
   (c) Microsoft Corporation (Visual C++ implementation for OpenMP C/C++ Version 2.0 March 2002)
		See: <https://msdn.microsoft.com/en-us/library/8y6825x5(v=vs.140).aspx>
 - (c) 1995-2013 zlib software (Jean-loup Gailly and Mark Adler - see: www.zlib.net)
		License: <http://www.zlib.net/zlib_license.html>
            or doc/licenses/zlib.lic
 
 (c) 2016, 2017 Florin Tulba <florintulba@yahoo.com>

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
 If not, see <http://www.gnu.org/licenses/agpl-3.0.txt>.
 ***********************************************************************************************/

#include "selectSymbols.h"
#include "controllerBase.h"
#include "appStart.h"
#include "pixMapSymBase.h" 
#include "symDataBase.h"
#include "matchEngine.h"
#include "settingsBase.h"
#include "symsSerialization.h"
#include "cmapPerspective.h"
#include "views.h"

#pragma warning ( push, 0 )

#include <iostream>
#include <ctime>

#include "boost_filesystem_operations.h"

#pragma warning ( pop )

using namespace std;
using namespace boost::filesystem;

SelectSymbols::SelectSymbols(const IController &ctrler_,
							 const MatchEngine &me_,
							 const CmapPerspective &cmP_,
							 const std::sharedPtr<CmapInspect> &pCmi_) :
	ctrler(ctrler_), me(me_), cmP(cmP_), pCmi(pCmi_) {}

#ifndef UNIT_TESTING

const ISymData* SelectSymbols::pointedSymbol(int x, int y) const {
	if(!pCmi->isBrowsable())
		return nullptr;

	const unsigned cellSide = pCmi->getCellSide(),
		r = (unsigned)y / cellSide, c = (unsigned)x / cellSide,
		symIdx = pCmi->getPageIdx()*pCmi->getSymsPerPage() + r*pCmi->getSymsPerRow() + c;

	if(symIdx >= me.getSymsCount())
		return nullptr;

	return *cmP.getSymsRange(symIdx, 1U).first;
}

void SelectSymbols::displaySymCode(unsigned long symCode) const {
	ostringstream oss;
	oss<<" [symbol "<<symCode<<']';
	ctrler.updateStatusBarCmapInspect(0U, oss.str()); // synchronous update
}

void SelectSymbols::enlistSymbolForInvestigation(const ISymData &sd) const {
	cout<<"Appending symbol "<<sd.getCode()<<" to the list needed for further investigations"<<endl;
	symsToInvestigate.push_back(255U - sd.getNegSym()); // enlist actual symbol, not its negative
}

void SelectSymbols::symbolsReadyToInvestigate() const {
	if(symsToInvestigate.empty()) {
		cout<<"The list of symbols for further investigations was empty, so there's nothing to save."<<endl;
		return;
	}

	path destFile = AppStart::dir();
	if(!exists(destFile.append("SymsSelections")))
		create_directory(destFile);
	destFile.append(to_string(time(nullptr))).concat(".txt");
	cout<<"The list of "<<symsToInvestigate.size()<<" symbols for further investigations will be saved to file "
		<<destFile<<" and then cleared."<<endl;

	ut::saveSymsSelection(destFile.string(), symsToInvestigate);
	symsToInvestigate.clear();
}

#endif // UNIT_TESTING
