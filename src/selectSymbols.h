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

#ifndef H_SELECT_SYMBOLS
#define H_SELECT_SYMBOLS

#include "selectSymbolsBase.h"

#pragma warning ( push, 0 )

#include <list>
#include <memory>

#include <opencv2/core/core.hpp>

#pragma warning ( pop )

// Forward declarations
struct IController;
class MatchEngine;
class CmapPerspective;
class CmapInspect;

/// Allows saving a selection of symbols pointed within the charmap viewer
class SelectSymbols : public ISelectSymbols {
protected:
	const IController &ctrler;

	/// A list of selected symbols to investigate separately.
	/// Useful while exploring ways to filter-out various symbols from charmaps.
	mutable std::list<const cv::Mat> symsToInvestigate;

	const MatchEngine &me;

	const CmapPerspective &cmP;	///< reorganized symbols to be visualized within the cmap viewer

	const std::shared_ptr<CmapInspect> &pCmi;

public:
	SelectSymbols(const IController &ctrler_,
				  const MatchEngine &me_,
				  const CmapPerspective &cmP_,
				  const std::shared_ptr<CmapInspect> &pCmi_);

	/// Provides details about the symbol under the mouse
	const SymData* pointedSymbol(int x, int y) const override;

	/// Appends the code of the symbol under the mouse to the status bar
	void displaySymCode(unsigned long symCode) const override;

	/// Appends the matrix of the pointed symbol (by Ctrl + left click) to a list for separate investigation
	void enlistSymbolForInvestigation(const SymData &sd) const override;

	/// Saves the list with the matrices of the symbols to investigate to a file and then clears this list
	void symbolsReadyToInvestigate() const override;
};

#endif // H_SELECT_SYMBOLS
