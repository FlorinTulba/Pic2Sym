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

#ifndef H_SELECT_SYMBOLS
#define H_SELECT_SYMBOLS

#include "selectSymbolsBase.h"

#pragma warning(push, 0)

#include <list>
#include <memory>

#include <opencv2/core/core.hpp>

#pragma warning(pop)

// Forward declarations
class IController;
class IMatchEngine;
class ICmapPerspective;
class ICmapInspect;

/// Allows saving a selection of symbols pointed within the charmap viewer
class SelectSymbols : public ISelectSymbols {
 public:
  SelectSymbols(const IController& ctrler_,
                const IMatchEngine& me_,
                const ICmapPerspective& cmP_,
                const std::unique_ptr<ICmapInspect>& pCmi_) noexcept;

  /// Provides details about the symbol under the mouse
  const ISymData* pointedSymbol(int x, int y) const noexcept override;

  /// Appends the code of the symbol under the mouse to the status bar
  void displaySymCode(unsigned long symCode) const noexcept override;

  /// Appends the matrix of the pointed symbol (by Ctrl + left click) to a list
  /// for separate investigation
  void enlistSymbolForInvestigation(const ISymData& sd) const noexcept override;

  /// Saves the list with the matrices of the symbols to investigate to a file
  /// and then clears this list
  void symbolsReadyToInvestigate() const noexcept override;

 private:
  const IController& ctrler;

  /// A list of selected symbols to investigate separately.
  /// Useful while exploring ways to filter-out various symbols from charmaps.
  mutable std::list<cv::Mat> symsToInvestigate;

  const IMatchEngine& me;

  /// Reorganized symbols to be visualized within the cmap viewer
  const ICmapPerspective& cmP;

  const std::unique_ptr<ICmapInspect>& pCmi;  ///< viewer of the Cmap
};

#endif  // H_SELECT_SYMBOLS
