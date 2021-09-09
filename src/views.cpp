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

#ifndef UNIT_TESTING

#include "views.h"

#include "misc.h"
#include "presentCmap.h"

#pragma warning(push, 0)

#include <gsl/gsl>

#include <opencv2/highgui/highgui.hpp>

#pragma warning(pop)

using namespace std;
using namespace gsl;
using namespace cv;

namespace pic2sym {

extern const int Comparator_trackMax;
extern const double Comparator_defaultTransparency;
extern const String Comparator_transpTrackName;
extern const String CmapInspect_pageTrackName;

namespace ui {

const Mat Comparator::noImage{imread("res/NoImage.jpg")};

CvWin::CvWin(const String& winName_) noexcept : _winName(winName_) {
  namedWindow(_winName);
}

void CvWin::setTitle(const std::string& title) const noexcept {
  setWindowTitle(_winName, title);
}

void CvWin::setOverlay(const std::string& overlay,
                       int timeoutMs /* = 0*/) const noexcept {
  displayOverlay(_winName, overlay, timeoutMs);
}

void CvWin::setStatus(const std::string& status,
                      int timeoutMs /* = 0*/) const noexcept {
  displayStatusBar(_winName, status, timeoutMs);
}

void CvWin::setPos(int x, int y) const noexcept {
  moveWindow(_winName, x, y);
}

void CvWin::permitResize(bool allow /* = true*/) const noexcept {
  if (allow) {
    setWindowProperty(_winName, cv::WND_PROP_AUTOSIZE, cv::WINDOW_NORMAL);
  } else {
    setWindowProperty(_winName, cv::WND_PROP_AUTOSIZE, cv::WINDOW_AUTOSIZE);
    setWindowProperty(_winName, cv::WND_PROP_ASPECT_RATIO,
                      cv::WINDOW_KEEPRATIO);
  }
}

void CvWin::resize(int w, int h) const noexcept {
  resizeWindow(_winName, w, h);
}

void Comparator::setTransparency(double transparency) noexcept {
  if (!initial.empty() && !result.empty()) {
    content().release();  // seems to be mandatory
    addWeighted(initial, transparency, result, 1. - transparency, 0.,
                content());
  }
  imshow(winName(), content());
}

void Comparator::setReference(const Mat& ref_) noexcept {
  Expects(!ref_.empty());  // provide a non-empty image

  initial = content() = ref_;
  if (!result.empty())
    result.release();
  if (trackPos != Comparator_trackMax)
    setTrackbarPos(Comparator_transpTrackName, winName(), Comparator_trackMax);
  else
    setTransparency(1.);
}

void Comparator::setResult(
    const Mat& res_,
    int transparency
    /* = (int)round(Comparator_defaultTransparency * Comparator_trackMax)*/) noexcept {
  Expects(!initial.empty());  // called before Comparator::setReference()

  // provide a resulted image of the same size & type as the original image
  Expects(initial.type() == res_.type() && initial.size == res_.size);
  result = res_;
  if (trackPos != transparency)
    setTrackbarPos(Comparator_transpTrackName, winName(), transparency);
  else
    setTransparency(Comparator_defaultTransparency);
}

void Comparator::updateTransparency(int newTransp, void* userdata) noexcept {
  not_null<Comparator*> pComp = static_cast<Comparator*>(userdata);
  pComp->setTransparency((double)newTransp / Comparator_trackMax);
}

void CmapInspect::updateGrid() noexcept {
  grid = createGrid();
}

void CmapInspect::clear() noexcept {
  updateGrid();
  content(grid);
  imshow(winName(), content());
  updatePagesCount(0U);
  setStatus("No Font Loaded");
  readyToBrowse = false;
}

void CmapInspect::updatePagesCount(unsigned cmapSize) noexcept {
  updatingPageMax = true;
  pagesCount = narrow_cast<unsigned>(ceil(cmapSize / (double)symsPerPage));
  setTrackbarMax(CmapInspect_pageTrackName, winName(),
                 max(1, (int)pagesCount - 1));

  // Sequence from below is required to really update the trackbar max & pos
  // The controller should prevent them to trigger CmapInspect::updatePageIdx
  setTrackbarPos(CmapInspect_pageTrackName, winName(), 1);
  updatingPageMax = false;
  setTrackbarPos(CmapInspect_pageTrackName, winName(), 0);  // => page = 0
}

void CmapInspect::showPage(unsigned pageIdx) noexcept {
  // Ignore call if pageIdx isn't valid,
  // or if the required hack (mentioned in 'views.h') provoked this call
  if (updatingPageMax || pageIdx >= pagesCount)
    return;

  if ((unsigned)page != pageIdx)
    setTrackbarPos(CmapInspect_pageTrackName, winName(),
                   (int)pageIdx);  // => page = pageIdx

  const unsigned idxOfFirstSymFromPage{symsPerPage * pageIdx};
  populateGrid(cmapPresenter->getFontFaces(idxOfFirstSymFromPage, symsPerPage),
               cmapPresenter->getClusterOffsets(), idxOfFirstSymFromPage);
  imshow(winName(), content());
}

void CmapInspect::updatePageIdx(int newPage, void* userdata) noexcept {
  // The caller ensures userdata is a `ICmapInspect*`, not a `CmapInspect*`
  not_null<ICmapInspect*> pCmi = static_cast<ICmapInspect*>(userdata);
  pCmi->showPage((unsigned)newPage);
}

}  // namespace ui
}  // namespace pic2sym

#endif  // UNIT_TESTING not defined
