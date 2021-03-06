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
 If not, see <http://www.gnu.org/licenses/agpl-3.0.txt>.
 ***********************************************************************************************/

#ifndef UNIT_TESTING

#include "views.h"
#include "presentCmap.h"
#include "misc.h"

#pragma warning ( push, 0 )

#include <opencv2/highgui/highgui.hpp>

#pragma warning ( pop )

using namespace std;
using namespace cv;

const Mat Comparator::noImage = imread("res/NoImage.jpg");

CvWin::CvWin(const String &winName_) : winName(winName_) {
	namedWindow(winName);
}

void CvWin::setTitle(const std::stringType &title) const {
	setWindowTitle(winName, title);
}

void CvWin::setOverlay(const std::stringType &overlay, int timeoutMs/* = 0*/) const {
	displayOverlay(winName, overlay, timeoutMs);
}

void CvWin::setStatus(const std::stringType &status, int timeoutMs/* = 0*/) const {
	displayStatusBar(winName, status, timeoutMs);
}

void CvWin::setPos(int x, int y) const {
	moveWindow(winName, x, y);
}

void CvWin::permitResize(bool allow/* = true*/) const {
	if(allow) {
		setWindowProperty(winName, CV_WND_PROP_AUTOSIZE, CV_WINDOW_NORMAL);
	} else {
		setWindowProperty(winName, CV_WND_PROP_AUTOSIZE, CV_WINDOW_AUTOSIZE);
		setWindowProperty(winName, CV_WND_PROP_ASPECTRATIO, CV_WINDOW_KEEPRATIO);
	}
	waitKey(1); // makes sure it works
}

void CvWin::resize(int w, int h) const {
	resizeWindow(winName, w, h);
}

extern const int Comparator_trackMax;
extern const double Comparator_defaultTransparency;
extern const String Comparator_transpTrackName;

void Comparator::setTransparency(double transparency) {
	if(!initial.empty() && !result.empty()) {
		content.release(); // seems to be mandatory
		addWeighted(initial, transparency, result, 1.-transparency, 0., content);
	}
	imshow(winName, content);
}

void Comparator::setReference(const Mat &ref_) {
	if(ref_.empty())
		THROW_WITH_CONST_MSG("Please provide a non-empty image to " __FUNCTION__ "!", invalid_argument);

	initial = content = ref_;
	if(!result.empty())
		result.release();
	if(trackPos != Comparator_trackMax)
		setTrackbarPos(Comparator_transpTrackName, winName, Comparator_trackMax);
	else
		setTransparency(1.);
}

void Comparator::setResult(const Mat &res_, int transparency
						   /* = (int)round(Comparator_defaultTransparency * Comparator_trackMax)*/) {
	if(initial.empty())
		THROW_WITH_CONST_MSG("Please call " __FUNCTION__ " after Comparator::setReference()!", logic_error);

	if(initial.type() != res_.type() || initial.size != res_.size)
		THROW_WITH_CONST_MSG("Please provide a resulted image of the same size & type as the original image!", invalid_argument);

	result = res_;
	if(trackPos != transparency)
		setTrackbarPos(Comparator_transpTrackName, winName, transparency);
	else
		setTransparency(Comparator_defaultTransparency);
}

void Comparator::updateTransparency(int newTransp, void *userdata) {
	Comparator *pComp = reinterpret_cast<Comparator*>(userdata);
	pComp->setTransparency((double)newTransp/Comparator_trackMax);
}

extern const String CmapInspect_pageTrackName;

void CmapInspect::updateGrid() {
	grid = createGrid();
}

void CmapInspect::clear() {
	updateGrid();
	content = grid;
	imshow(winName, content);
	updatePagesCount(0U);
	setStatus("No Font Loaded");
	readyToBrowse = false;
}

void CmapInspect::updatePagesCount(unsigned cmapSize) {
	updatingPageMax = true;
	pagesCount = (unsigned)ceil(cmapSize / (double)symsPerPage);
	setTrackbarMax(CmapInspect_pageTrackName, winName, max(1, (int)pagesCount-1));

	// Sequence from below is required to really update the trackbar max & pos
	// The controller should prevent them to trigger CmapInspect::updatePageIdx
	setTrackbarPos(CmapInspect_pageTrackName, winName, 1);
	updatingPageMax = false;
	setTrackbarPos(CmapInspect_pageTrackName, winName, 0); // => page = 0
}

void CmapInspect::showPage(unsigned pageIdx) {
	// Ignore call if pageIdx isn't valid,
	// or if the required hack (mentioned in 'ui.h') provoked this call
	if(updatingPageMax || pageIdx >= pagesCount)
		return;

	if((unsigned)page != pageIdx)
		setTrackbarPos(CmapInspect_pageTrackName, winName, (int)pageIdx); // => page = pageIdx

	const unsigned idxOfFirstSymFromPage = symsPerPage*pageIdx;
	populateGrid(cmapPresenter.getFontFaces(idxOfFirstSymFromPage, symsPerPage),
				 cmapPresenter.getClusterOffsets(), idxOfFirstSymFromPage);
	imshow(winName, content);
}

void CmapInspect::updatePageIdx(int newPage, void *userdata) {
	// The caller ensures userdata is a `ICmapInspect*`, not a `CmapInspect*`
	ICmapInspect *pCmi = reinterpret_cast<ICmapInspect*>(userdata);
	pCmi->showPage((unsigned)newPage);
}

#endif // UNIT_TESTING not defined
