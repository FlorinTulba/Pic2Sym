/****************************************************************************************
 The application Pic2Sym approximates images by a
 grid of colored symbols with colored backgrounds.

 This file was created on 2016-1-22
 and belongs to the Pic2Sym project.

 Copyrights from the libraries used by the program:
 - (c) 2015 Boost (www.boost.org)
   License: <http://www.boost.org/LICENSE_1_0.txt>
            or doc/licenses/Boost.lic
 - (c) 2015 The FreeType Project (www.freetype.org)
   License: <http://git.savannah.gnu.org/cgit/freetype/freetype2.git/plain/docs/FTL.TXT>
	        or doc/licenses/FTL.txt
 - (c) 2015 OpenCV (www.opencv.org)
   License: <http://opencv.org/license.html>
            or doc/licenses/OpenCV.lic
 
 (c) 2016 Florin Tulba <florintulba@yahoo.com>

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
 ****************************************************************************************/

#include "controller.h"

#include <opencv2/highgui.hpp>

using namespace std;
using namespace cv;

const Mat Comparator::noImage = imread("res/NoImage.jpg");

CvWin::CvWin(const String &winName_) : winName(winName_) {
	namedWindow(winName);
}

void CvWin::setTitle(const std::string &title) const {
	setWindowTitle(winName, title);
}

void CvWin::setOverlay(const std::string &overlay, int timeoutMs/* = 0*/) const {
	displayOverlay(winName, overlay, timeoutMs);
}

void CvWin::setStatus(const std::string &status, int timeoutMs/* = 0*/) const {
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

void Comparator::setTransparency(double transparency) {
	if(!initial.empty() && !result.empty()) {
		content.release(); // seems to be mandatory
		addWeighted(initial, transparency, result, 1.-transparency, 0., content);
	}
	imshow(winName, content);
}

void Comparator::setReference(const Mat &ref_) {
	if(ref_.empty())
		throw invalid_argument("Please provide a non-empty image to " __FUNCTION__ "!");
	initial = content = ref_;
	if(!result.empty())
		result.release();
	if(trackPos != trackMax)
		setTrackbarPos(transpTrackName, winName, trackMax);
	else
		setTransparency(1.);
}

void Comparator::setResult(const Mat &res_, int transparency/* = (int)round(defaultTransparency * trackMax)*/) {
	if(initial.empty())
		throw logic_error("Please call " __FUNCTION__ " after Comparator::setReference()!");
	if(initial.type() != res_.type() || initial.size != res_.size)
		throw invalid_argument("Please provide a resulted image of the same size & type as the original image!");
	result = res_;
	if(trackPos != transparency)
		setTrackbarPos(transpTrackName, winName, transparency);
	else
		setTransparency(defaultTransparency);
}

void Comparator::updateTransparency(int newTransp, void *userdata) {
	Comparator *pComp = reinterpret_cast<Comparator*>(userdata);
	pComp->setTransparency((double)newTransp/trackMax);
}

unsigned CmapInspect::computeSymsPerPage() const {
	const int cellSide = 1+ctrler.getFontSize();
	return (pageSz.width / cellSide) * (pageSz.height / cellSide);
}

void CmapInspect::updateGrid() {
	grid = createGrid();
}

void CmapInspect::updatePagesCount(unsigned cmapSize) {
	updatingPageMax = true;
	symsPerPage = computeSymsPerPage();
	pagesCount = (unsigned)ceil(cmapSize / (double)symsPerPage);
	setTrackbarMax(pageTrackName, winName, max(1, (int)pagesCount-1));

	// Sequence from below is required to really update the trackbar max & pos
	// The controller should prevent them to trigger CmapInspect::updatePageIdx
	setTrackbarPos(pageTrackName, winName, 1);
	updatingPageMax = false;
	setTrackbarPos(pageTrackName, winName, 0); // => page = 0
}

void CmapInspect::showPage(unsigned pageIdx) {
	// Ignore call if pageIdx isn't valid,
	// or if the required hack (mentioned in 'ui.h') provoked this call
	if(updatingPageMax || pageIdx >= pagesCount)
		return;

	if((unsigned)page != pageIdx)
		setTrackbarPos(pageTrackName, winName, pageIdx); // => page = pageIdx

	populateGrid(ctrler.getFontFaces(symsPerPage*pageIdx, symsPerPage));
	imshow(winName, content);
}

void CmapInspect::updatePageIdx(int newPage, void *userdata) {
	CmapInspect *pCmi = reinterpret_cast<CmapInspect*>(userdata);
	pCmi->showPage(newPage);
}