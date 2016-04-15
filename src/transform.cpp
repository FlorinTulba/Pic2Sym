/****************************************************************************************
 The application Pic2Sym approximates images by a
 grid of colored symbols with colored backgrounds.

 This file was created on 2016-1-8
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

#include "transform.h"
#include "picTransformProgressTracker.h"
#include "misc.h"
#include "timing.h"
#include "settings.h"
#include "matchSettingsManip.h"
#include "matchParams.h"
#include "patch.h"
#include "transformTrace.h"

#include <sstream>
#include <numeric>

#include <boost/filesystem/operations.hpp>

using namespace std;
using namespace boost::filesystem;
using namespace cv;

Transformer::Transformer(const IPicTransformProgressTracker &ctrler_, const Settings &cfg_,
						 MatchEngine &me_, Img &img_) :
		ctrler(ctrler_), cfg(cfg_), me(me_), img(img_) {
	createOutputFolder();
}

void Transformer::run() {
	me.updateSymbols(); // throws for invalid cmap/size

	// throws when no image
	const ResizedImg resizedImg(img, cfg.imgSettings(), cfg.symSettings().getFontSz());
	const_cast<IPicTransformProgressTracker&>(ctrler).updateResizedImg(resizedImg);
	const Mat &resized = resizedImg.get();
	const bool isColor = img.isColor();
	
	// keep this after ResizedImg, to display updated resized version as comparing image
	Timer timer = ctrler.createTimerForImgTransform();

	const string &studiedCase = textStudiedCase(resized.rows, resized.cols);

#ifndef UNIT_TESTING
	path resultFile(MatchSettingsManip::instance().getWorkDir());
	resultFile.append("Output").append(studiedCase).
		concat(".jpg");
	// generating a JPG result file (minor quality loss, but significant space requirements reduction)

	if(exists(resultFile)) {
		result = imread(resultFile.string(), ImreadModes::IMREAD_UNCHANGED);
		timer.release();
		ctrler.reportTransformationProgress(1.);

		infoMsg("This image has already been transformed under these settings.\n"
				"Displaying the available result");
		return;
	}
#endif

	me.getReady();

	const unsigned sz = cfg.symSettings().getFontSz();
	TransformTrace tt(studiedCase, sz, me.usesUnicode()); // log support (DEBUG mode only)

	result = Mat(resized.rows, resized.cols, resized.type());
	Mat resizedBlurred;
	extern const Size BlurWinSize;
	extern const double BlurStandardDeviation;
	GaussianBlur(resized, resizedBlurred, BlurWinSize, BlurStandardDeviation, 0., BORDER_REPLICATE);

	for(unsigned r = 0U, h = (unsigned)resized.rows; r<h; r += sz) {
		ctrler.reportTransformationProgress((double)r/h);

		const Range rowRange(r, r+sz);

		for(unsigned c = 0U, w = (unsigned)resized.cols; c<w; c += sz) {
			const Range colRange(c, c+sz);
			const Mat patch(resized, rowRange, colRange),
						blurredPatch(resizedBlurred, rowRange, colRange);

			// Building a Patch with the blurred patch computed for its actual borders
			Patch p(patch, blurredPatch, isColor);
			const BestMatch best = me.approxPatch(p);
			const Mat &approximation = best.bestVariant.approx;
			Mat destRegion(result, rowRange, colRange);
			approximation.copyTo(destRegion);

			tt.newEntry(r, c, best); // log the data about best match (DEBUG mode only)
		}
	}

#ifndef UNIT_TESTING
	timer.pause();

	cout<<"Writing result to "<<resultFile<<endl<<endl;
	imwrite(resultFile.string(), result);

	timer.resume();
#endif
}

#ifndef UNIT_TESTING // Unit Testing module has different implementations for these methods
void Transformer::createOutputFolder() {
	// Ensure there is an Output folder
	path outputFolder = MatchSettingsManip::instance().getWorkDir();
	if(!exists(outputFolder.append("Output")))
		create_directory(outputFolder);
}
#endif