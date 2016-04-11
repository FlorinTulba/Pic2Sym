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

#include "controller.h"

#include "misc.h"
#include "matchSettingsManip.h"
#include "settings.h"
#include "matchParams.h"

#include <sstream>
#include <numeric>

#if defined _DEBUG && !defined UNIT_TESTING
#	include <fstream>
#endif

#include <boost/filesystem/operations.hpp>

using namespace std;
using namespace boost::filesystem;

Transformer::Transformer(const Controller &ctrler_, const Settings &cfg_, MatchEngine &me_, Img &img_) :
		ctrler(ctrler_), cfg(cfg_), me(me_), img(img_) {
	createOutputFolder();
}

void Transformer::run() {
	me.updateSymbols(); // throws for invalid cmap/size

	// throws when no image
	const ResizedImg resizedImg(img, cfg.imgSettings(), cfg.symSettings().getFontSz());
	const_cast<Controller&>(ctrler).updateResizedImg(resizedImg);
	const cv::Mat &resized = resizedImg.get();
	
	// keep it after ResizedImg, to display updated resized version as comparing image
	Controller::Timer timer(ctrler, Controller::Timer::ComputationType::IMG_TRANSFORM);

	const string &studiedCase = textStudiedCase(resized.rows, resized.cols);

#ifndef UNIT_TESTING
	path resultFile(MatchSettingsManip::instance().getWorkDir());
	resultFile.append("Output").append(studiedCase).
		concat(".jpg");
	// generating a JPG result file (minor quality loss, but significant space requirements reduction)

	if(exists(resultFile)) {
		result = cv::imread(resultFile.string(), cv::ImreadModes::IMREAD_UNCHANGED);
		timer.release();
		ctrler.reportTransformationProgress(1.);

		infoMsg("This image has already been transformed under these settings.\n"
				"Displaying the available result");
		return;
	}
#endif

	me.getReady();

#if defined _DEBUG && !defined UNIT_TESTING
	static const wstring COMMA(L",\t");
	path traceFile(MatchSettingsManip::instance().getWorkDir());
	traceFile.append("data_").concat(studiedCase).
		concat(".csv"); // generating a CSV trace file
	wofstream ofs(traceFile.c_str());
	ofs<<"#Row"<<COMMA<<"#Col"<<COMMA<<BestMatch::HEADER<<endl;

	// Unicode symbols are logged in symbol format, while other encodings log their code
	const bool isUnicode = me.usesUnicode();
#endif
	
	const unsigned sz = cfg.symSettings().getFontSz();
	result = cv::Mat(resized.rows, resized.cols, resized.type());

	cv::Mat resizedBlurred;
	cv::GaussianBlur(resized, resizedBlurred,
					 StructuralSimilarity::WIN_SIZE, StructuralSimilarity::SIGMA, 0.,
					 cv::BORDER_REPLICATE);

	for(unsigned r = 0U, h = (unsigned)resized.rows; r<h; r += sz) {
		ctrler.reportTransformationProgress((double)r/h);

		const cv::Range rowRange(r, r+sz);

		for(unsigned c = 0U, w = (unsigned)resized.cols; c<w; c += sz) {
			const cv::Range colRange(c, c+sz);
			const cv::Mat patch(resized, rowRange, colRange),
						blurredPatch(resizedBlurred, rowRange, colRange),
						destRegion(result, rowRange, colRange);

			// Don't approximate rather uniform patches
			double minVal, maxVal;
			cv::Mat grayBlurredPatch;
			if(img.isColor())
				cv::cvtColor(blurredPatch, grayBlurredPatch, cv::COLOR_RGB2GRAY);
			else
				grayBlurredPatch = blurredPatch;

			cv::minMaxIdx(grayBlurredPatch, &minVal, &maxVal); // assessed on blurred patch, to avoid outliers bias
			extern const double Transformer_run_THRESHOLD_CONTRAST_BLURRED;
			if(maxVal-minVal < Transformer_run_THRESHOLD_CONTRAST_BLURRED) {
				blurredPatch.copyTo(destRegion);
				continue;
			}

			// The patch is less uniform, so it needs approximation
#if defined _DEBUG && !defined UNIT_TESTING
			BestMatch best(isUnicode);
#else
			BestMatch best;
#endif

			const cv::Mat approximation = me.approxPatch(patch, best);

			// For non-hybrid result mode, just copy the approximation to the result
			if(!cfg.matchSettings().isHybridResult()) {
				approximation.copyTo(destRegion);
				continue;
			}

			// Hybrid Result Mode - Combine the approximation with the blurred patch:
			// the less satisfactory the approximation is,
			// the more the weight of the blurred patch should be
			cv::Scalar miu, sdevApproximation, sdevBlurredPatch;
			meanStdDev(patch-approximation, miu, sdevApproximation);
			meanStdDev(patch-blurredPatch, miu, sdevBlurredPatch);
			double totalSdevBlurredPatch = *sdevBlurredPatch.val,
					totalSdevApproximation = *sdevApproximation.val;
			if(img.isColor()) {
				totalSdevBlurredPatch += sdevBlurredPatch.val[1] + sdevBlurredPatch.val[2];
				totalSdevApproximation += sdevApproximation.val[1] + sdevApproximation.val[2];
			}
			const double sdevSum = totalSdevBlurredPatch + totalSdevApproximation;
			const double weight = (sdevSum > 0.) ? (totalSdevApproximation / sdevSum) : 0.;
			cv::Mat combination;
			addWeighted(blurredPatch, weight, approximation, 1.-weight, 0., combination);
			combination.copyTo(destRegion);

#if defined _DEBUG && !defined UNIT_TESTING
			ofs<<r/sz<<COMMA<<c/sz<<COMMA<<best<<endl;
#endif
		}
#if defined _DEBUG && !defined UNIT_TESTING
		ofs.flush(); // flush after processing a full row (of height sz) of the image
#endif
	}

#if defined _DEBUG && !defined UNIT_TESTING
	// Flushing and closing the trace file, to be also ready when inspecting the resulted image
	ofs.close();
#endif

#ifndef UNIT_TESTING
	cout<<"Writing result to "<<resultFile<<endl<<endl;
	imwrite(resultFile.string(), result);
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