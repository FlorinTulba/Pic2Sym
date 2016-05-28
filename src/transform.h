/****************************************************************************************
 The application Pic2Sym approximates images by a
 grid of colored symbols with colored backgrounds.

 This file belongs to the Pic2Sym project.

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

#ifndef H_TRANSFORM
#define H_TRANSFORM

#include "matchEngine.h"
#include "img.h"

#include <string>
#include <vector>

#include <opencv2/imgcodecs.hpp>

class Settings;		// global settings
struct IPicTransformProgressTracker;	// data & views manager

/// Transformer allows images to be approximated as a table of colored symbols from font files.
class Transformer {
protected:
	const IPicTransformProgressTracker &ctrler;	///< data & views manager

	const Settings &cfg;		///< general configuration
	MatchEngine &me;			///< approximating patches
	Img &img;					///< current image to process

	cv::Mat result;				///< the result of the transformation
	
	const std::string textStudiedCase(int rows, int cols) const;

	void createOutputFolder();

public:
	Transformer(const IPicTransformProgressTracker &ctrler_, const Settings &cfg_,
				MatchEngine &me_, Img &img_);

	void run();	///< applies the configured transformation onto current/new image

	const cv::Mat& getResult() const { return result; }
};

#endif