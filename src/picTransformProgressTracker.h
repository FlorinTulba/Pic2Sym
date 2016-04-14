/****************************************************************************************
 The application Pic2Sym approximates images by a
 grid of colored symbols with colored backgrounds.

 This file was created on 2016-4-14
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

#ifndef H_PIC_TRANSFORM_PROGRESS_TRACKER
#define H_PIC_TRANSFORM_PROGRESS_TRACKER

#include "img.h"
#include "controllerBase.h"
#include "misc.h"

/**
Interface to monitor the progress of loading and preprocessing a charmap.
*/
struct IPicTransformProgressTracker /*abstract*/ : virtual IController {
	virtual void updateResizedImg(const ResizedImg &resizedImg_) = 0;

	virtual void reportTransformationProgress(double progress) const = 0;
	
	/// Called by TimerActions_ImgTransform from below when starting and ending the image transformation
	virtual void imgTransform(bool done = false, double elapsed = 0.) const = 0;

	/// Creates the monitor to time the picture approximation process
	virtual Timer createTimerForImgTransform() const = 0;

	virtual ~IPicTransformProgressTracker() = 0 {}
};

#endif