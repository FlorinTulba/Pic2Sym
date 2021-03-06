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

#ifndef H_PIC_TRANSFORM_PROGRESS_TRACKER
#define H_PIC_TRANSFORM_PROGRESS_TRACKER

#include "picTransformProgressTrackerBase.h"

struct IController; // forward declaration

/// Implementation of the interface monitoring the progress of transforming an image.
class PicTransformProgressTracker : public IPicTransformProgressTracker {
protected:
	IController &ctrler;

public:
	PicTransformProgressTracker(IController &ctrler_);

	void operator=(const PicTransformProgressTracker&) = delete;

	/// Called when unable to load the symbols right when attempting to transform an image
	void transformFailedToStart() override;

	/**
	An hourglass window displays the progress [0..1] of the transformation in %.
	If showDraft is true, and a draft is available, it will be presented within Comparator window.
	*/
	void reportTransformationProgress(double progress, bool showDraft = false) const override;

	/**
	Present the partial / final result after the transformation has been canceled / has finished.
	When the transformation completes, there'll be a report about the duration of the process.
	Otherwise, completionDurationS will have its default negative value and no duration report will be issued.

	@param completionDurationS the duration of the transformation in seconds or a negative value for aborted transformations
	*/
	void presentTransformationResults(double completionDurationS = -1.) const override;

	/// Creates the monitor to time the picture approximation process
	Timer createTimerForImgTransform() const override;
};

#endif // H_PIC_TRANSFORM_PROGRESS_TRACKER
