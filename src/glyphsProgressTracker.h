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

#ifndef H_GLYPHS_PROGRESS_TRACKER
#define H_GLYPHS_PROGRESS_TRACKER

#include "controllerBase.h"
#include "timing.h"

/**
Interface to monitor the progress of loading and preprocessing a charmap.
*/
struct IGlyphsProgressTracker /*abstract*/ : virtual IController {
	/// Report progress about loading, adapting glyphs
	virtual void reportGlyphProgress(double progress) const = 0;

	/// Called when starting and ending the update of the symbol set
	virtual void symsSetUpdate(bool done = false, double elapsed = 0.) const = 0;

	/// Creates the monitor to time the glyph loading and preprocessing
	virtual Timer createTimerForGlyphs() const = 0;

	virtual ~IGlyphsProgressTracker() = 0 {}
};

#endif