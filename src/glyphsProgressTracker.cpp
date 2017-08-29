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
 
 (c) 2016, 2017 Florin Tulba <florintulba@yahoo.com>

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

#include "glyphsProgressTracker.h"
#include "controllerBase.h"
#include "updateSymsActions.h"

using namespace std;

extern const stringType Controller_PREFIX_GLYPH_PROGRESS;

namespace { // Anonymous namespace
	/// Actions for start & stop chronometer while timing glyphs loading & preprocessing
	class TimerActions : public ITimerActions {
	protected:
		const IController &ctrler;

	public:
		TimerActions(const IController &ctrler_) : ctrler(ctrler_) {}
		void operator=(const TimerActions&) = delete;

		/// Action to be performed when the timer is started
		void onStart() override {
			ctrler.hourGlass(0., Controller_PREFIX_GLYPH_PROGRESS, true); // async call
		}

		/// Action to be performed when the timer is released/deleted
		/// @param elapsedS total elapsed time in seconds
		void onRelease(double elapsedS) override {
			ctrler.getGlyphsProgressTracker().updateSymsDone(elapsedS);
		}
	};
} // Anonymous namespace

GlyphsProgressTracker::GlyphsProgressTracker(const IController &ctrler_) : ctrler(ctrler_) {}

Timer GlyphsProgressTracker::createTimerForGlyphs() const {
	return Timer(std::makeShared<TimerActions>(ctrler)); // RVO
}

#ifndef UNIT_TESTING

void GlyphsProgressTracker::updateSymsDone(double durationS) const {
	ctrler.hourGlass(1., Controller_PREFIX_GLYPH_PROGRESS); // sync call
	ctrler.reportDuration("The update of the symbols set took", durationS);
}

#endif // UNIT_TESTING
