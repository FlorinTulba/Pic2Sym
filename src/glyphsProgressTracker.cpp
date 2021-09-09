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

#include "glyphsProgressTracker.h"

using namespace std;
using namespace gsl;

namespace pic2sym {

extern const string Controller_PREFIX_GLYPH_PROGRESS;

namespace {  // Anonymous namespace
/// Actions for start & stop chronometer while timing glyphs loading &
/// preprocessing
class TimerActions : public ITimerActions {
 public:
  explicit TimerActions(const IController& ctrler_) noexcept
      : ctrler(&ctrler_) {}
  ~TimerActions() noexcept override = default;

  // Prevent proliferation of such listeners
  TimerActions(const TimerActions&) = delete;
  TimerActions(TimerActions&&) = delete;

  // 'ctrler' is supposed not to change
  void operator=(const TimerActions&) = delete;
  void operator=(TimerActions&&) = delete;

  /// Action to be performed when the timer is started
  void onStart() noexcept override {
    ctrler->hourGlass(0., Controller_PREFIX_GLYPH_PROGRESS,
                      true);  // async call
  }

  /// Action to be performed when the timer is released/deleted
  /// @param elapsedS total elapsed time in seconds
  void onRelease(double elapsedS) noexcept override {
    ctrler->getGlyphsProgressTracker().updateSymsDone(elapsedS);
  }

 private:
  not_null<const IController*> ctrler;
};
}  // Anonymous namespace

GlyphsProgressTracker::GlyphsProgressTracker(
    const IController& ctrler_) noexcept
    : ctrler(&ctrler_) {}

Timer GlyphsProgressTracker::createTimerForGlyphs() const noexcept {
  return Timer{std::make_shared<TimerActions>(*ctrler)};
}

#ifndef UNIT_TESTING

void GlyphsProgressTracker::updateSymsDone(double durationS) const noexcept {
  ctrler->hourGlass(1., Controller_PREFIX_GLYPH_PROGRESS);  // sync call
  ctrler->reportDuration("The update of the symbols set took", durationS);
}

#endif  // UNIT_TESTING

}  // namespace pic2sym
