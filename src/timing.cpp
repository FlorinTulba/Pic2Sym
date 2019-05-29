/******************************************************************************
 The application Pic2Sym approximates images by a
 grid of colored symbols with colored backgrounds.

 Copyrights from the libraries used by the program:
 - (c) 2003 Boost (www.boost.org)
     License: doc/licenses/Boost.lic
     http://www.boost.org/LICENSE_1_0.txt
 - (c) 2015-2016 OpenCV (www.opencv.org)
     License: doc/licenses/OpenCV.lic
     http://opencv.org/license/
 - (c) 1996-2002, 2006 The FreeType Project (www.freetype.org)
     License: doc/licenses/FTL.txt
     http://git.savannah.gnu.org/cgit/freetype/freetype2.git/plain/docs/FTL.TXT
 - (c) 1997-2002 OpenMP Architecture Review Board (www.openmp.org)
   (c) Microsoft Corporation (implementation for OpenMP C/C++ v2.0 March 2002)
     See: https://msdn.microsoft.com/en-us/library/8y6825x5.aspx
 - (c) 1995-2017 zlib software (Jean-loup Gailly and Mark Adler - www.zlib.net)
     License: doc/licenses/zlib.lic
     http://www.zlib.net/zlib_license.html


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
 If not, see: http://www.gnu.org/licenses/agpl-3.0.txt .
 *****************************************************************************/

#include "precompiled.h"

#include "timing.h"

using namespace std;
using namespace std::chrono;

ActiveTimer::ActiveTimer(
    const vector<std::shared_ptr<ITimerActions>>& observers_) noexcept
    : observers(observers_),
      _lastStart(high_resolution_clock::now()),
      _elapsedS(0.) {
  for (shared_ptr<ITimerActions> observer : observers)
    observer->onStart();
}

ActiveTimer::ActiveTimer(std::shared_ptr<ITimerActions> observer) noexcept
    : ActiveTimer(vector<std::shared_ptr<ITimerActions>>{observer}) {}

ActiveTimer::~ActiveTimer() noexcept {
  if (!_valid)
    return;

  release();
}

void ActiveTimer::invalidate() noexcept {
  _valid = false;
}

void ActiveTimer::cancel(
    const string& reason /* = "The task was canceled"*/) noexcept {
  if (!_valid)
    return;

  _valid = false;

  for (shared_ptr<ITimerActions> observer : observers)
    observer->onCancel(reason);
}

void ActiveTimer::pause() noexcept {
  if (!_valid || _paused)
    return;

  _elapsedS += high_resolution_clock::now() - _lastStart;

  _paused = true;

  for (shared_ptr<ITimerActions> observer : observers)
    observer->onPause(_elapsedS.count());
}

void ActiveTimer::resume() noexcept {
  if (!_valid || !_paused)
    return;

  _paused = false;

  _lastStart = high_resolution_clock::now();

  for (shared_ptr<ITimerActions> observer : observers)
    observer->onResume();
}

void ActiveTimer::release() noexcept {
  if (!_valid)
    return;

  _valid = false;

  if (!_paused)
    _elapsedS += high_resolution_clock::now() - _lastStart;

  for (shared_ptr<ITimerActions> observer : observers)
    observer->onRelease(_elapsedS.count());
}

Timer::Timer(const vector<std::shared_ptr<ITimerActions>>& observers_) noexcept
    : ActiveTimer(observers_) {}

Timer::Timer(std::shared_ptr<ITimerActions> observer) noexcept
    : ActiveTimer(observer) {}

double Timer::elapsed() const noexcept {
  if (!valid())
    return 0.;

  if (paused())
    return elapsedS().count();

  auto durationToReport = elapsedS();
  durationToReport += high_resolution_clock::now() - lastStart();

  return durationToReport.count();
}
