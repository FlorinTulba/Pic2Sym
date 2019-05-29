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

#ifndef H_TIMING_BASE
#define H_TIMING_BASE

#pragma warning(push, 0)

#include <string>
#include "misc.h"

#pragma warning(pop)

// Timing jobs:

/// Interface for any observer of a timer (see the Timer class)
class ITimerActions /*abstract*/ {
 public:
  /// Action to be performed when the timer is started
  virtual void onStart() noexcept {}

  /// action to be performed when the timer is paused
  /// @param elapsedS time elapsed this far in seconds
  virtual void onPause(double elapsedS) noexcept {
    UNREFERENCED_PARAMETER(elapsedS);
  }

  /// Action to be performed when the timer is resumed
  virtual void onResume() noexcept {}

  /// action to be performed when the timer is released/deleted
  /// @param elapsedS total elapsed time in seconds
  virtual void onRelease(double elapsedS) noexcept {
    UNREFERENCED_PARAMETER(elapsedS);
  }

  /// action to be performed when the timer is canceled
  /// @param reason explanation for cancellation
  virtual void onCancel(const std::string& reason = "") noexcept {
    UNREFERENCED_PARAMETER(reason);
  }

  virtual ~ITimerActions() noexcept {}

  // No intention to copy / move such trackers
  ITimerActions(const ITimerActions&) = delete;
  ITimerActions(ITimerActions&&) = delete;
  ITimerActions& operator=(const ITimerActions&) = delete;
  ITimerActions& operator=(ITimerActions&&) = delete;

 protected:
  constexpr ITimerActions() noexcept {}
};

/// Getting the duration of a job
class ITimerResult /*abstract*/ {
 public:
  /// Reports elapsed duration depending on valid & paused
  virtual double elapsed() const noexcept = 0;

  virtual ~ITimerResult() noexcept {}

  // Slicing prevention
  ITimerResult(const ITimerResult&) = delete;
  ITimerResult(ITimerResult&&) = delete;
  ITimerResult& operator=(const ITimerResult&) = delete;
  ITimerResult& operator=(ITimerResult&&) = delete;

 protected:
  constexpr ITimerResult() noexcept {}
};

/// Commands for an alive Timer: pause/resume and cancel
class IActiveTimer /*abstract*/ {
 public:
  /// Pauses the timer and reports duration to all observers
  virtual void pause() noexcept = 0;

  virtual void resume() noexcept = 0;  ///< resumes the timer

  /// Cancels a timing task.
  /// @param reason explanation for cancellation
  virtual void cancel(
      const std::string& reason = "The task was canceled") noexcept = 0;

  virtual ~IActiveTimer() noexcept {}

  // Slicing prevention
  IActiveTimer(const IActiveTimer&) = delete;
  IActiveTimer(IActiveTimer&&) = delete;
  IActiveTimer& operator=(const IActiveTimer&) = delete;
  IActiveTimer& operator=(IActiveTimer&&) = delete;

 protected:
  constexpr IActiveTimer() noexcept {}
};

#endif  // H_TIMING_BASE
