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

#ifndef H_TIMING_BASE
#define H_TIMING_BASE

#include "misc.h"

#pragma warning(push, 0)

#include <string>
#include <string_view>

#pragma warning(pop)

namespace pic2sym {

// Timing jobs:

/// Interface for any observer of a timer (see the Timer class)
class ITimerActions /*abstract*/ {
 public:
  /// Action to be performed when the timer is started
  virtual void onStart() noexcept {}

  /// action to be performed when the timer is paused
  /// @param elapsedS time elapsed this far in seconds
  virtual void onPause(double elapsedS [[maybe_unused]]) noexcept {}

  /// Action to be performed when the timer is resumed
  virtual void onResume() noexcept {}

  /// action to be performed when the timer is released/deleted
  /// @param elapsedS total elapsed time in seconds
  virtual void onRelease(double elapsedS [[maybe_unused]]) noexcept {}

  /// action to be performed when the timer is canceled
  /// @param reason explanation for cancellation
  virtual void onCancel(std::string_view reason
                        [[maybe_unused]] = "") noexcept {}

  virtual ~ITimerActions() noexcept = 0 {}
};

/// Getting the duration of a job
class ITimerResult /*abstract*/ {
 public:
  /// Reports elapsed duration depending on valid & paused
  virtual double elapsed() const noexcept = 0;

  virtual ~ITimerResult() noexcept = 0 {}
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
      std::string_view reason = "The task was canceled") noexcept = 0;

  virtual ~IActiveTimer() noexcept = 0 {}
};

}  // namespace pic2sym

#endif  // H_TIMING_BASE
