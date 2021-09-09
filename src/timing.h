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

#ifndef H_TIMING
#define H_TIMING

#include "timingBase.h"

#pragma warning(push, 0)

#include <chrono>
#include <memory>
#include <vector>

#pragma warning(pop)

extern template class std::chrono::time_point<
    std::chrono::high_resolution_clock>;

namespace pic2sym {

/**
ActiveTimer class:
- realization of IActiveTimer
- one of the 2 base classes of Timer

Timer becomes a `Concentrator class` if it realizes alone both IActiveTimer and
ITimerResult.
*/
class ActiveTimer /*abstract*/ : public IActiveTimer {
 public:
  /// If not canceled / released, reports duration to all observers
  ~ActiveTimer() noexcept override;

  // Slicing prevention
  ActiveTimer(const ActiveTimer&) = delete;
  ActiveTimer(ActiveTimer&&) = delete;
  void operator=(const ActiveTimer&) = delete;
  void operator=(ActiveTimer&&) = delete;

  void invalidate() noexcept;  ///< prevents further use of this timer

  /// Stops the timer and reports duration to all observers
  virtual void release() noexcept;

  /// Pauses the timer and reports duration to all observers
  void pause() noexcept override;

  void resume() noexcept override;  ///< resumes the timer

  /// Cancels a timing task.
  /// @param reason explanation for cancellation
  void cancel(
      std::string_view reason = "The task was canceled") noexcept override;

 protected:
  /// Initializes lastStart and notifies all observers
  explicit ActiveTimer(
      const std::vector<std::shared_ptr<ITimerActions>>& observers_) noexcept;

  /// Initializes lastStart and notifies the observer
  explicit ActiveTimer(std::shared_ptr<ITimerActions> observer) noexcept;

  bool valid() const noexcept { return _valid; }
  bool paused() const noexcept { return _paused; }

  /// sum of previous intervals, when repeatedly paused and resumed
  std::chrono::duration<double> elapsedS() const noexcept { return _elapsedS; }

  /// The moment when computation started / was resumed last time
  std::chrono::time_point<std::chrono::high_resolution_clock> lastStart()
      const noexcept {
    return _lastStart;
  }

 private:
  /// Observers to be notified, which outlive this Timer or are `kept alive`
  /// until its destruction (due to shared_ptr)
  std::vector<std::shared_ptr<ITimerActions>> observers;

  /// The moment when computation started / was resumed last time
  std::chrono::time_point<std::chrono::high_resolution_clock> _lastStart;

  /// Sum of previous intervals, when repeatedly paused and resumed
  std::chrono::duration<double> _elapsedS;

  bool _paused{false};  ///< true as long as not paused
  bool _valid{true};    ///< true as long as not canceled / released
};

/// Timer class
class Timer : public ActiveTimer, public ITimerResult {
 public:
  /// Initializes lastStart and notifies all observers
  explicit Timer(
      const std::vector<std::shared_ptr<ITimerActions>>& observers_) noexcept;

  /// Initializes lastStart and notifies the observer
  explicit Timer(std::shared_ptr<ITimerActions> observer) noexcept;

  // Slicing prevention
  Timer(const Timer&) = delete;
  Timer(Timer&&) = delete;
  void operator=(const Timer&) = delete;
  void operator=(Timer&&) = delete;

  /// Reports elapsed duration depending on valid & paused
  double elapsed() const noexcept override;
};

}  // namespace pic2sym

#endif  // H_TIMING
