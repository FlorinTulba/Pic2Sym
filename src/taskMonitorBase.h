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

#ifndef H_TASK_MONITOR_BASE
#define H_TASK_MONITOR_BASE

#include "misc.h"

#pragma warning(push, 0)

#include <stdexcept>
#include <string>

#pragma warning(pop)

namespace pic2sym::ui {

/// Exception thrown when a method cannot be applied to an aborted job / subtask
class AbortedJob : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

/// Abstract class for monitoring progress of a specific activity within a given
/// job
class AbsTaskMonitor /*abstract*/ {
 public:
  // No intention to copy / move such trackers
  AbsTaskMonitor(const AbsTaskMonitor&) = delete;
  AbsTaskMonitor(AbsTaskMonitor&&) = delete;
  void operator=(const AbsTaskMonitor&) = delete;
  void operator=(AbsTaskMonitor&&) = delete;

  virtual ~AbsTaskMonitor() noexcept {}

  /// Name of the activity
  const std::string& monitoredTask() const noexcept { return _monitoredTask; }

  /// Total steps required to finish the activity. Throws AbortedJob if the
  /// parent job was aborted
  virtual void setTotalSteps(size_t totalSteps_) = 0;

  /**
  Task performer reports its progress
  @throw logic_error if called before setTotalSteps()
  Exception to be only reported, not handled

  @throw AbortedJob if the parent job was aborted.
  This exception must be handled
  */
  virtual void taskAdvanced(size_t steps = 1U) = 0;

  /// Task performer reports finishing this activity. Throws AbortedJob if the
  /// parent job was aborted
  virtual void taskDone() = 0;

  /// Task performer reports that the activity was aborted
  virtual void taskAborted() noexcept = 0;

 protected:
  explicit AbsTaskMonitor(const std::string& monitoredTask) noexcept
      : _monitoredTask(monitoredTask) {}

 private:
  std::string _monitoredTask;  ///< name of the monitored task
};

}  // namespace pic2sym::ui

#endif  // H_TASK_MONITOR_BASE
