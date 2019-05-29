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

#ifndef H_TASK_MONITOR_BASE
#define H_TASK_MONITOR_BASE

#pragma warning(push, 0)

#include <string>

#pragma warning(pop)

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

  /// Total steps required to finish the activity
  virtual void setTotalSteps(size_t totalSteps_) noexcept = 0;

  /**
  Task performer reports its progress
  @throw logic_error if called before setTotalSteps()

  Exception to be only reported, not handled
  */
  virtual void taskAdvanced(size_t steps = 1U) noexcept(!UT) = 0;

  /// Task performer reports finishing this activity
  virtual void taskDone() noexcept = 0;

  /// Task performer reports that the activity was aborted
  virtual void taskAborted() noexcept = 0;

 protected:
  explicit AbsTaskMonitor(const std::string& monitoredTask) noexcept
      : _monitoredTask(monitoredTask) {}

 private:
  const std::string _monitoredTask;  ///< name of the monitored task
};

#endif  // H_TASK_MONITOR_BASE
