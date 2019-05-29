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

#ifndef H_JOB_MONITOR_BASE
#define H_JOB_MONITOR_BASE

#include "misc.h"

#pragma warning(push, 0)

#include <string>
#include <vector>

#pragma warning(pop)

extern template class std::vector<double>;

// Forward declarations
class AbsTaskMonitor;
class ITimerResult;

/// Abstract class for monitoring progress of a given job
class AbsJobMonitor /*abstract*/ {
 public:
  // No intention to copy / move
  AbsJobMonitor(const AbsJobMonitor&) = delete;
  AbsJobMonitor(AbsJobMonitor&&) = delete;
  void operator=(const AbsJobMonitor&) = delete;
  void operator=(AbsJobMonitor&&) = delete;

  virtual ~AbsJobMonitor() noexcept {}

  /// Name of the job
  const std::string& monitoredJob() const noexcept { return _monitoredJob; }

  /// Overall progress of the job (0..1 range)
  double progress() const noexcept { return _progress; }

  /// Reports if the job was aborted or not
  bool wasAborted() const noexcept { return aborted; }

  /**
  Before starting a certain job, usually there is enough information to provide
  some estimates about the weight of each particular task of the job.
  All these estimates must be positive and could sum up to 1 or 100 or anything
  meaningful. Internally, they will be scaled to sum exactly 1.

  The parameter timer_ is the associated timer for reporting elapsed and
  estimated remaining time

  @throw invalid_argument if totalContribValues contains negative values or 0-s

  Exception to be only reported, not handled
  */
  virtual void setTasksDetails(const std::vector<double>& totalContribValues,
                               ITimerResult& timer_) noexcept(!UT) = 0;

  /**
  At the start of each task of a given job, the user must create a
  method-static instance of AbsTaskMonitor-derived, which registers itself
  to a AbsJobMonitor-derived using:
  monitorNewTask(*this)
  The job monitor will return the order of the newly registered task
  within the sequence of tasks required for this job.

  @param newActivity task that registers itself as part of a job

  @return the order of this new task among job's tasks
  */
  virtual unsigned monitorNewTask(AbsTaskMonitor& newActivity) noexcept = 0;

  /**
  A task monitor reports the progress of its supervised task.

  @param taskProgress value in 0..1 range representing the progress of the task
  @param taskSeqId the order of the task among job's tasks

  @throw invalid_argument if taskProgress is outside 0..1
  @throw out_of_range if taskSeqId is an invalid index in details

  Exceptions to be only reported, not handled
  */
  virtual void taskAdvanced(double taskProgress,
                            unsigned taskSeqId) noexcept(!UT) = 0;

  /**
  A task monitor reports the completion of its supervised task.

  @param taskSeqId the order of the task among job's tasks
  @throw out_of_range only in UnitTesting if taskSeqId is an invalid index in
  details

  Exception can be caught only in UnitTesting
  */
  virtual void taskDone(unsigned taskSeqId) noexcept(!UT) = 0;

  /**
  A task monitor reports the abortion of its supervised task.

  @param taskSeqId the order of the task among job's tasks
  @throw out_of_range only in UnitTesting if taskSeqId is an invalid index in
  details

  Exception can be caught only in UnitTesting
  */
  virtual void taskAborted(unsigned taskSeqId) noexcept(!UT) = 0;

 protected:
  explicit AbsJobMonitor(const std::string& monitoredJob) noexcept
      : _monitoredJob(monitoredJob) {}

  /// Prepares the monitor for a new timing using timer_
  virtual void getReady(ITimerResult& timer) noexcept;

  /**
  @return reference to the provided timer, if any
  @throw logic_error if called before getReady(), which sets a timer

  Exception to be only reported, not handled
  */
  ITimerResult& timer() const noexcept;

  /**
  Sets a new progress
  @throw invalid_argument if the argument is outside 0..1

  Exception to be only reported, not handled
  */
  void progress(double progress) noexcept;

 private:
  const std::string _monitoredJob;  ///< name of the job

  /// Timer for reporting elapsed and estimated remaining time
  ITimerResult* _timer = nullptr;

  double _progress = 0.;  ///< actual known job's progress

  bool aborted = false;  ///< set if the job was aborted
};

#endif  // H_JOB_MONITOR_BASE
