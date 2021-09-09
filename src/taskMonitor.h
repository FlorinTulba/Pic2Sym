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

#ifdef UNIT_TESTING
#include "../test/mockTaskMonitor.h"

#else  // UNIT_TESTING not defined

#ifndef H_TASK_MONITOR
#define H_TASK_MONITOR

#include "taskMonitorBase.h"

#include "jobMonitorBase.h"

#pragma warning(push, 0)

#include <gsl/gsl>

#pragma warning(pop)

namespace pic2sym::ui {

/// Implementation of AbsTaskMonitor for supervising a task with a job
class TaskMonitor : public AbsTaskMonitor {
 public:
  /**
  Used to construct a method-static monitor of a task (monitoredActivity) within
  a given job (parent_). The constructor calls parent_.monitorNewTask(*this) to
  initialize field seqId and to let the parent job know about this new task.
  */
  TaskMonitor(const std::string& monitoredActivity,
              AbsJobMonitor& parent_) noexcept;

  /// Total steps required to finish the activity. Throws AbortedJob if the
  /// parent job was aborted
  void setTotalSteps(size_t totalSteps_) override;

  /**
  Task performer reports its progress
  @throw logic_error if called before setTotalSteps()
  Exception to be only reported, not handled

  @throw AbortedJob if the parent job was aborted.
  This exception must be handled
  */
  void taskAdvanced(size_t steps /* = 1U*/) override;

  /// Task performer reports finishing this activity. Throws AbortedJob if the
  /// parent job was aborted
  void taskDone() override;

  /// Task performer reports that the activity was aborted
  void taskAborted() noexcept override;

 private:
  gsl::not_null<AbsJobMonitor*> parent;  ///< reference of the parent job

  /**
  Total count of the required steps to complete the task.
  Kept as double to reduce the conversions required to obtain progress value
  (steps/totalSteps).
  */
  double totalSteps{};

  unsigned seqId;  ///< order of the supervised task among job's tasks
};

}  // namespace pic2sym::ui

#endif  // H_TASK_MONITOR

#endif  // UNIT_TESTING not defined
