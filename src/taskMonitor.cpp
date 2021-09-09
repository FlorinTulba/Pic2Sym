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

#ifndef UNIT_TESTING

#include "taskMonitor.h"

#include "warnings.h"

#pragma warning(push, 0)

#include <iostream>

#include <gsl/gsl>

#pragma warning(pop)

using namespace std;

namespace pic2sym::ui {

#pragma warning(disable : WARN_BASE_INIT_USING_THIS)
TaskMonitor::TaskMonitor(const string& monitoredActivity,
                         AbsJobMonitor& parent_) noexcept
    : AbsTaskMonitor{monitoredActivity},
      parent(&parent_),

      // register itself to the parent job monitor and get the order of this
      // task within job's tasks
      seqId(parent_.monitorNewTask(*this)) {}
#pragma warning(default : WARN_BASE_INIT_USING_THIS)

void TaskMonitor::setTotalSteps(size_t totalSteps_) {
  if (parent->wasAborted())
    reportAndThrow<AbortedJob>("Attempting to set up '" + monitoredTask() +
                               "' subtask of aborted job '"s +
                               parent->monitoredJob() + "'!"s);

  // Kept as double to reduce the conversions required to obtain progress value
  // (steps/totalSteps)
  totalSteps = (double)totalSteps_;
}

void TaskMonitor::taskAdvanced(size_t steps /* = 1U*/) {
  EXPECTS_OR_REPORT_AND_THROW_CONST_MSG(
      totalSteps > 0., logic_error,
      HERE.function_name() + " called before setTotalSteps()!"s);

  if (parent->wasAborted())
    reportAndThrow<AbortedJob>("Called "s + HERE.function_name() + " on '"s +
                               monitoredTask() + "' subtask of aborted job '"s +
                               parent->monitoredJob() + "'!"s);

  if (!steps)
    return;

  double taskProgress{steps / totalSteps};
  if (taskProgress > EpsPlus1) {
    cerr << "Current task stage (" << steps << ") is more than task's span ("
         << (size_t)totalSteps << ")" << endl;
    taskProgress = 1.;
  }

  parent->taskAdvanced(taskProgress, seqId);
}

void TaskMonitor::taskDone() {
  if (parent->wasAborted())
    reportAndThrow<AbortedJob>("Called "s + HERE.function_name() + " on '"s +
                               monitoredTask() + "' subtask of aborted job '"s +
                               parent->monitoredJob() + "'!"s);

  parent->taskDone(seqId);
}

void TaskMonitor::taskAborted() noexcept {
  if (!parent->wasAborted())
    parent->taskAborted(seqId);
}

}  // namespace pic2sym::ui

#endif  // UNIT_TESTING not defined
