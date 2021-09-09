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

#include "jobMonitor.h"

#include "progressNotifier.h"
#include "taskMonitorBase.h"
#include "timingBase.h"
#include "warnings.h"

#pragma warning(push, 0)

#include <iomanip>
#include <iostream>
#include <numeric>

#include <gsl/gsl>

#pragma warning(pop)

using namespace std;
using namespace gsl;

namespace pic2sym::ui {

/**
@param val a value to investigate
@return val if > 0
@throw invalid_argument if val <= 0
*/
static double strictlyPositive(double val) {
  EXPECTS_OR_REPORT_AND_THROW_CONST_MSG(
      val > 0., invalid_argument, HERE.function_name() + " needs val > 0!"s);
  return val;
};

void AbsJobMonitor::getReady(ITimerResult& timer) noexcept {
  _timer = &timer;
  _progress = 0.;
  aborted = false;
}

#pragma warning(disable : WARN_THROWS_ALTHOUGH_NOEXCEPT)
ITimerResult& AbsJobMonitor::timer() const noexcept {
  EXPECTS_OR_REPORT_AND_THROW_CONST_MSG(
      _timer, logic_error,
      HERE.function_name() + " called before getReady()!"s);

  return *_timer;
}
#pragma warning(default : WARN_THROWS_ALTHOUGH_NOEXCEPT)

#pragma warning(disable : WARN_THROWS_ALTHOUGH_NOEXCEPT)
void AbsJobMonitor::progress(double progress) noexcept {
  EXPECTS_OR_REPORT_AND_THROW(
      progress >= 0. && progress <= EpsPlus1, invalid_argument,
      HERE.function_name() + " - progress ("s + to_string(progress) +
          ") needs to be between 0..1!"s);

  _progress = progress;
}
#pragma warning(default : WARN_THROWS_ALTHOUGH_NOEXCEPT)

#pragma warning(disable : WARN_THROWS_ALTHOUGH_NOEXCEPT)
JobMonitor::JobMonitor(const string& monitoredActivity,
                       std::unique_ptr<IProgressNotifier> userNotifier_,
                       double minProgressForUserNotifications_) noexcept(!UT)
    : AbsJobMonitor{monitoredActivity},
      notifier(move(userNotifier_)),

      /*
      Slightly diminish the threshold for how often to notify the progress of
      the job to the user, to ensure that even the last expected report will
      be delivered.

      For instance 100*(0.01-Eps) cannot overshoot 1,
      while 100*0.01 might be slightly larger than 1, so losing a report
      */
      minProgressForNotifications(minProgressForUserNotifications_ - Eps) {
  EXPECTS_OR_REPORT_AND_THROW_CONST_MSG(
      notifier, invalid_argument,
      HERE.function_name() + " needs non-null notifier!"s);
  EXPECTS_OR_REPORT_AND_THROW_CONST_MSG(
      minProgressForUserNotifications_ > 0. &&
          minProgressForUserNotifications_ <= 1.,
      invalid_argument,
      HERE.function_name() +
          " needs minProgressForUserNotifications_ to be between 0 .. 1!"s);
}
#pragma warning(default : WARN_THROWS_ALTHOUGH_NOEXCEPT)

unsigned JobMonitor::monitorNewTask(AbsTaskMonitor& newActivity) noexcept {
  // the order of the new task among all job's tasks
  const unsigned seqIdxOfNewActivity{narrow_cast<unsigned>(size(tasks))};
  tasks.push_back(&newActivity);
  return seqIdxOfNewActivity;
}

void JobMonitor::getReady(ITimerResult& timer_) noexcept {
  AbsJobMonitor::getReady(timer_);
  lastUserNotifiedProgress = lastUserNotifiedElapsedTime = 0.;
}

void JobMonitor::setTasksDetails(const vector<double>& totalContribValues,
                                 ITimerResult& timer_) noexcept(!UT) {
  const size_t totalContribItems{size(totalContribValues)};
  details.resize(totalContribItems);

  if (!totalContribItems)
    return;

  const double totalContribSum{
      strictlyPositive(accumulate(CBOUNDS(totalContribValues), 0.))};

  // Estimated start moment of each task can be deduced from estimated total
  // times of previous tasks
  double contribStart{
      strictlyPositive(totalContribValues[0ULL] / totalContribSum)};

  details[0ULL] = {.contribStart = 0., .totalContrib = contribStart};
  for (unsigned i{1U}; i < totalContribItems; ++i) {
    const double totalTaskContrib{
        strictlyPositive(totalContribValues[(size_t)i] / totalContribSum)};
    details[(size_t)i] = {.contribStart = contribStart,
                          .totalContrib = totalTaskContrib};
    contribStart += totalTaskContrib;
  }

  getReady(timer_);
}

void JobMonitor::taskAdvanced(double taskProgress, unsigned taskSeqId) {
  EXPECTS_OR_REPORT_AND_THROW(
      taskProgress >= 0. && taskProgress <= EpsPlus1, invalid_argument,
      HERE.function_name() + " - taskProgress ("s + to_string(taskProgress) +
          ") needs to be between 0..1!"s);
  EXPECTS_OR_REPORT_AND_THROW((size_t)taskSeqId < size(details), out_of_range,
                              HERE.function_name() + " - taskSeqId ("s +
                                  to_string(taskSeqId) +
                                  ") isn't valid index within details["s +
                                  to_string(size(details)) + "]"s);

  if (wasAborted())
    reportAndThrow<AbortedJob>(
        "Call for '" + tasks[(size_t)taskSeqId]->monitoredTask() +
        "' subtask of aborted job '"s + monitoredJob() + "'!"s);

  const TaskDetails& taskDetails = details[(size_t)taskSeqId];
  const double progress_{taskDetails.contribStart +
                         taskProgress * taskDetails.totalContrib};
  progress(progress_);

  // Notify the user only as frequently as demanded
  if (progress_ - lastUserNotifiedProgress > minProgressForNotifications) {
    const double updatedElapsed{timer().elapsed()};
    const double timeDiff{updatedElapsed - lastUserNotifiedElapsedTime};
    const double progressDiff{progress_ - lastUserNotifiedProgress};

    if (timeDiff > 0. && progressDiff > 0. && progress_ < 1.) {
      const double remainingProgress{1. - progress_};
      const double estimatedRemainingTime{remainingProgress * timeDiff /
                                          progressDiff};
      cout << monitoredJob() << " -> Elapsed time:" << setw(10) << right
           << fixed << setprecision(2) << updatedElapsed << "s ; "
           << "Estimated remaining time:" << setw(10) << right << fixed
           << setprecision(2) << estimatedRemainingTime << "s\r";
    }

    lastUserNotifiedElapsedTime = updatedElapsed;

    notifier->notifyUser(monitoredJob(), lastUserNotifiedProgress = progress());
  }
}

void JobMonitor::taskDone(unsigned taskSeqId) {
  taskAdvanced(1., taskSeqId);
}

#pragma warning(disable : WARN_THROWS_ALTHOUGH_NOEXCEPT)
void JobMonitor::taskAborted(unsigned taskSeqId) noexcept(!UT) {
  EXPECTS_OR_REPORT_AND_THROW((size_t)taskSeqId < size(details), out_of_range,
                              HERE.function_name() + " - taskSeqId ("s +
                                  to_string(taskSeqId) +
                                  ") isn't valid index within details["s +
                                  to_string(size(details)) + "]"s);

  if (wasAborted())
    return;

  cout << '\n'
       << monitoredJob() << " was aborted after " << fixed << setprecision(2)
       << timer().elapsed() << "s at " << fixed << setprecision(2)
       << 100. * progress() << "% during " << tasks[taskSeqId]->monitoredTask()
       << '!' << endl;
  abort();
  lastUserNotifiedProgress = lastUserNotifiedElapsedTime = 0.;
}
#pragma warning(default : WARN_THROWS_ALTHOUGH_NOEXCEPT)

}  // namespace pic2sym::ui

#endif  // UNIT_TESTING not defined
