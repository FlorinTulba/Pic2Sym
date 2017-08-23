/************************************************************************************************
 The application Pic2Sym approximates images by a
 grid of colored symbols with colored backgrounds.

 Copyrights from the libraries used by the program:
 - (c) 2016 Boost (www.boost.org)
		License: <http://www.boost.org/LICENSE_1_0.txt>
			or doc/licenses/Boost.lic
 - (c) 2015 OpenCV (www.opencv.org)
		License: <http://opencv.org/license.html>
            or doc/licenses/OpenCV.lic
 - (c) 2015 The FreeType Project (www.freetype.org)
		License: <http://git.savannah.gnu.org/cgit/freetype/freetype2.git/plain/docs/FTL.TXT>
	        or doc/licenses/FTL.txt
 - (c) 1997-2002 OpenMP Architecture Review Board (www.openmp.org)
   (c) Microsoft Corporation (Visual C++ implementation for OpenMP C/C++ Version 2.0 March 2002)
		See: <https://msdn.microsoft.com/en-us/library/8y6825x5(v=vs.140).aspx>
 - (c) 1995-2013 zlib software (Jean-loup Gailly and Mark Adler - see: www.zlib.net)
		License: <http://www.zlib.net/zlib_license.html>
            or doc/licenses/zlib.lic
 
 (c) 2016, 2017 Florin Tulba <florintulba@yahoo.com>

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
 If not, see <http://www.gnu.org/licenses/agpl-3.0.txt>.
 ***********************************************************************************************/

#ifndef UNIT_TESTING

#include "jobMonitor.h"
#include "taskMonitorBase.h"
#include "progressNotifier.h"
#include "timing.h"

#pragma warning ( push, 0 )

#include <iostream>
#include <iomanip>
#include <cassert>

#pragma warning ( pop )

using namespace std;

JobMonitor::JobMonitor(const stringType &monitoredActivity, std::sharedPtr<IProgressNotifier> userNotifier_,
					   double minProgressForUserNotifications_) :
		AbsJobMonitor(monitoredActivity), notifier(userNotifier_),

		// Slightly diminish the threshold for how often to notify the progress of the job to the user
		minProgressForNotifications(minProgressForUserNotifications_ - EPS) {

	assert(notifier);
	assert(minProgressForUserNotifications_ >= 0. && minProgressForUserNotifications_ <= 1.);
}

unsigned JobMonitor::monitorNewTask(AbsTaskMonitor &newActivity) {
	const unsigned seqIdxOfNewActivity = (unsigned)tasks.size(); // the order of the new task among all job's tasks
	tasks.push_back(&newActivity);
	return seqIdxOfNewActivity;
}

void JobMonitor::setTasksDetails(const vector<double> &totalContribValues, Timer &timer_) {
	const size_t totalContribItems = totalContribValues.size();
	details.resize(totalContribItems);

	if(0U == totalContribItems)
		return;

	// Estimated start moment of each task can be deduced from estimated total times of previous tasks
	double contribStart;
	details[0ULL] = TaskDetails(0., contribStart = totalContribValues[0ULL]);
	for(unsigned i = 1U; i < totalContribItems; ++i) {
		const double totalTaskContrib = totalContribValues[(size_t)i];
		details[(size_t)i] = TaskDetails(contribStart, totalTaskContrib);
		contribStart += totalTaskContrib;
	}

	assert(abs(contribStart - 1.) < EPS);

	aborted = false;
	lastUserNotifiedProgress = lastUserNotifiedElapsedTime = progress_ = 0.;
	timer = &timer_;
}

void JobMonitor::taskAdvanced(double taskProgress, unsigned taskSeqId) {
	const TaskDetails &taskDetails = details.at(taskSeqId);
	progress_ = taskDetails.contribStart + taskProgress * taskDetails.totalContrib;

	// Notify the user only as frequently as demanded
	if(progress_ - lastUserNotifiedProgress > minProgressForNotifications) {
		const double updatedElapsed = timer->elapsed(),
					timeDiff = updatedElapsed - lastUserNotifiedElapsedTime,
					progressDiff = progress_ - lastUserNotifiedProgress;

		if(timeDiff > 0. && progressDiff > 0. && progress_ < 1.) {
			const double remainingProgress = 1. - progress_,
						estimatedRemainingTime = remainingProgress * timeDiff / progressDiff;
			cout<<monitoredJob_<<" -> Elapsed time:"<<setw(10)<<right<<fixed<<setprecision(2)<<updatedElapsed<<"s ; "
				<<"Estimated remaining time:"<<setw(10)<<right<<fixed<<setprecision(2)<<estimatedRemainingTime<<"s\r";
		}

		lastUserNotifiedElapsedTime = updatedElapsed;

		notifier->notifyUser(monitoredJob_,
							 lastUserNotifiedProgress = progress_);
	}
}

void JobMonitor::taskDone(unsigned taskSeqId) {
	taskAdvanced(1., taskSeqId);
}

void JobMonitor::taskAborted(unsigned taskSeqId) {
	cout<<endl<<monitoredJob_<<" was aborted after "
		<<fixed<<setprecision(2)<<timer->elapsed()<<"s at "
		<<fixed<<setprecision(2)<<100.*progress_<<"% during "
		<<tasks.at(taskSeqId)->monitoredTask()<<'!'<<endl;
	lastUserNotifiedProgress = lastUserNotifiedElapsedTime = progress_ = 0.;
}

#endif // UNIT_TESTING not defined
