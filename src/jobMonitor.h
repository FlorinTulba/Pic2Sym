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

#ifdef UNIT_TESTING
#	include "../test/mockJobMonitor.h"

#else // UNIT_TESTING not defined

#ifndef H_JOB_MONITOR
#define H_JOB_MONITOR

#include "jobMonitorBase.h"

#pragma warning ( push, 0 )

#include <limits>
#include <memory>

#pragma warning ( pop )

struct IProgressNotifier; // forward declaration

/// Implementation of AbsJobMonitor for supervising a given job
class JobMonitor : public AbsJobMonitor {
protected:
	/// Estimates about when a certain task should start while performing a given job and how long will it take
	struct TaskDetails {
		double contribStart;	///< estimate of the starting point of this task within the job ([0..1) range)
		double totalContrib;	///< estimate of the duration of this task within the job ((0..1] range)

		TaskDetails(double contribStart_ = std::numeric_limits<double>::infinity(),
					double totalContrib_ = 0.) :
					contribStart(contribStart_), totalContrib(totalContrib_) {}
	};

	std::shared_ptr<IProgressNotifier> notifier;	///< needed for reporting the progress of the job to the user
	double minProgressForNotifications;	///< how frequent should be the user notifications (ex.: 0.05 for reporting only every 5%)

	/**
	Tasks comprising this job, to be executed sequentially.
	They register themselves to this job monitor by a method-static declaration of an AbsTaskMonitor-derived object
	who's calling monitorNewTask(*this) within its constructor.
	*/
	std::vector<AbsTaskMonitor*> tasks;

	/**
	Estimates about the start moment and duration of each task within the job.
	These get set before starting the actual job, when such information is most likely to become available.
	*/
	std::vector<TaskDetails> details;

	double lastUserNotifiedProgress = 0.;		///< last value of job's progress reported to the user
	double lastUserNotifiedElapsedTime = 0.;	///< last elapsed time reported to the user

public:
	/**
	Provides the job with its name, a notifier that informs the user about the progress and
	the minimum threshold for how frequent the user notifications should be issued.
	*/
	JobMonitor(const std::string &monitoredActivity, std::shared_ptr<IProgressNotifier> userNotifier_,
			   double minProgressForUserNotifications_);
	void operator=(const JobMonitor&) = delete;

	/**
	Before starting a certain job, usually there is enough information to provide
	some estimates about the weight of each particular task of the job.
	All these estimates must sum up to 1.

	The parameter timer_ is the associated timer for reporting elapsed and estimated remaining time
	*/
	void setTasksDetails(const std::vector<double> &totalContribValues, Timer &timer_) override;

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
	unsigned monitorNewTask(AbsTaskMonitor &newActivity) override;

	/**
	A task monitor reports the progress of its supervised task.

	@param taskProgress value in 0..1 range representing the progress of the task
	@param taskSeqId the order of the task among job's tasks
	*/
	void taskAdvanced(double taskProgress, unsigned taskSeqId) override;

	/**
	A task monitor reports the completion of its supervised task.

	@param taskSeqId the order of the task among job's tasks
	*/
	void taskDone(unsigned taskSeqId) override;

	/**
	A task monitor reports the abortion of its supervised task.

	@param taskSeqId the order of the task among job's tasks
	*/
	void taskAborted(unsigned taskSeqId) override;
};

#endif // H_JOB_MONITOR

#endif // UNIT_TESTING not defined
