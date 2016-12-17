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
 
 (c) 2016 Florin Tulba <florintulba@yahoo.com>

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

#ifndef H_JOB_MONITOR_BASE
#define H_JOB_MONITOR_BASE

#pragma warning ( push, 0 )

#include <string>
#include <vector>

#pragma warning ( pop )

// Forward declarations
class AbsTaskMonitor;
class Timer;

/// Abstract class for monitoring progress of a given job
class AbsJobMonitor /*abstract*/ {
protected:
	const std::string monitoredJob_;	///< name of the job
	Timer *timer = nullptr;				///< timer for reporting elapsed and estimated remaining time

	double progress_ = 0.;				///< actual known job's progress

	bool aborted = false;				///< set if the job was aborted

public:
	AbsJobMonitor(const std::string &monitoredJob) : monitoredJob_(monitoredJob) {}
	void operator=(const AbsJobMonitor&) = delete;
	virtual ~AbsJobMonitor() = 0 {};

	const std::string& monitoredJob() const { return monitoredJob_; } ///< name of the job
	double progress() const { return progress_; }	///< Overall progress of the job (0..1 range)
	bool wasAborted() const { return aborted; }		///< Reports if the job was aborted or not

	/**
	Before starting a certain job, usually there is enough information to provide
	some estimates about the weight of each particular task of the job.
	All these estimates must sum up to 1.

	The parameter timer_ is the associated timer for reporting elapsed and estimated remaining time
	*/
	virtual void setTasksDetails(const std::vector<double> &totalContribValues, Timer &timer_) = 0;

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
	virtual unsigned monitorNewTask(AbsTaskMonitor &newActivity) = 0;

	/**
	A task monitor reports the progress of its supervised task.

	@param taskProgress value in 0..1 range representing the progress of the task
	@param taskSeqId the order of the task among job's tasks
	*/
	virtual void taskAdvanced(double taskProgress, unsigned taskSeqId) = 0;

	/**
	A task monitor reports the completion of its supervised task.

	@param taskSeqId the order of the task among job's tasks
	*/
	virtual void taskDone(unsigned taskSeqId) = 0;

	/**
	A task monitor reports the abortion of its supervised task.

	@param taskSeqId the order of the task among job's tasks
	*/
	virtual void taskAborted(unsigned taskSeqId) = 0;
};

#endif // H_JOB_MONITOR_BASE