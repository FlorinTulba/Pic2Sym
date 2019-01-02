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
 If not, see <http://www.gnu.org/licenses/agpl-3.0.txt>.
 ***********************************************************************************************/

#ifdef UNIT_TESTING
#	include "../test/mockTaskMonitor.h"

#else // UNIT_TESTING not defined

#ifndef H_TASK_MONITOR
#define H_TASK_MONITOR

#include "taskMonitorBase.h"

class AbsJobMonitor; // forward declaration

/// Implementation of AbsTaskMonitor for supervising a task with a job
class TaskMonitor : public AbsTaskMonitor {
protected:
	AbsJobMonitor &parent;	///< reference of the parent job

	/**
	Total count of the required steps to complete the task.
	Kept as double to reduce the conversions required to obtain progress value (steps/totalSteps).
	*/
	double totalSteps = 0.;

	unsigned seqId;			///< order of the supervised task among job's tasks

public:
	/**
	Used to construct a method-static monitor of a task (monitoredActivity) within a given job (parent_).
	The constructor calls parent_.monitorNewTask(*this) to initialize field seqId and
	to let the parent job know about this new task.
	*/
	TaskMonitor(const std::stringType &monitoredActivity, AbsJobMonitor &parent_);
	void operator=(const TaskMonitor&) = delete;

	void setTotalSteps(size_t totalSteps_) override;	///< total steps required to finish the activity

	void taskAdvanced(size_t steps/* = 1U*/) override;	///< task performer reports its progress
	void taskDone() override;							///< task performer reports finishing this activity
	void taskAborted() override;	///< task performer reports that the activity was aborted
};

#endif // H_TASK_MONITOR

#endif // UNIT_TESTING not defined
