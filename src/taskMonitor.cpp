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

#ifndef UNIT_TESTING

#include "taskMonitor.h"
#include "jobMonitorBase.h"
#include "warnings.h"

#pragma warning ( push, 0 )

#include <iostream>
#include <cassert>

#pragma warning ( pop )

using namespace std;

extern const double EPSp1();

#pragma warning( disable : WARN_BASE_INIT_USING_THIS )
TaskMonitor::TaskMonitor(const stringType &monitoredActivity, AbsJobMonitor &parent_) :
		AbsTaskMonitor(monitoredActivity), parent(parent_),

		// register itself to the parent job monitor and get the order of this task within job's tasks
		seqId(parent_.monitorNewTask(*this)) {}
#pragma warning( default : WARN_BASE_INIT_USING_THIS )

void TaskMonitor::setTotalSteps(size_t totalSteps_) {
	// Kept as double to reduce the conversions required to obtain progress value (steps/totalSteps)
	totalSteps = (double)totalSteps_;
}

void TaskMonitor::taskAdvanced(size_t steps/* = 1U*/) {
	assert(totalSteps > 0.); // Please call this method only after setTotalSteps()
	if(0ULL == steps)
		return;

	double taskProgress = steps / totalSteps;
	if(taskProgress > EPSp1()) {
		cerr<<"Current task stage ("<<steps<<") is more than task's span ("<<(size_t)totalSteps<<")"<<endl;
		taskProgress = 1.;
	}

	parent.taskAdvanced(taskProgress, seqId);
}

void TaskMonitor::taskDone() {
	parent.taskAdvanced(1., seqId);
}

void TaskMonitor::taskAborted() {
	parent.taskAborted(seqId);
}

#endif // UNIT_TESTING not defined
