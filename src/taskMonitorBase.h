/************************************************************************************************
 The application Pic2Sym approximates images by a
 grid of colored symbols with colored backgrounds.

 This file belongs to the Pic2Sym project.

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

#ifndef H_TASK_MONITOR_BASE
#define H_TASK_MONITOR_BASE

#pragma warning ( push, 0 )

#include <string>

#pragma warning ( pop )

/// Abstract class for monitoring progress of a specific activity within a given job
class AbsTaskMonitor /*abstract*/ {
protected:
	const std::string monitoredTask_;	///< name of the monitored task

public:
	AbsTaskMonitor(const std::string &monitoredTask) : monitoredTask_(monitoredTask) {}
	void operator=(const AbsTaskMonitor&) = delete;
	virtual ~AbsTaskMonitor() = 0 {}

	const std::string& monitoredTask() const { return monitoredTask_; } ///< name of the activity

	virtual void setTotalSteps(size_t totalSteps_) = 0;	///< total steps required to finish the activity
	virtual void taskAdvanced(size_t steps = 1U) = 0;	///< task performer reports its progress
	virtual void taskDone() = 0;						///< task performer reports finishing this activity
	virtual void taskAborted() = 0;	///< task performer reports that the activity was aborted
};

#endif // H_TASK_MONITOR_BASE