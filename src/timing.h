/****************************************************************************************
 The application Pic2Sym approximates images by a
 grid of colored symbols with colored backgrounds.

 This file belongs to the Pic2Sym project.

 Copyrights from the libraries used by the program:
 - (c) 2015 Boost (www.boost.org)
   License: <http://www.boost.org/LICENSE_1_0.txt>
            or doc/licenses/Boost.lic
 - (c) 2015 The FreeType Project (www.freetype.org)
   License: <http://git.savannah.gnu.org/cgit/freetype/freetype2.git/plain/docs/FTL.TXT>
	        or doc/licenses/FTL.txt
 - (c) 2015 OpenCV (www.opencv.org)
   License: <http://opencv.org/license.html>
            or doc/licenses/OpenCV.lic
 - (c) 1997-2002 OpenMP Architecture Review Board (www.openmp.org)
   (c) Microsoft Corporation (Visual C++ implementation for OpenMP C/C++ Version 2.0 March 2002)
   See: <https://msdn.microsoft.com/en-us/library/8y6825x5(v=vs.140).aspx>
 
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
 ****************************************************************************************/

#ifndef H_TIMING
#define H_TIMING

#include "misc.h"

#include <chrono>
#include <vector>
#include <memory>
#include <string>

// Timing jobs:
/// Interface for any observer of a timer (see below the Timer class)
struct ITimerActions /*abstract*/ {
	virtual void onStart() {}	///< action to be performed when the timer is started

	/// action to be performed when the timer is paused
	/// @param elapsedS time elapsed this far in seconds
	virtual void onPause(double elapsedS) { UNREFERENCED_PARAMETER(elapsedS); }

	virtual void onResume() {}	///< action to be performed when the timer is resumed

	/// action to be performed when the timer is released/deleted
	/// @param elapsedS total elapsed time in seconds
	virtual void onRelease(double elapsedS) { UNREFERENCED_PARAMETER(elapsedS); }

	/// action to be performed when the timer is canceled
	/// @param reason explanation for cancellation
	virtual void onCancel(const std::string &reason = "") { UNREFERENCED_PARAMETER(reason); }

	virtual ~ITimerActions() = 0 {}
};

/// Timer class
class Timer {
protected:
	const std::vector<std::shared_ptr<ITimerActions>> observers; ///< to be notified

	/// the moment when computation started / was resumed last time
	std::chrono::time_point<std::chrono::high_resolution_clock> lastStart;

	/// sum of previous intervals, when repeatedly paused and resumed
	std::chrono::duration<double> elapsedS;

	bool paused = false;	///< true as long as not paused
	bool valid = true;		///< true as long as not canceled / released

public:
	/// Initializes lastStart and notifies all observers
	Timer(const std::vector<std::shared_ptr<ITimerActions>> &observers_);

	Timer(std::shared_ptr<ITimerActions> observer); ///< initializes lastStart and notifies the observer

	/**
	This class relies on automatic destructor calling, so duplicates mean 2 destructor calls.
	There has to be only 1 notifier, so Timer cannot have copies.
	So, there'll be ONLY the move constructor controlling the destruction of the source object!
	*/
	Timer(Timer &&other);
	Timer(const Timer&) = delete;
	void operator=(const Timer&) = delete;
	void operator=(Timer&&) = delete;

	virtual ~Timer();				///< if not canceled / released, reports duration to all observers

	virtual void pause();			///< pauses the timer and reports duration to all observers
	virtual void resume();			///< resumes the timer
	virtual void release();			///< stops the timer and reports duration to all observers

	/// Cancels a timing task.
	/// @param reason explanation for cancellation
	virtual void cancel(const std::string &reason = "The task was canceled");
};

#endif