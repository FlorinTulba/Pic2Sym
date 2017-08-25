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

#ifndef H_TIMING
#define H_TIMING

#include "timingBase.h"

#pragma warning ( push, 0 )

#include "std_memory.h"
#include <chrono>
#include <vector>

#pragma warning ( pop )

/**
ActiveTimer class:
- realization of IActiveTimer
- one of the 2 base classes of Timer

Timer becomes a `Concentrator class` if it realizes alone both IActiveTimer and ITimerResult.
*/
class ActiveTimer /*abstract*/ : public IActiveTimer {
protected:
	const std::vector<std::sharedPtr<ITimerActions>> observers; ///< to be notified

	/// the moment when computation started / was resumed last time
	std::chrono::time_point<std::chrono::high_resolution_clock> lastStart;

	/// sum of previous intervals, when repeatedly paused and resumed
	std::chrono::duration<double> elapsedS;

	bool paused = false;	///< true as long as not paused
	bool valid = true;		///< true as long as not canceled / released

	/// Initializes lastStart and notifies all observers
	ActiveTimer(const std::vector<std::sharedPtr<ITimerActions>> &observers_);

	ActiveTimer(std::sharedPtr<ITimerActions> observer); ///< initializes lastStart and notifies the observer

	/**
	This class relies on automatic destructor calling, so duplicates mean 2 destructor calls.
	There has to be only 1 notifier, so ActiveTimer cannot have copies.
	So, there'll be ONLY the move constructor controlling the destruction of the source object!
	*/
	ActiveTimer(ActiveTimer &&other);
	ActiveTimer(const ActiveTimer&) = delete;
	void operator=(const ActiveTimer&) = delete;
	void operator=(ActiveTimer&&) = delete;

public:
	virtual ~ActiveTimer();		///< if not canceled / released, reports duration to all observers

	void invalidate();			///< prevents further use of this timer

	virtual void release();		///< stops the timer and reports duration to all observers

	void pause() override;		///< pauses the timer and reports duration to all observers
	void resume() override;		///< resumes the timer

	/// Cancels a timing task.
	/// @param reason explanation for cancellation
	void cancel(const std::stringType &reason = "The task was canceled") override;
};

/// Timer class
class Timer : public ActiveTimer, public ITimerResult {
public:
	/// Initializes lastStart and notifies all observers
	Timer(const std::vector<std::sharedPtr<ITimerActions>> &observers_);

	Timer(std::sharedPtr<ITimerActions> observer); ///< initializes lastStart and notifies the observer

	/**
	This class relies on automatic destructor calling, so duplicates mean 2 destructor calls.
	There has to be only 1 notifier, so Timer cannot have copies.
	So, there'll be ONLY the move constructor controlling the destruction of the source object!
	*/
	Timer(Timer &&other);
	Timer(const Timer&) = delete;
	void operator=(const Timer&) = delete;
	void operator=(Timer&&) = delete;

	double elapsed() const override;	///< reports elapsed duration depending on valid & paused
};

#endif // H_TIMING
