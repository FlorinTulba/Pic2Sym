/****************************************************************************************
 The application Pic2Sym approximates images by a
 grid of colored symbols with colored backgrounds.

 This file was created on 2016-1-8
 and belongs to the Pic2Sym project.

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

#ifndef H_MISC
#define H_MISC

#include <iostream>
#include <iomanip>
#include <string>
#include <chrono>
#include <vector>
#include <memory>

// Error margin
const double EPS = 1e-6;

// Display an expression and its value
#define PRINT(expr)			std::cout<<#expr " : "<<(expr)
#define PRINTLN(expr)		PRINT(expr)<<std::endl
#define PRINT_H(expr)		std::cout<<#expr " : 0x"<<std::hex<<(expr)<<std::dec
#define PRINTLN_H(expr)		PRINT_H(expr)<<std::endl

// Oftentimes functions operating on ranges need the full range.
// Example: copy(x.begin(), x.end(), ..) => copy(BOUNDS(x), ..)
#define BOUNDS(iterable)	std::begin(iterable), std::end(iterable)
#define CBOUNDS(iterable)	std::cbegin(iterable), std::cend(iterable)

// string <-> wstring conversions
std::wstring str2wstr(const std::string &str);
std::string wstr2str(const std::wstring &wstr);

// Notifying the user
void infoMsg(const std::string &text, const std::string &title = "");
void warnMsg(const std::string &text, const std::string &title = "");
void errMsg(const std::string &text, const std::string &title = "");

// Timing jobs:
/// Interface for any observer of a timer (see below the Timer class)
struct ITimerActions /*abstract*/ {
	virtual void onStart() {}	///< action to be performed when the timer is started

	/// action to be performed when the timer is paused
	/// @param elapsedS time elapsed this far in seconds
	virtual void onPause(double /*elapsedS*/) {}
	
	virtual void onResume() {}	///< action to be performed when the timer is resumed

	/// action to be performed when the timer is released/deleted
	/// @param elapsedS total elapsed time in seconds
	virtual void onRelease(double /*elapsedS*/) {}

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
	bool valid = true;		///< true as long as not released

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

	virtual ~Timer();				///< if not released, reports duration to all observers

	virtual void pause();			///< pauses the timer and reports duration to all observers
	virtual void resume();			///< resumes the timer
	virtual void release();			///< stops the timer and reports duration to all observers
};

#endif