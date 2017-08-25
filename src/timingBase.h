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

#ifndef H_TIMING_BASE
#define H_TIMING_BASE

#pragma warning ( push, 0 )

#include "std_string.h"
#include "misc.h"

#pragma warning ( pop )

// Timing jobs:

/// Interface for any observer of a timer (see the Timer class)
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
	virtual void onCancel(const std::stringType &reason = "") { UNREFERENCED_PARAMETER(reason); }

	virtual ~ITimerActions() = 0 {}
};

/// Getting the duration of a job
struct ITimerResult /*abstract*/ {
	virtual double elapsed() const = 0;	///< reports elapsed seconds

	virtual ~ITimerResult() = 0 {}
};

/// Commands for an alive Timer: pause/resume and cancel
struct IActiveTimer /*abstract*/ {
	virtual void pause() = 0;			///< pauses the timer and reports duration to all observers
	virtual void resume() = 0;			///< resumes the timer

	/// Cancels a timing task.
	/// @param reason explanation for cancellation
	virtual void cancel(const std::stringType &reason = "The task was canceled") = 0;

	virtual ~IActiveTimer() = 0 {}
};

#endif // H_TIMING_BASE
