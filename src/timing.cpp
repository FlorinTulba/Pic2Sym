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

#include "timing.h"

using namespace std;
using namespace std::chrono;

Timer::Timer(const vector<std::shared_ptr<ITimerActions>> &observers_) :
		observers(observers_), lastStart(high_resolution_clock::now()), elapsedS(0.) {
	for(auto observer : observers)
		observer->onStart();
}

Timer::Timer(std::shared_ptr<ITimerActions> observer) :
		Timer(vector<std::shared_ptr<ITimerActions>> { observer }) {}

Timer::Timer(Timer &&other) :
		observers(std::move(const_cast<vector<std::shared_ptr<ITimerActions>>&>(other.observers))),
		lastStart(other.lastStart), elapsedS(other.elapsedS),
		paused(other.paused), valid(other.valid) {
	other.valid = false;
}

Timer::~Timer() {
	if(!valid)
		return;

	release();
}

double Timer::elapsed() const {
	if(!valid)
		return 0.;

	if(paused)
		return elapsedS.count();

	auto durationToReport = elapsedS;
	durationToReport += high_resolution_clock::now() - lastStart;

	return durationToReport.count();
}

void Timer::cancel(const string &reason/* = "The task was canceled"*/) {
	if(!valid)
		return;

	valid = false;

	for(auto observer : observers)
		observer->onCancel(reason);
}

void Timer::pause() {
	if(!valid || paused)
		return;

	elapsedS += high_resolution_clock::now() - lastStart;

	paused = true;

	for(auto observer : observers)
		observer->onPause(elapsedS.count());
}

void Timer::resume() {
	if(!valid || !paused)
		return;

	paused = false;

	lastStart = high_resolution_clock::now();

	for(auto observer : observers)
		observer->onResume();
}

void Timer::release() {
	if(!valid)
		return;

	valid = false;

	if(!paused)
		elapsedS += high_resolution_clock::now() - lastStart;

	for(auto observer : observers)
		observer->onRelease(elapsedS.count());
}
