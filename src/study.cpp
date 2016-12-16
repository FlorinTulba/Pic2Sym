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

#ifndef UNIT_TESTING

#include "study.h"
#include "misc.h"

#pragma warning ( push, 0 )

#include <chrono>
#include <string>
#include <iostream>

#pragma warning ( pop )

using namespace std;
using namespace std::chrono;

namespace {
	/// Simple timer class to measure the performance of some task
	class Timer {
	protected:
		const string taskName;	///< name of the monitored task
		time_point<high_resolution_clock> startedAt;	///< starting moment

	public:
		Timer(const string &taskName_ = "") : taskName(taskName_), startedAt(high_resolution_clock::now()) {}
		void operator=(const Timer&) = delete;
		~Timer() {
			const double elapsedS = elapsed();
			cout<<"Task "<<taskName<<" required: "<<elapsedS<<"s!"<<endl;
		}

		/// Returns the time elapse since the Timer was started (created)
		double elapsed() const {
			const duration<double> elapsedS = high_resolution_clock::now() - startedAt;
			return elapsedS.count();
		}
	};
} // anonymous namespace

bool studying() {
	return false;
}

void study(int argc, char* argv[]) {
	UNREFERENCED_PARAMETER(argc);
	UNREFERENCED_PARAMETER(argv);
}

#endif // UNIT_TESTING not defined
