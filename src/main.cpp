/****************************************************************************************
 The application Pic2Sym approximates images by a
 grid of colored symbols with colored backgrounds.

 This file was created on 2016-3-1
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

#include "controller.h"
#include "matchSettingsManip.h"
#include "settings.h"

#include <omp.h>

#include <boost/filesystem/operations.hpp>

using namespace std;
using namespace cv;
using namespace boost::filesystem;

/// Proper usage
void showUsage();

/// Prevents closing the console before the user sees an error message
void pauseAfterError();

extern const string copyrightText;

#ifdef _DEBUG
/// Definition of the lock for tracing messages generated by ompPrintf()
omp_lock_t ompTraceLock;
#endif // defined(_DEBUG)

namespace {
	/// Displays Copyright text
	void copyrightNotice() {
		const string hBar(80, '=');
		cout<<hBar<<endl;
		cout<<copyrightText<<endl;
		cout<<hBar<<endl<<endl;
	}

	/// Normal mode launch. appFile is a path ending in 'Pic2Sym.exe'
	void normalLaunch(const string &appFile) {
		// Ensure control over the number of threads during the application
		if(omp_get_dynamic())
			omp_set_dynamic(0);

		// Use as many threads as processors
		omp_set_num_threads(omp_get_num_procs());

		// Ensure no nested parallelism
		if(omp_get_nested())
			omp_set_nested(0);

#ifdef _DEBUG
		omp_init_lock(&ompTraceLock); // Initialize lock for trace messages generated by ompPrintf()
#endif // defined(_DEBUG)

		copyrightNotice();

		MatchSettingsManip::init(appFile);
		Settings s;
		Controller c(s);
		Controller::handleRequests();

#ifdef _DEBUG
		omp_destroy_lock(&ompTraceLock);
#endif // defined(_DEBUG)
	}

	/**
	View Mismatches Launch mode opens a Comparator window allowing the developer to observe
	the less fortunate approximations of reference patches.

	testTitle is the title of the Comparator window and
	is also the stem of the jpg file handled by the Comparator.
	*/
	void viewMismatchesMode(const string &testTitle) {
		path mismatchesDir(path(".").append("UnitTesting").append("Mismatches"));
		if(!exists(mismatchesDir)) {
			cerr<<"Expected work directory: <SolutionDir>/x64/<ConfigType>/ , but the actual one is: "<<absolute(".")<<'.'<<endl;
			cerr<<"It has to contain folder 'UnitTesting'."<<endl;
			pauseAfterError();
			return;
		}

		path mismatchesFile(absolute(mismatchesDir).append(testTitle).concat(".jpg"));
		if(!exists(mismatchesFile)) {
			cerr<<"There has to be a jpg file with the provided <testTitle> ("<<testTitle<<"), but file "<<mismatchesFile<<" doesn't exist!"<<endl;
			pauseAfterError();
			return;
		}

		const Mat mismatches = imread(mismatchesFile.string(), ImreadModes::IMREAD_UNCHANGED);
		if(mismatches.empty()) {
			cerr<<"Invalid jpg file for mismatches: "<<mismatchesFile<<'.'<<endl;
			pauseAfterError();
			return;
		}

		const int twiceTheRows = mismatches.rows, rows = twiceTheRows>>1, cols = mismatches.cols;
		const Mat reference = mismatches.rowRange(0, rows), // upper half is the reference
				result = mismatches.rowRange(rows, twiceTheRows); // lower half is the result

		// Comparator window size should stay within ~ 800x600
		// Enlarge up to 3 times if resulting rows < 600.
		// Enlarge also when resulted width would be less than 140 (width when the slider is visible)
		const double resizeFactor = max(140./cols, min(600./rows, 3.));

		ostringstream oss;
		oss<<"View mismatches for "<<testTitle;
		const string title(oss.str());

		Comparator comp;
		comp.setPos(0, 0);
		comp.permitResize();
		comp.resize(4+(int)ceil(cols*resizeFactor), 70+(int)ceil(rows*resizeFactor));
		comp.setTitle(title.c_str());
		comp.setStatus("Press Esc to leave.");
		comp.setReference(reference);
		comp.setResult(result, 90); // Emphasize the references 
	
		Controller::handleRequests();
	}
} // anonymous namespace

void main(int argc, char* argv[]) {
	switch(argc) {
		case 1:
			normalLaunch(argv[0]);
			break;
		
		case 2:
			viewMismatchesMode(argv[1]);
			break;
		
		default: // Wrong # of parameters
			cerr<<"There were "<<argc-1<<" parameters, while the application expects at most 1!"<<endl;
			showUsage();
	}
}
