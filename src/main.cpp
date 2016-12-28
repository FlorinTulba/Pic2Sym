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

#include "controller.h"
#include "appStart.h"
#include "settings.h"
#include "study.h"
#include "ompTraceSwitch.h"

#pragma warning ( push, 0 )

#include <Windows.h>

#include <omp.h>

#include <boost/filesystem/operations.hpp>
#include <boost/optional/optional.hpp>

#include <opencv2/imgcodecs/imgcodecs.hpp>

#pragma warning ( pop )

using namespace std;
using namespace cv;
using namespace boost;
using namespace boost::filesystem;

/// Proper usage
void showUsage();

/// Prevents closing the console before the user sees an error message
void pauseAfterError();

/// Presents the misidentified symbols
void viewMismatches(const string &testTitle, const Mat &mismatches);

/// Presents the misfiltered symbols
void viewMisfiltered(const string &testTitle, const Mat &misfiltered);

extern const string copyrightText;
extern bool AllowReprocessingCases;

#ifdef GENERATE_OPEN_MP_TRACE
/// Definition of the lock for tracing messages generated by ompPrintf()
omp_lock_t ompTraceLock;
#endif // defined(GENERATE_OPEN_MP_TRACE)

namespace {
	const string hBar(80, '=');

	/// Displays Copyright text
	void copyrightNotice() {
		cout<<hBar<<endl;
		cout<<copyrightText<<endl;
		cout<<hBar<<endl<<endl;
	}

	/// Normal mode launch
	void normalLaunch() {
		// Ensure control over the number of threads during the application
		if(omp_get_dynamic())
			omp_set_dynamic(0);

		// Use as many threads as processors
		omp_set_num_threads(omp_get_num_procs());

		// Ensure no nested parallelism
		if(omp_get_nested())
			omp_set_nested(0);

#ifdef GENERATE_OPEN_MP_TRACE
		omp_init_lock(&ompTraceLock); // Initialize lock for trace messages generated by ompPrintf()
#endif // defined(GENERATE_OPEN_MP_TRACE)

		copyrightNotice();

		Settings s;
		Controller c(s);
		Controller::handleRequests();

#ifdef GENERATE_OPEN_MP_TRACE
		omp_destroy_lock(&ompTraceLock);
#endif // defined(GENERATE_OPEN_MP_TRACE)
	}

	/// Perform an image transformation under certain conditions.
	void timingScenario(const string &caseName,				///< title of the scenario
						const string &settingsPath,			///< path towards a `p2s` file prescribing the settings for the transformation
						const string &imgPath,				///< the image to transform
						const string &reportFilePath = ""	///< the file where to report the duration
						) {
		AllowReprocessingCases = true;

		Settings s;
		Controller c(s);

		if(!c.loadSettings(settingsPath))
			THROW_WITH_VAR_MSG("Couldn't load settings from `" + settingsPath + '`', runtime_error);

		if(!c.newImage(imgPath, true))
			THROW_WITH_VAR_MSG("Couldn't load the image `" + imgPath + '`', runtime_error);

		c.newSymsBatchSize(0); // ensure no drafts

		double durationS;
		if(!c.performTransformation(&durationS))
			THROW_WITH_VAR_MSG("Couldn't start the transformation for `" + caseName + '`', runtime_error);

		if(!reportFilePath.empty()) {
			ofstream ofs(reportFilePath, ios::app);
			ofs<<caseName<<'\t'<<durationS<<endl;
		}
	}

	/// Returns the Mat object contained within the Unit Testing report.
	optional<const Mat> contentOfReport(const string &testTitle, const string &contentDirName) {
		path contentDir(path(".").append("UnitTesting").append(contentDirName));
		if(!exists(contentDir)) {
			cerr<<"Expected work directory: <SolutionDir>/x64/<ConfigType>/ , but the actual one is: "<<absolute(".")<<'.'<<endl;
			cerr<<"It has to contain folder 'UnitTesting'."<<endl;
			pauseAfterError();
			return none;
		}

		path contentFile(absolute(contentDir).append(testTitle).concat(".jpg"));
		if(!exists(contentFile)) {
			cerr<<"There has to be a jpg file with the provided <testTitle> ("<<testTitle<<"), but file "<<contentFile<<" doesn't exist!"<<endl;
			pauseAfterError();
			return none;
		}

		const Mat content = imread(contentFile.string(), ImreadModes::IMREAD_UNCHANGED);
		if(content.empty()) {
			cerr<<"Invalid jpg file for "<<contentDirName<<": "<<contentFile<<'.'<<endl;
			pauseAfterError();
			return none;
		}

		return content;
	}

	/**
	View Mismatches Launch mode opens a Comparator window allowing the developer to observe
	the less fortunate approximations of reference patches.

	testTitle is the title of the Comparator window and
	is also the stem of the jpg file handled by the Comparator.
	*/
	void viewMismatchesMode(const string &testTitle) {
		optional<const Mat> mismatches = contentOfReport(testTitle, "Mismatches");
		if(!mismatches)
			return;

		viewMismatches(testTitle, mismatches.value());
	}

	/**
	View Misfiltered symbols Launch mode opens a window allowing the developer to observe
	the less fortunate filtered symbols.

	testTitle is the title of the window and
	is also the stem of the jpg file handled by the viewer.
	*/
	void viewMisfilteredMode(const string &testTitle) {
		optional<const Mat> misfiltered = contentOfReport(testTitle, "Misfiltered");
		if(!misfiltered)
			return;

		viewMisfiltered(testTitle, misfiltered.value());
	}

	/**
	The system might contain more versions of the dll-s required by this application.
	Some of such versions might not be appropriate for Pic2Sym.
	Therefore it is mandatory to provide the correct dll-s, especially when deploying
	the program on other machines.

	Some of the dll-s need to be selected while still loading the application (before it starts running).
	Placing these dll-s in a folder "Pic2Sym.exe.local" (near Pic2Sym.exe) is enough (This is a basic 
	DLL-s redirection technique).
	
	The dll-s loaded after the start of the application were conveniently copied into the same directory.
	However, "Pic2Sym.exe.local" is ignored for these dll-s when the application is installed
	in the default Program Files location, unless forcefully pointed with 'SetDllDirectory'.

	The plugins from Qt are a special dll category and can be located by:
	- either calling QCoreApplication::addLibraryPath("Pic2Sym.exe.local");
	
	- or creating 'qt.conf' file near Pic2Sym.exe containing:
		[Paths]
		Plugins=Pic2Sym.exe.local
	
	- or by setting QT_QPA_PLATFORM_PLUGIN_PATH in the local environment to:
		Pic2Sym.exe.local/platforms

	Last solution was the one adopted.
	*/
	void providePrivateDLLsPaths(const string &appPath) {
		const auto dllsPath = absolute(appPath).concat(".local");
		SetDllDirectory(dllsPath.wstring().c_str());
		_putenv_s("QT_QPA_PLATFORM_PLUGIN_PATH", 
				  path(dllsPath).append("platforms").string().c_str());
	}
} // anonymous namespace

/**
Starts the application in:
- study mode, when studying() is configured to return true from 'study.cpp'
- normal mode when there are no parameters
- timing mode when there are 5 parameters (how long it takes transforming an image in a certain context)
- unit test mode for 2 provided parameters

Next to 'Pic2Sym.exe' there should be:
- res/ folder with the resources required by the application
- Pic2Sym.exe.local/ folder with the used non-system dll-s (basic Dll redirection mechanism)

When serving the UnitTesting project, Pic2Sym.exe is called from the post-build process of unit testing.
There were 2 approaches for using Pic2Sym.exe as helper for UnitTesting project:

I. The current method for presenting the issues found by unit testing is to register them in a file
and visualize its entries when unit testing finishes.

II. Previous approach was to invoke 'Pic2Sym.exe' through a detached process (using CreateProcess).
However, after introducing the Dll redirection mechanism:
- calling separately 'Pic2Sym.exe' with the parameters for unit testing worked as expected
- calling 'Pic2Sym.exe' from unit testing as a detached process couldn't localize 'qwindows.dll' from Qt.
	See discussion from providePrivateDLLsPaths about that
*/
void main(int argc, char* argv[]) {
	// argv[0] is a path ending in 'Pic2Sym.exe'
	AppStart::determinedBy(argv[0]);
	providePrivateDLLsPaths(argv[0]);

	// Some matters need separate studying, so don't start the actual application when studying them
	if(studying()) {
		study(argc, argv);
		return;
	}
		
	if(1 == argc) { // no parameters
		normalLaunch();

	} else {
		const string firstParam(argv[1]);
		if(3 == argc) { // 2 parameters
			const string secondParam(argv[2]);

			if(firstParam.compare("mismatches") == 0) {
				viewMismatchesMode(secondParam);

			} else if(firstParam.compare("misfiltered") == 0) {
				viewMisfilteredMode(secondParam);

			} else {
				cerr<<"Invalid first parameter '"<<firstParam<<'\''<<endl;
				showUsage();
			}

		} else if(6 == argc) { // 5 parameters
			if(firstParam.compare("timing") == 0) {
				timingScenario(argv[2], argv[3], argv[4], argv[5]);

			} else {
				cerr<<"Invalid first parameter '"<<firstParam<<'\''<<endl;
				showUsage();
			}

		} else { // Wrong # of parameters
			cerr<<"There were "<<argc-1<<" parameters!"<<endl;
			showUsage();
		}
	}
}

#endif // UNIT_TESTING not defined
