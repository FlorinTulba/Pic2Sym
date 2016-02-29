/**************************************************************************************
 This file belongs to the 'Pic2Sym' application, which
 approximates images by a grid of colored symbols with colored backgrounds.

 Project:     Pic2Sym 
 File:        main.cpp
 
 Author:      Florin Tulba
 Created on:  2016-1-8

 Copyrights from the libraries used by 'Pic2Sym':
 - © 2015 Boost (www.boost.org)
   License: http://www.boost.org/LICENSE_1_0.txt
            or doc/licenses/Boost.lic
 - © 2015 The FreeType Project (www.freetype.org)
   License: http://git.savannah.gnu.org/cgit/freetype/freetype2.git/plain/docs/FTL.TXT
	        or doc/licenses/FTL.txt
 - © 2015 OpenCV (www.opencv.org)
   License: http://opencv.org/license.html
            or doc/licenses/OpenCV.lic
 
 © 2016 Florin Tulba <florintulba@yahoo.com>

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
 **************************************************************************************/

#include "controller.h"

#include <boost/filesystem/operations.hpp>

using namespace std;
using namespace cv;
using namespace boost::filesystem;

// Prevents closing the console before the user sees an error message
void pauseAfterError() {
	string line;
	cout<<endl<<"Press Enter to leave"<<endl;
	getline(cin, line);
}

// Proper usage
void showUsage() {
	cout<<"Usage:"<<endl;
	cout<<"There are 2 launch modes:"<<endl;
	cout<<"A) Normal launch mode (no parameters)"<<endl;
	cout<<"		Pic2Sym.exe"<<endl<<endl;
	cout<<"B) View mismatches launch mode (Support for Unit Testing, using 1 parameters)"<<endl;
	cout<<"		Pic2Sym.exe \"<testTitle>\""<<endl<<endl;
	pauseAfterError();
}

void copyrightNotice() {
	cout<<string(80, '=')<<endl;
	cout<<"Running 'Pic2Sym' application, which approximates images by a grid of colored"<<endl
		<<"symbols with colored backgrounds."<<endl<<endl<<endl;
	cout<<"(c) 2016 Florin Tulba  <florintulba@yahoo.com>"<<endl<<endl<<endl;
	cout<<"The entire project can be found at: <https://github.com/FlorinTulba/Pic2Sym>" <<endl<<endl;
	cout<<"This program is free software: you can use its results,\n" \
		"redistribute it and/or modify it under the terms of the GNU\n" \
		"Affero General Public License version 3 as published by the\n" \
		"Free Software Foundation.\n\n" \
		"This program is distributed in the hope that it will be useful,\n" \
		"but WITHOUT ANY WARRANTY; without even the implied warranty of\n" \
		"MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the\n" \
		"GNU Affero General Public License for more details.\n\n" \
		"You should have received a copy of the GNU Affero General Public License\n" \
		"along with this program ('agpl-3.0.txt').\n" \
		"If not, see <http://www.gnu.org/licenses/agpl-3.0.txt>."<<endl;
	cout<<string(80, '=')<<endl<<endl;
}

// Normal mode launch. appFile is a path ending in Pic2Sym.exe
void normalLaunch(const string &appFile) {
	copyrightNotice();

	Settings s(move(MatchSettings(appFile)));
	Controller c(s);
	Controller::handleRequests();
}

/*
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
