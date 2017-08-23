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

#include "symsSerialization.h"

#pragma warning ( push, 0 )

#include <fstream>
#include <iomanip>

#pragma warning ( pop )

using namespace std;
using namespace cv;

namespace ut {
	void saveSymsSelection(const stringType &destFile,
						   const list<const Mat> &symsSelection) {
		ofstream ofs(destFile);
		ofs<<symsSelection.size()<<endl; // first line specifies the number of symbols in the list
		for(const Mat &m : symsSelection) {
			const int rows = m.rows, cols = m.cols;
			ofs<<rows<<' '<<cols<<endl; // every symbol is preceded by a header with the number of rows and columns
			for(int r = 0; r < rows; ++r) {
				for(int c = 0; c < cols; ++c)
					ofs<<setw(3)<<right // align values to the right
						<<(unsigned)m.at<unsigned char>(r, c)
						<<' '; // pixel values are delimited by space
				ofs<<endl;
			}
		}
	}

	void loadSymsSelection(const stringType &srcFile,
						   vector<const Mat> &symsSelection) {
		ifstream ifs(srcFile);

		unsigned symsCount = 0U;
		ifs>>symsCount;
		symsSelection.reserve(symsCount);

		for(unsigned symIdx = 0U; symIdx < symsCount; ++symIdx) {
			int rows, cols;
			ifs>>rows>>cols; // every symbol is preceded by a header with the number of rows and columns
			assert(rows == cols);

			Mat symMat(rows, cols, CV_8UC1);
			for(int r = 0; r < rows; ++r) {
				for(int c = 0; c < cols; ++c) {
					unsigned v; ifs>>v;
					symMat.at<unsigned char>(r, c) = (unsigned char)v;
				}
			}

			symsSelection.push_back(symMat);
		}
	}
}
