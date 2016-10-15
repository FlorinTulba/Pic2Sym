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

// Serialization support for cv::Mat 

// Code adapted from the one provided by user1520427 in thread:
// http://stackoverflow.com/questions/4170745/serializing-opencv-mat-vec3f

#ifndef H_MAT_SERIALIZATION
#define H_MAT_SERIALIZATION

#include <opencv2/core/core.hpp>

namespace boost {
	namespace serialization {
		template<class Archive>
		void serialize(Archive &ar, cv::Mat& mat, const unsigned int) {
			ar & mat.rows & mat.cols;

			ar & mat.flags; // provides the matrix type and continuity flag

			const bool continuous = mat.isContinuous();

			if(Archive::is_loading::value)
				mat.create(mat.rows, mat.cols, mat.type());

			if(continuous) {
				const auto data_size = mat.total() * mat.elemSize();
				ar & boost::serialization::make_array(mat.ptr(), data_size);
			} else {
				const auto row_size = mat.cols * mat.elemSize();
				for(int i = 0; i < mat.rows; i++)
					ar & boost::serialization::make_array(mat.ptr(i), row_size);
			}
		}
	}
}

#endif // H_MAT_SERIALIZATION