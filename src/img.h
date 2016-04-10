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

#ifndef H_IMG
#define H_IMG

#include <string>

#include <boost/filesystem/path.hpp>
#include <opencv2/imgproc/imgproc.hpp>

// forward declaration
class ImgSettings;
class ResizedImg;

/// Img holds the data of the original image
class Img {
protected:
	boost::filesystem::path imgPath;	///< path of current image
	std::string imgName;				///< stem part of the image file name
	cv::Mat source;						///< the original image
	bool color = false;					///< color / grayscale

#ifdef UNIT_TESTING
public: // Providing reset(Mat) as public for Unit Testing
#endif
	bool reset(const cv::Mat &source_);

public:
	/**
	Creates an Img object with default fields.
	
	The parameter just supports a macro mechanism that creates several object types
	with variable number of parameters.
	
	For Img, instead of 'Img field;', it would generate 'Img field();'   
	which is interpreted as a function declaration.

	Adding this extra param generates no harm in the rest of the project,
	but allows the macro to see it as object 'Img field(nullptr);', not a function.
	*/
	Img(void** /*hackParam*/ = nullptr) {} 

	/// setting a new source image. Returns false for invalid images
	bool reset(const std::string &picName);

	const cv::Mat& original() const { return source; }

	bool isColor() const { return color; }	///< color / grayscale image
	const std::string& name() const { return imgName; } ///< return the stem of the image file name

	/// @return absolute path of the image file name
	const boost::filesystem::path& absPath() const { return imgPath; }
};

/// ResizedImg is the version of the original image which is ready to be transformed
class ResizedImg {
protected:
	cv::Mat res; ///< the resized image

public:
	const Img &original; ///< reference to initial image

	/**
	If possible, it adapts the original image to the parameters of the transformation:
	- The image must fit within prescribed bounds
	- The image must preserve its original aspect ratio and cannot become larger
	*/
	ResizedImg(const Img &img, const ImgSettings &is, unsigned patchSz);

	const cv::Mat& get() const { return res; }
};

#endif