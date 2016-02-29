/**************************************************************************************
 This file belongs to the 'Pic2Sym' application, which
 approximates images by a grid of colored symbols with colored backgrounds.

 Project:     Pic2Sym 
 File:        img.h
 
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

#ifndef H_IMG
#define H_IMG

#include "config.h"

#include <string>

#include <boost/filesystem/path.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/serialization/split_member.hpp>
#include <boost/serialization/version.hpp>
#include <opencv2/imgproc/imgproc.hpp>

/*
Contains max count of horizontal & vertical patches to process.
The image is resized appropriately before processing.
*/
class ImgSettings {
	unsigned hMaxSyms;	// Count of resulted horizontal symbols
	unsigned vMaxSyms;	// Count of resulted vertical symbols

	template<class Archive>
	void load(Archive &ar, const unsigned version) {
		// It is useful to see which settings changed when loading
		ImgSettings defSettings(*this); // create as copy of previous values

		// read user default match settings
		ar >> defSettings.hMaxSyms >> defSettings.vMaxSyms;

		// these show message when there are changes
		setMaxHSyms(defSettings.hMaxSyms);
		setMaxVSyms(defSettings.vMaxSyms);
	}
	template<class Archive>
	void save(Archive &ar, const unsigned version) const {
		ar << hMaxSyms << vMaxSyms;
	}
	BOOST_SERIALIZATION_SPLIT_MEMBER();
	friend class boost::serialization::access;

public:
	// Constructor takes initial values just to present a valid sliders positions in Control Panel
	ImgSettings(unsigned hMaxSyms_, unsigned vMaxSyms_) :
		hMaxSyms(hMaxSyms_), vMaxSyms(vMaxSyms_) {}

	unsigned getMaxHSyms() const { return hMaxSyms; }
	void setMaxHSyms(unsigned syms);

	unsigned getMaxVSyms() const { return vMaxSyms; }
	void setMaxVSyms(unsigned syms);

	friend std::ostream& operator<<(std::ostream &os, const ImgSettings &is);
};

BOOST_CLASS_VERSION(ImgSettings, 0)

// Img provides necessary API for manipulating the images to transform
class Img final {
	boost::filesystem::path imgPath;	// path of current image
	std::string imgName;				// stem part of the image file name
	cv::Mat source, res;				// the original & resized image
	bool color = false;					// color / grayscale


#ifdef UNIT_TESTING
public: // Providing reset(Mat) as public for Unit Testing
#endif
	bool reset(const cv::Mat &source_);

public:
	/*
	Creates an Img object with default fields.
	
	The parameter just supports a macro mechanism that creates several object types
	with variable number of parameters.
	
	For Img, instead of 'Img field;', it would generate 'Img field();'   
	which is interpreted as a function declaration.

	Adding this extra param generates no harm in the rest of the project,
	but allows the macro to see it as object 'Img field(nullptr);', not function.
	*/
	Img(void** /*hackParam*/ = nullptr) {} 

	bool reset(const std::string &picName); // setting a new source image. Returns false for invalid images

	const cv::Mat& original() const { return source; }

	/*
	If possible, 'resized' method adapts the original image to the parameters of the transformation:
	- The image must fit within prescribed bounds
	- The image must preserve its original aspect ratio and cannot become larger
	*/
	cv::Mat resized(const ImgSettings &is, unsigned patchSz);

	bool isColor() const { return color; }	// color / grayscale image
	const std::string& name() const { return imgName; } // return the stem of the image file name
	const boost::filesystem::path& absPath() const { return imgPath; } // return the absolute path of the image file name

	const cv::Mat& getResized() const { return res; }
};

#endif