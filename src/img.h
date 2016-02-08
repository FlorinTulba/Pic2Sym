/**********************************************************
 Project:     Pic2Sym
 File:        img.h

 Author:      Florin Tulba
 Created on:  2015-12-21
 
 Copyright (c) 2016 Florin Tulba
 **********************************************************/

#ifndef H_IMG
#define H_IMG

#include "config.h"

#include <string>

#include <boost/filesystem/path.hpp>
#include <opencv2/imgproc/imgproc.hpp>

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
	cv::Mat resized(const Config &cfg);

	bool isColor() const { return color; }	// color / grayscale image
	const std::string& name() const { return imgName; } // return the stem of the image file name
	const boost::filesystem::path& absPath() const { return imgPath; } // return the absolute path of the image file name

	const cv::Mat& getResized() const { return res; }
};

#endif