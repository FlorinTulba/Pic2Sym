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
	bool color;							// color / grayscale

public:
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