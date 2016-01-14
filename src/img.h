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

#include <boost/filesystem.hpp>
#include <opencv2/imgproc/imgproc.hpp>

// Img provides necessary API for manipulating the images to transform
class Img {
	boost::filesystem::path imgPath;	// path of current image
	std::string imgName;				// stem part of the image file name
	cv::Mat source;						// the original image
	bool color;							// color / grayscale

public:
	Img() {}

	bool reset(const std::string &picName); // setting a new source image. Returns false for invalid images

	/*
	If possible, 'resized' method adapts the original image to the parameters of the transformation:
	- The image must fit within prescribed bounds
	- The image must preserve its original aspect ratio and cannot become larger

	It also returns a grayscale version of the resized image if grayVersion isn't NULL.
	*/
	cv::Mat resized(const Config &cfg, cv::Mat *grayVersion = nullptr) const;

	bool isColor() const { return color; }	// color / grayscale image
	const std::string& name() const { return imgName; } // return the stem of the image file name
};

#endif