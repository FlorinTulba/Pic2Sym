/**********************************************************
 Project:     Pic2Sym
 File:        transform.h

 Author:      Florin Tulba
 Created on:  2016-1-6
 
 Copyright (c) 2016 Florin Tulba
 **********************************************************/

#ifndef H_TRANSFORM
#define H_TRANSFORM

#include "config.h"
#include "match.h"
#include "img.h"

#include <string>
#include <vector>

#include <opencv2/imgcodecs.hpp>

class Settings;		// global settings
class Controller;	// data & views manager

// Transformer allows images to be approximated as a table of colored symbols from font files.
class Transformer final {
	const Controller &ctrler;	// data & views manager

	const Settings &cfg;		// general configuration
	MatchEngine &me;			// approximating patches
	Img &img;					// current image to process

	cv::Mat result;				// the result of the transformation

	void createOutputFolder();

public:
	Transformer(const Controller &ctrler_, const Settings &cfg_,
				MatchEngine &me_, Img &img_); // use initial configuration

	void run();				// applies the configured transformation onto current/new image

	const cv::Mat& getResult() const { return result; }
};

#endif