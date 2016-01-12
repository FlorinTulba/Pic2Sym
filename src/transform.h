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
#include "img.h"

#include <string>
#include <vector>

#include <opencv2/imgcodecs.hpp>

/*
Transformer allows images to be approximated as a table of colored symbols from font files.
*/
class Transformer {
	Config &cfg;				// general configuration
	FontEngine fe;				// charset manager
	Img img;					// current image to process
	bool newSettings = false;	// flag to signal the settings have changed

	std::vector<std::pair<cv::Mat, cv::Mat>> charset; // current charset & its inverse in Mat format

public:
	Transformer(Config &cfg_); // use initial configuration

	void reconfig();	// permits changing the configuration
	void run();			// applies the configured transformation onto current/new image
};

#endif