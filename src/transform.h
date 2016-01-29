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

class Controller; // data & views manager

/*
Transformer allows images to be approximated as a table of colored symbols from font files.
*/
class Transformer final {
	Controller &ctrler;			// data & views manager

	Config cfg;					// general configuration
	FontEngine fe;				// symbols set manager
	Img img;					// current image to process

	std::vector<const cv::Mat*> pNegatives;				// pointers to glyphs' inverses
	std::vector<std::pair<cv::Mat, cv::Mat>> symsSet;	// current symbol set & its inverse in Mat format

	cv::Mat result;				// the result of the transformation

	std::string symsIdReady;	// type of symbols ready to use for transformation
	std::string getIdForSymsToUse(); // type of the symbols determined by fe & cfg

public:
	Transformer(Controller &ctrler_, const std::string &cmd); // use initial configuration

	void updateSymbols();	// using different charmap - also useful for displaying these changes
	void run();				// applies the configured transformation onto current/new image

	// Needed to display the cmap
	const std::vector<const cv::Mat*>& getNegatives() const { return pNegatives; }

	Config& getCfg() { return cfg; }
	FontEngine& getFe() { return fe; }
	Img& getImg() { return img; }

	const cv::Mat& getResult() const { return result; }
};

#endif