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

// Transformer allows images to be approximated as a table of colored symbols from font files.
class Transformer final {
public:
	typedef std::vector<std::vector<const cv::Mat>> VVMat;
	typedef VVMat::const_iterator VVMatCIt;
	typedef std::pair<VVMatCIt, VVMatCIt> VVMatCItPair;

private:
	Controller &ctrler;			// data & views manager

	Config cfg;					// general configuration
	FontEngine fe;				// symbols set manager
	Img img;					// current image to process

	VVMat symsSet;	// set of symbols&inverses + 4 masks

	cv::Mat result;				// the result of the transformation

	std::string symsIdReady;	// type of symbols ready to use for transformation
	std::string getIdForSymsToUse(); // type of the symbols determined by fe & cfg

public:
	Transformer(Controller &ctrler_, const std::string &cmd); // use initial configuration

	void updateSymbols();	// using different charmap - also useful for displaying these changes
	void run();				// applies the configured transformation onto current/new image

	// Needed to display the cmap - returns a pair of symsSet iterators
	VVMatCItPair getSymsRange(unsigned from, unsigned count) const;

	Config& getCfg() { return cfg; }
	FontEngine& getFe() { return fe; }
	Img& getImg() { return img; }

	const cv::Mat& getResult() const { return result; }
};

#endif