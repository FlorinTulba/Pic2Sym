/**********************************************************
 Project:     Pic2Sym
 File:        img.cpp

 Author:      Florin Tulba
 Created on:  2015-12-21
 
 Copyright (c) 2016 Florin Tulba
 **********************************************************/

#include "img.h"

#include <iostream>

#include <boost/filesystem/operations.hpp>
#include <opencv2/imgcodecs.hpp>

using namespace std;
using namespace boost::filesystem;
using namespace cv;

bool Img::reset(const cv::Mat &source_) {
	if(source_.empty())
		return false;

	source = source_;
	color = source.channels() > 1;
	return true;
}

bool Img::reset(const string &picName) {
	path newPic(absolute(picName));
	if(imgPath.compare(newPic) == 0)
		return true; // image already in use

	const Mat source_ = imread(picName, ImreadModes::IMREAD_UNCHANGED);
	if(!reset(source_)) {
		cerr<<"Couldn't read image "<<picName<<endl;
		return false;
	}

	imgPath = std::move(newPic);
	imgName = imgPath.stem().string();

	cout<<"The image to process is "<<imgPath<<" (";
	if(color)
		cout<<"Color";
	else
		cout<<"Grayscale";
	cout<<" w="<<source.cols<<"; h="<<source.rows<<')'<<endl<<endl;
	return true;
}

Mat Img::resized(const Config &cfg) {
	if(source.empty()) {
		cerr<<"No image set yet"<<endl;
		throw logic_error("No image set yet");
	}

	const int initW = source.cols, initH = source.rows;
	const double initAr = initW / (double)initH;
	const unsigned patchSz = cfg.getFontSz();
	unsigned w = min(patchSz*cfg.getMaxHSyms(), (unsigned)initW),
			h = min(patchSz*cfg.getMaxVSyms(), (unsigned)initH);
	w -= w%patchSz;
	h -= h%patchSz;

	if(w / (double)h > initAr) {
		w = (unsigned)round(h*initAr);
		w -= w%patchSz;
	} else {
		h = (unsigned)round(w/initAr);
		h -= h%patchSz;
	}

	if(w==initW && h==initH)
		res = source;
	else {
		resize(source, res, Size(w, h), 0, 0, CV_INTER_AREA);
		cout<<"Resized to ("<<w<<'x'<<h<<')'<<endl;
	}

	cout<<"The result will be "<<w/patchSz<<" symbols wide and "<<h/patchSz<<" symbols high."<<endl<<endl;

	return res;
}
