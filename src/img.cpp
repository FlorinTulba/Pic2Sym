/**********************************************************
 Project:     Pic2Sym
 File:        img.cpp

 Author:      Florin Tulba
 Created on:  2015-12-21
 
 Copyright (c) 2016 Florin Tulba
 **********************************************************/

#include "img.h"

#include <iostream>

#include <opencv2/imgcodecs.hpp>

using namespace std;
using namespace boost::filesystem;
using namespace cv;

bool Img::reset(const string &picName) {
	path newPic(absolute(picName));
	if(imgPath.compare(newPic) == 0)
		return true; // image already in use

	Mat source_ = imread(picName, ImreadModes::IMREAD_UNCHANGED);
	if(source_.data == nullptr) {
		cerr<<"Couldn't read image "<<picName<<endl;
		return false;
	}

	source = source_;
	imgPath = move(newPic);
	imgName = imgPath.stem().string();

	rgb = source.channels() > 1;
	cout<<"Processing "<<imgPath<<" (";
	if(rgb)
		cout<<"RGB";
	else
		cout<<"Grayscale";
	cout<<" w="<<source.cols<<"; h="<<source.rows<<')'<<endl<<endl;
	return true;
}

Mat Img::resized(const Config &cfg, cv::Mat *grayVersion/* = nullptr*/) const {
	if(source.data == nullptr) {
		cerr<<"No image set yet"<<endl;
		throw logic_error("No image set yet");
	}

	int initW = source.cols, initH = source.rows;
	double initAr = initW / (double)initH;
	unsigned patchSz = cfg.getFontSz(),
		w = min(patchSz*cfg.getOutW(), (unsigned)initW),
		h = min(patchSz*cfg.getOutH(), (unsigned)initH);
	w -= w%patchSz;
	h -= h%patchSz;
	double ar = w / (double)h;
	if(ar > initAr) {
		w = (unsigned)round(h*initAr);
		w -= w%patchSz;
	} else {
		h = (unsigned)round(w/initAr);
		h -= h%patchSz;
	}
	Mat resized_;
	if(w==initW && h==initH)
		resized_ = source;
	else {
		resize(source, resized_, Size(w, h), 0, 0, CV_INTER_AREA);
		cout<<"Resized to ("<<w<<'x'<<h<<')'<<endl<<endl;
	}

	cout<<"The result will be "<<w/patchSz<<" characters wide and "<<h/patchSz<<" characters high."<<endl<<endl;

	if(grayVersion != nullptr) {
		if(rgb)
			cvtColor(resized_, *grayVersion, COLOR_RGB2GRAY);
		else
			*grayVersion = resized_;
	}

	return resized_;
}
