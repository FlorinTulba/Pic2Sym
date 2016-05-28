/****************************************************************************************
 The application Pic2Sym approximates images by a
 grid of colored symbols with colored backgrounds.

 This file belongs to the Pic2Sym project.

 Copyrights from the libraries used by the program:
 - (c) 2015 Boost (www.boost.org)
   License: <http://www.boost.org/LICENSE_1_0.txt>
            or doc/licenses/Boost.lic
 - (c) 2015 The FreeType Project (www.freetype.org)
   License: <http://git.savannah.gnu.org/cgit/freetype/freetype2.git/plain/docs/FTL.TXT>
	        or doc/licenses/FTL.txt
 - (c) 2015 OpenCV (www.opencv.org)
   License: <http://opencv.org/license.html>
            or doc/licenses/OpenCV.lic
 
 (c) 2016 Florin Tulba <florintulba@yahoo.com>

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
 ****************************************************************************************/

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

Mat Img::resized(const ImgSettings &is, unsigned patchSz) {
	if(source.empty()) {
		cerr<<"No image set yet"<<endl;
		throw logic_error("No image set yet");
	}

	const int initW = source.cols, initH = source.rows;
	const double initAr = initW / (double)initH;
	unsigned w = min(patchSz*is.getMaxHSyms(), (unsigned)initW),
			h = min(patchSz*is.getMaxVSyms(), (unsigned)initH);
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

void ImgSettings::setMaxHSyms(unsigned syms) {
	if(syms == hMaxSyms)
		return;
	cout<<"hMaxSyms"<<" : "<<hMaxSyms<<" -> "<<syms<<endl;
	hMaxSyms = syms;
}

void ImgSettings::setMaxVSyms(unsigned syms) {
	if(syms == vMaxSyms)
		return;
	cout<<"vMaxSyms"<<" : "<<vMaxSyms<<" -> "<<syms<<endl;
	vMaxSyms = syms;
}

ostream& operator<<(ostream &os, const ImgSettings &is) {
	os<<"hMaxSyms"<<" : "<<is.hMaxSyms<<endl;
	os<<"vMaxSyms"<<" : "<<is.vMaxSyms<<endl;
	return os;
}