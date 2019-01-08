/************************************************************************************************
 The application Pic2Sym approximates images by a
 grid of colored symbols with colored backgrounds.

 Copyrights from the libraries used by the program:
 - (c) 2016 Boost (www.boost.org)
		License: <http://www.boost.org/LICENSE_1_0.txt>
			or doc/licenses/Boost.lic
 - (c) 2015 OpenCV (www.opencv.org)
		License: <http://opencv.org/license.html>
            or doc/licenses/OpenCV.lic
 - (c) 2015 The FreeType Project (www.freetype.org)
		License: <http://git.savannah.gnu.org/cgit/freetype/freetype2.git/plain/docs/FTL.TXT>
	        or doc/licenses/FTL.txt
 - (c) 1997-2002 OpenMP Architecture Review Board (www.openmp.org)
   (c) Microsoft Corporation (Visual C++ implementation for OpenMP C/C++ Version 2.0 March 2002)
		See: <https://msdn.microsoft.com/en-us/library/8y6825x5(v=vs.140).aspx>
 - (c) 1995-2013 zlib software (Jean-loup Gailly and Mark Adler - see: www.zlib.net)
		License: <http://www.zlib.net/zlib_license.html>
            or doc/licenses/zlib.lic
 
 (c) 2016-2019 Florin Tulba <florintulba@yahoo.com>

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
 ***********************************************************************************************/

#include "img.h"
#include "imgSettings.h"

#pragma warning ( push, 0 )

#include <iostream>

#include "boost_filesystem_operations.h"
#include <opencv2/imgcodecs/imgcodecs.hpp>

#pragma warning ( pop )

using namespace std;
using namespace boost::filesystem;
using namespace cv;

unsigned ImgSettings::VERSION_FROM_LAST_IO_OP = UINT_MAX;

bool Img::reset(const Mat &source_) {
	if(source_.empty())
		return false;

	source = source_;
	color = source.channels() > 1;
	return true;
}

bool Img::reset(const stringType &picName) {
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

uniquePtr<IfImgSettings> ImgSettings::clone() const {
	return makeUnique<ImgSettings>(*this);
}

bool ImgSettings::olderVersionDuringLastIO() {
	return VERSION_FROM_LAST_IO_OP < VERSION;
}
