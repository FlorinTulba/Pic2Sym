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

#ifndef H_RESIZED_IMG
#define H_RESIZED_IMG

#include "resizedImgBase.h"

struct IfImgSettings; // forward declaration

/// ResizedImg is the version of the original image which is ready to be transformed
class ResizedImg : public IResizedImg {
protected:
	cv::Mat res;			///< the resized image

	unsigned patchSz;		///< patch / font size

public:
	/**
	If possible, it adapts the original image to the parameters of the transformation:
	- The image must fit within prescribed bounds
	- The image must preserve its original aspect ratio and cannot become larger
	*/
	ResizedImg(const cv::Mat &source, const IfImgSettings &is, unsigned patchSz_);
	ResizedImg(const ResizedImg&) = default;
	void operator=(const ResizedImg&) = delete;

	const cv::Mat& get() const override final { return res; }

	bool operator==(const ResizedImg &other) const;
	bool operator!=(const ResizedImg &other) const { return !(*this == other); }
};

#endif // H_RESIZED_IMG
