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

#ifndef H_TRANSFORM_SUPPORT_BASE
#define H_TRANSFORM_SUPPORT_BASE

#pragma warning ( push, 0 )

#include <opencv2/core/core.hpp>

#pragma warning ( pop )

/// Interface for TransformSupport* classes (Initializing and updating draft matches)
struct ITransformSupport /*abstract*/ {
	/// Initializes the drafts when a new image needs to be approximated
	virtual void initDrafts(bool isColor, unsigned patchSz,
							unsigned patchesPerCol, unsigned patchesPerRow) = 0;

	/// Resets the drafts when current image needs to be approximated in a different context
	virtual void resetDrafts(unsigned patchesPerCol) = 0;

	/**
	Approximates row r of patches of size patchSz from an image with given width.
	It checks only the symbols with indices in range [fromSymIdx, upperSymIdx).
	*/
	virtual void approxRow(int r, int width, unsigned patchSz,
						   unsigned fromSymIdx, unsigned upperSymIdx, cv::Mat &result) = 0;

	virtual ~ITransformSupport() = 0 {}
};

#endif // H_TRANSFORM_SUPPORT_BASE
