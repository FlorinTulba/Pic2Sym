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
 
 (c) 2016, 2017 Florin Tulba <florintulba@yahoo.com>

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

#include "blur.h"
#include "floatType.h"
#include "misc.h"

using namespace std;
using namespace cv;

size_t BlurEngine::PixelsCountLargestData() {
	extern const unsigned Settings_MAX_FONT_SIZE;
	return Settings_MAX_FONT_SIZE * Settings_MAX_FONT_SIZE;
}

size_t BlurEngine::PixelsCountTinySym() {
	extern unsigned TinySymsSz();
	return TinySymsSz() * TinySymsSz();
}

BlurEngine::ConfiguredInstances& BlurEngine::configuredInstances() {
#pragma warning ( disable : WARN_THREAD_UNSAFE )
	static ConfiguredInstances configuredInstances_;
#pragma warning ( default : WARN_THREAD_UNSAFE )

	return configuredInstances_;
}

BlurEngine::ConfInstRegistrator::ConfInstRegistrator(const string &blurType, const BlurEngine &configuredInstance) {
	configuredInstances().emplace(blurType, &configuredInstance);
}

const BlurEngine& BlurEngine::byName(const string &blurType) {
	try {
		return *configuredInstances().at(blurType);
	} catch(out_of_range&) {
		THROW_WITH_VAR_MSG("Unknown blur type: '" + blurType + "' in " __FUNCTION__, invalid_argument);
	}
}

void BlurEngine::process(const Mat &toBlur, Mat &blurred, bool forTinySym) const {
	extern const unsigned Settings_MAX_FONT_SIZE;
	assert(!toBlur.empty() && toBlur.type() == CV_FC1 &&
		   toBlur.rows <= (int)Settings_MAX_FONT_SIZE && toBlur.cols <= (int)Settings_MAX_FONT_SIZE);

	blurred = Mat(toBlur.size(), toBlur.type(), 0.);
	
	doProcess(toBlur, blurred, forTinySym);
}
