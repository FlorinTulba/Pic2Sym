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

#ifdef UNIT_TESTING
#	error Should not include this file when UNIT_TESTING is defined

#else // UNIT_TESTING not defined

#include "sliderConversion.h"
#include "misc.h"

using namespace std;

ProportionalSliderValue::Params::Params(int maxSlider_, double maxVal_) :
		SliderConvParams(maxSlider_), maxVal(maxVal_) {
	if(maxVal_ <= 0.)
		THROW_WITH_CONST_MSG("ProportionalSliderValue::Params should get a positive maxVal_ parameter",
			invalid_argument);
}

ProportionalSliderValue::ProportionalSliderValue(std::uniquePtr<const Params> sp_) :
		SliderConverter(std::move(sp_)) {
	if(!sp) // Testing on sp, since sp_ was moved to sp
		THROW_WITH_CONST_MSG("ProportionalSliderValue received a nullptr Params parameter",
			invalid_argument);
}

double ProportionalSliderValue::fromSlider(int sliderPos) const {
	const Params *lsp = dynamic_cast<const Params*>(sp.get());
	return (double)sliderPos * lsp->maxVal / (double)lsp->getMaxSlider();
}

int ProportionalSliderValue::toSlider(double actualValue) const {
	const Params *lsp = dynamic_cast<const Params*>(sp.get());
	return int(.5 + actualValue * (double)lsp->getMaxSlider() / lsp->maxVal); // performs rounding
}

#endif // UNIT_TESTING
