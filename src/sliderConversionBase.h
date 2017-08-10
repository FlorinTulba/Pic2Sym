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
#	error Should not include this header when UNIT_TESTING is defined

#else // UNIT_TESTING not defined

#ifndef H_SLIDER_CONVERSION_BASE
#define H_SLIDER_CONVERSION_BASE

#pragma warning ( push, 0 )

#include <memory>

#pragma warning ( pop )

/// Base class for defining the parameters used while interpreting/generating the value of a slider
class SliderConvParams /*abstract*/ {
protected:
	int maxSlider; ///< largest slider value (at least 1)

public:
	/// Ensure a positive maxSlider field 
	SliderConvParams(int maxSlider_) : maxSlider((maxSlider_<1) ? 1 : maxSlider_) {}

	inline int getMaxSlider() const { return maxSlider; }

	virtual ~SliderConvParams() = 0 {}
};

/// Base class for performing conversions from and to slider range
class SliderConverter /*abstract*/ {
protected:
	std::unique_ptr<const SliderConvParams> sp; ///< parameters required for interpreting/generating slider values

public:
	/// Take ownership of the parameter
	SliderConverter(std::unique_ptr<const SliderConvParams> sp_) : sp(std::move(sp_)) {}

	SliderConverter(const SliderConverter&) = delete;
	void operator=(const SliderConverter&) = delete;

	virtual double fromSlider(int sliderPos) const = 0;
	virtual int toSlider(double actualValue) const = 0;

	virtual ~SliderConverter() = 0 {}
};

#endif // H_SLIDER_CONVERSION_BASE

#endif // UNIT_TESTING
