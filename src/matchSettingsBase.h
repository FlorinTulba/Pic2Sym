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

#ifndef H_MATCH_SETTINGS_BASE
#define H_MATCH_SETTINGS_BASE

#pragma warning ( push, 0 )

#include "std_string.h"
#include "std_memory.h"
#include <iostream>

#pragma warning ( pop )

/// IMatchSettings class controls the matching parameters for transforming one or more images
struct IMatchSettings /*abstract*/ {
	/// 'normal' means actual result; 'hybrid' cosmeticizes the result
	virtual const bool& isHybridResult() const = 0;
	virtual IMatchSettings& setResultMode(bool hybridResultMode_) = 0;	///< Displays the update, if any

	/// power of factor controlling structural similarity
	virtual const double& get_kSsim() const = 0;
	virtual IMatchSettings& set_kSsim(double kSsim_) = 0;				///< Displays the update, if any

	/// power of factor controlling correlation aspect
	virtual const double& get_kCorrel() const = 0;
	virtual IMatchSettings& set_kCorrel(double kCorrel_) = 0;				///< Displays the update, if any

	/// power of factor for foreground glyph-patch correlation
	virtual const double& get_kSdevFg() const = 0;
	virtual IMatchSettings& set_kSdevFg(double kSdevFg_) = 0;			///< Displays the update, if any

	/// power of factor for contour glyph-patch correlation
	virtual const double& get_kSdevEdge() const = 0;
	virtual IMatchSettings& set_kSdevEdge(double kSdevEdge_) = 0;		///< Displays the update, if any

	/// power of factor for background glyph-patch correlation
	virtual const double& get_kSdevBg() const = 0;
	virtual IMatchSettings& set_kSdevBg(double kSdevBg_) = 0;			///< Displays the update, if any

	/// power of factor for the resulted glyph contrast
	virtual const double& get_kContrast() const = 0;
	virtual IMatchSettings& set_kContrast(double kContrast_) = 0;		///< Displays the update, if any

	/// power of factor targeting smoothness (mass-centers angle)
	virtual const double& get_kCosAngleMCs() const = 0;
	virtual IMatchSettings& set_kCosAngleMCs(double kCosAngleMCs_) = 0;	///< Displays the update, if any

	/// power of factor targeting smoothness (mass-center offset)
	virtual const double& get_kMCsOffset() const = 0;
	virtual IMatchSettings& set_kMCsOffset(double kMCsOffset_) = 0;		///< Displays the update, if any

	/// power of factor aiming fanciness, not correctness
	virtual const double& get_kSymDensity() const = 0;
	virtual IMatchSettings& set_kSymDensity(double kSymDensity_) = 0;	///< Displays the update, if any

	/// Using Blank character replacement under this threshold
	virtual unsigned getBlankThreshold() const = 0;
	virtual IMatchSettings& setBlankThreshold(unsigned threshold4Blank_) = 0;///< Displays the update, if any

#ifndef UNIT_TESTING
	/// loads user defaults or throws for obsolete / invalid file
	virtual void replaceByUserDefaults() = 0;
	virtual void saveAsUserDefaults() const = 0;///< save these as user defaults
#endif // UNIT_TESTING not defined

	/// Provides a representation of the settings in a verbose manner or not
	virtual const std::stringType toString(bool verbose) const = 0;

	/// @return a clone of current settings
	virtual std::uniquePtr<IMatchSettings> clone() const = 0;

	virtual ~IMatchSettings() = 0 {}
};

std::ostream& operator<<(std::ostream &os, const IMatchSettings &ms);

#endif // H_MATCH_SETTINGS_BASE
