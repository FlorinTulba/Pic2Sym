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

#ifndef H_SETTINGS_BASE
#define H_SETTINGS_BASE

#pragma warning ( push, 0 )

#include <iostream>

#pragma warning ( pop )

// Forward declarations
struct ISymSettings;
struct IfImgSettings;
struct IMatchSettings;

/**
Interface providing read-only access to all parameters required for transforming images.
Allows serialization and feeding such objects to output streams.
*/
struct ISettings /*abstract*/ {
	// Static validator methods
	static bool isBlanksThresholdOk(unsigned t);
	static bool isHmaxSymsOk(unsigned syms);
	static bool isVmaxSymsOk(unsigned syms);
	static bool isFontSizeOk(unsigned fs);

	virtual const ISymSettings& getSS() const = 0;	///< returns existing symbols settings
	virtual const IfImgSettings& getIS() const = 0;	///< returns existing image settings
	virtual const IMatchSettings& getMS() const = 0;///< returns existing match settings

	virtual ~ISettings() = 0 {}
};

std::ostream& operator<<(std::ostream &os, const ISettings &s);

/// The ISettings interface plus accessors for settings modification
struct ISettingsRW /*abstract*/ : ISettings {
	virtual ISymSettings& refSS() = 0;		///< allows current symbols settings to be changed
	virtual IfImgSettings& refIS() = 0;		///< allows current image settings to be changed
	virtual IMatchSettings& refMS() = 0;	///< allows current match settings to be changed

	virtual ~ISettingsRW() = 0 {}
};

#endif // H_SETTINGS_BASE
