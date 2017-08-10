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

#ifndef H_SETTINGS
#define H_SETTINGS

#include "settingsBase.h"
#include "matchSettings.h"
#include "imgSettings.h"
#include "symSettings.h"

/// Envelopes all parameters required for transforming images
class Settings : public ISettingsRW {
protected:
	SymSettings ss;		///< parameters concerning the symbols set used for approximating patches
	ImgSettings is;		///< contains max count of horizontal & vertical patches to process
	MatchSettings ms;	///< settings used during approximation process

public:
	/**
	Creates a complete set of settings required during image transformations.

	@param ms_ incoming parameter copied to ms field.
	*/
	Settings(const MatchSettings &ms_);
	Settings(); ///< Creates Settings with empty MatchSettings

	// Read-only accessors
	const SymSettings& getSS() const override final;
	const ImgSettings& getIS() const override final;
	const MatchSettings& getMS() const override final;

	// Accessors for changing the settings
	SymSettings& refSS() override final;
	ImgSettings& refIS() override final;
	MatchSettings& refMS() override final;
};

#endif // H_SETTINGS