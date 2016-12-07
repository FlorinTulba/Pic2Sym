/************************************************************************************************
 The application Pic2Sym approximates images by a
 grid of colored symbols with colored backgrounds.

 This file belongs to the Pic2Sym project.

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
 ***********************************************************************************************/

#ifndef H_SETTINGS
#define H_SETTINGS

#include "matchSettings.h"
#include "imgSettings.h"
#include "symSettings.h"

class Controller;

/// Envelopes all parameters required for transforming images
class Settings {
	SymSettings ss;		///< parameters concerning the symbols set used for approximating patches
	ImgSettings is;		///< contains max count of horizontal & vertical patches to process
	MatchSettings ms;	///< settings used during approximation process
	friend class Controller; ///< the unique setter of ss, is, ms (apart serialization)

	/**
	Loads or saves a Settings object.

	@param ar the source/target archive
	@param version When loading (overwriting *this with the Settings from ar),
	it represents the version of the object loaded from ar;
	When saving to ar, it's the last version of Settings
	*/
	template<class Archive>
	void serialize(Archive &ar, const unsigned /*version*/) {
		ar & ss & is & ms;
	}
	friend class boost::serialization::access;

public:
	static bool isBlanksThresholdOk(unsigned t);
	static bool isHmaxSymsOk(unsigned syms);
	static bool isVmaxSymsOk(unsigned syms);
	static bool isFontSizeOk(unsigned fs);

	/**
	Creates a complete set of settings required during image transformations.

	@param ms_ incoming parameter copied to ms field.
	*/
	Settings(const MatchSettings &ms_);
	Settings(); ///< Creates Settings with empty MatchSettings

	const SymSettings& symSettings() const { return ss; }
	const ImgSettings& imgSettings() const { return is; }
	const MatchSettings& matchSettings() const { return ms; }

	friend std::ostream& operator<<(std::ostream &os, const Settings &s);
};

BOOST_CLASS_VERSION(Settings, 0)

#endif