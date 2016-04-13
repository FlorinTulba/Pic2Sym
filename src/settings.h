/****************************************************************************************
 The application Pic2Sym approximates images by a
 grid of colored symbols with colored backgrounds.

 This file was created on 2016-4-8
 and belongs to the Pic2Sym project.

 Copyrights from the libraries used by the program:
 - (c) 2015 Boost (www.boost.org)
   License: <http://www.boost.org/LICENSE_1_0.txt>
            or doc/licenses/Boost.lic
 - (c) 2015 The FreeType Project (www.freetype.org)
   License: <http://git.savannah.gnu.org/cgit/freetype/freetype2.git/plain/docs/FTL.TXT>
	        or doc/licenses/FTL.txt
 - (c) 2015 OpenCV (www.opencv.org)
   License: <http://opencv.org/license.html>
            or doc/licenses/OpenCV.lic
 
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
 ****************************************************************************************/

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
	void serialize(Archive &ar, const unsigned version) {
		ar & ss & is & ms;
	}
	friend class boost::serialization::access;

public:
	static const unsigned // Limits  
		MIN_FONT_SIZE, MAX_FONT_SIZE, DEF_FONT_SIZE,
		MAX_THRESHOLD_FOR_BLANKS,
		MIN_H_SYMS, MAX_H_SYMS,
		MIN_V_SYMS, MAX_V_SYMS;

	static bool isBlanksThresholdOk(unsigned t) { return t < MAX_THRESHOLD_FOR_BLANKS; }
	static bool isHmaxSymsOk(unsigned syms) { return syms>=MIN_H_SYMS && syms<=MAX_H_SYMS; }
	static bool isVmaxSymsOk(unsigned syms) { return syms>=MIN_V_SYMS && syms<=MAX_V_SYMS; }
	static bool isFontSizeOk(unsigned fs) { return fs>=MIN_FONT_SIZE && fs<=MAX_FONT_SIZE; }

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