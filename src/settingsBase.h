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

#ifndef H_SETTINGS_BASE
#define H_SETTINGS_BASE

#include "misc.h"

#pragma warning ( push, 0 )

#include <iostream>

#ifndef AI_REVIEWER_CHECK

#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/serialization/split_member.hpp>
#include <boost/serialization/version.hpp>

#endif // AI_REVIEWER_CHECK not defined

#pragma warning ( pop )

// Forward declarations
class SymSettings;
class ImgSettings;
class MatchSettings;

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

	virtual const SymSettings& getSS() const = 0;	///< returns existing symbols settings
	virtual const ImgSettings& getIS() const = 0;	///< returns existing image settings
	virtual const MatchSettings& getMS() const = 0;	///< returns existing match settings

	friend std::ostream& operator<<(std::ostream &os, const ISettings &s);

	virtual ~ISettings() = 0 {}

protected:
	/// Overwriting the read-only version not allowed, so it throws logic_error
	template<class Archive>
	void load(Archive&, const unsigned) {
		THROW_WITH_CONST_MSG("Don't use the read-only ISettings interface when loading new Settings!",
							 std::logic_error)
	}

	/// Saves *this to ar
	template<class Archive>
	void save(Archive &ar, const unsigned) const {
#ifndef AI_REVIEWER_CHECK
		ar << getSS() << getIS() << getMS();
#endif // AI_REVIEWER_CHECK not defined
	}
#ifndef AI_REVIEWER_CHECK
	BOOST_SERIALIZATION_SPLIT_MEMBER();
	friend class boost::serialization::access;
#endif // AI_REVIEWER_CHECK not defined
};

#ifndef AI_REVIEWER_CHECK
BOOST_CLASS_VERSION(ISettings, 0)
#endif // AI_REVIEWER_CHECK not defined


/// The ISettings interface plus accessors for settings modification
struct ISettingsRW /*abstract*/ : ISettings {
	virtual SymSettings& SS() = 0;		///< allows current symbols settings to be changed
	virtual ImgSettings& IS() = 0;		///< allows current image settings to be changed
	virtual MatchSettings& MS() = 0;	///< allows current match settings to be changed

	virtual ~ISettingsRW() = 0 {}

protected:
	/**
	Overwrites *this with the ISettingsRW object read from ar.

	@param ar source of the object to load
	@param version the version of the loaded ISettingsRW
	*/
	template<class Archive>
	void load(Archive &ar, const unsigned version) {
		UNREFERENCED_PARAMETER(version);

		// read user default match settings
#ifndef AI_REVIEWER_CHECK
		ar >> SS() >> IS() >> MS();
#endif // AI_REVIEWER_CHECK not defined
	}

	/// Saves *this to ar
	template<class Archive>
	void save(Archive &ar, const unsigned version) const {
		ISettings::save(ar, version);
	}
#ifndef AI_REVIEWER_CHECK
	BOOST_SERIALIZATION_SPLIT_MEMBER();
	friend class boost::serialization::access;
#endif // AI_REVIEWER_CHECK not defined
};

#ifndef AI_REVIEWER_CHECK
BOOST_CLASS_VERSION(ISettingsRW, 0)
#endif // AI_REVIEWER_CHECK not defined

#endif // H_SETTINGS_BASE