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

#if !defined UNIT_TESTING && !defined H_MATCH_SETTINGS_MANIP
#define H_MATCH_SETTINGS_MANIP

#include "matchSettings.h"
#include "misc.h"

#pragma warning ( push, 0 )

#include <string>

#include <boost/filesystem/path.hpp>

#pragma warning ( pop )

/// Singleton isolating the logic for the update of MatchSettings from disk in various scenarios
class MatchSettingsManip {
	MatchSettingsManip(const MatchSettingsManip&) = delete;
	void operator=(const MatchSettingsManip&) = delete;

protected:
	boost::filesystem::path defCfgPath;	///< Path of the original configuration file
	boost::filesystem::path cfgPath;	///< Path of the user configuration file

	/**
	Loading MatchSettings fields of a given version into ms.

	@param ms the target MatchSettings object
	@param ar the source of the object
	@param version what version is the loaded object

	@throw invalid_argument when loading an obsolete 'initMatchSettings.cfg'
	*/
	template<class Archive>
	void load(MatchSettings &ms, Archive &ar, const unsigned version) {
		UNREFERENCED_PARAMETER(ar);

		if(version < MatchSettings::VERSION) {
			/*
			MatchSettings is considered correctly initialized if its data is read from
			'res/defaultMatchSettings.txt'(most up-to-date file, which always exists) or
			'initMatchSettings.cfg'(if it exists and is newer than 'res/defaultMatchSettings.txt').

			Each launch of the application will either create / update 'initMatchSettings.cfg'
			if this doesn't exist / is older than 'res/defaultMatchSettings.txt'.

			Besides, anytime MatchSettings::VERSION is increased, 'initMatchSettings.cfg' becomes
			obsolete, so it must be overwritten with the fresh data from 'res/defaultMatchSettings.txt'.

			Initialized is set to true at the end of the construction.
			*/
			if(!ms.initialized) // can happen only when loading an obsolete 'initMatchSettings.cfg'
				THROW_WITH_CONST_MSG("Obsolete version of 'initMatchSettings.cfg'!", invalid_argument);

			// Point reachable while reading Settings with an older version of MatchSettings field
		}
	}

	/**
	Initializes the fields from initMatchSettings.cfg when this file exists and isn't obsolete,
	otherwise from res/defaultMatchSettings.txt.
	The latter file is used to conveniently alter the defaults during development.

	@param ms the MatchSettings to be initialized
	*/
	void initMatchSettings(MatchSettings &ms);
	
	friend class MatchSettings; // allows access to previous 2 methods

	/// creates 'initMatchSettings.cfg' with data from 'res/defaultMatchSettings.txt'
	void createUserDefaults(MatchSettings &ms);

	/// Creates the Singleton.
	MatchSettingsManip();

public:
	static MatchSettingsManip& instance(); ///< returns a reference to the singleton

	bool parseCfg(MatchSettings &ms, const boost::filesystem::path &cfgFile); ///< Loads the settings provided in cfgFile

	/// Overwrites current settings with those from initMatchSettings.cfg.
	void loadUserDefaults(MatchSettings &ms);
	void saveUserDefaults(const MatchSettings &ms) const; ///< Overwrites initMatchSettings.cfg with current settings
};

#endif