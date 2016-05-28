/****************************************************************************************
 The application Pic2Sym approximates images by a
 grid of colored symbols with colored backgrounds.

 This file belongs to the Pic2Sym project.

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

#ifndef H_CONFIG
#define H_CONFIG

#include <string>

#include <boost/filesystem/path.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/serialization/split_member.hpp>
#include <boost/serialization/version.hpp>

/// MatchSettings class controls the matching parameters for transforming one or more images.
class MatchSettings {
public:
	static const unsigned VERSION = 1U; ///< version of MatchSettings class

private:
	boost::filesystem::path workDir;	///< Folder where the application was launched
	boost::filesystem::path defCfgPath;	///< Path of the original configuration file
	boost::filesystem::path cfgPath;	///< Path of the user configuration file

	double kSsim = 0.;				///< power of factor controlling structural similarity
	double kSdevFg = 0.;			///< power of factor for foreground glyph-patch correlation
	double kSdevEdge = 0.;			///< power of factor for contour glyph-patch correlation
	double kSdevBg = 0.;			///< power of factor for background glyph-patch correlation
	double kContrast = 0.;			///< power of factor for the resulted glyph contrast
	double kMCsOffset = 0.;			///< power of factor targeting smoothness (mass-center offset)
	double kCosAngleMCs = 0.;		///< power of factor targeting smoothness (mass-centers angle)
	double kSymDensity = 0.;		///< power of factor aiming fanciness, not correctness
	unsigned threshold4Blank = 0U;	///< Using Blank character replacement under this threshold

	/**
	MatchSettings is considered correctly initialized if its data is read from
	'res/defaultMatchSettings.txt'(most up-to-date file, which always exists) or
	'initMatchSettings.cfg'(if it exists and is newer than 'res/defaultMatchSettings.txt').

	Each launch of the application will either create / update 'initMatchSettings.cfg'
	if this doesn't exist / is older than 'res/defaultMatchSettings.txt'.

	Besides, anytime MatchSettings::VERSION is increased, 'initMatchSettings.cfg' becomes
	obsolete, so it must be overwritten with the fresh data from 'res/defaultMatchSettings.txt'.

	Initialized is set to false before calling 'loadUserDefaults' in the constructor
	and it is always set to true at the end of the construction.
	*/
	bool initialized = true;

	/**
	Loading a MatchSettings object of a given version.
	It overwrites *this, reporting any changes

	@param ar the source of the object
	@param version what version is the loaded object

	@throw invalid_argument when loading an obsolete 'initMatchSettings.cfg'
	*/
	template<class Archive>
	void load(Archive &ar, const unsigned version) {
		if(version < VERSION) {
			if(!initialized) // can happen only when loading an obsolete 'initMatchSettings.cfg'
				throw invalid_argument("Obsolete version of 'initMatchSettings.cfg'!");

			// Point reachable while reading Settings with an older version of MatchSettings field
		}

		// It is useful to see which settings changed when loading =>
		// Loading data in a temporary object and comparing with existing values.
		MatchSettings defSettings(*this); // create as copy of previous values

		// read user default match settings
		if(version > 0U) {
			ar >> defSettings.kSsim; // versions > 0 use kSsim
		} else {
			defSettings.kSsim = 0.; // version 0 didn't use kSsim
		}
		ar >> defSettings.kSdevFg >> defSettings.kSdevEdge >> defSettings.kSdevBg
			>> defSettings.kContrast
			>> defSettings.kMCsOffset >> defSettings.kCosAngleMCs
			>> defSettings.kSymDensity
			>> defSettings.threshold4Blank;

		// these show message when there are changes
		set_kSsim(defSettings.kSsim);
		set_kSdevFg(defSettings.kSdevFg);
		set_kSdevEdge(defSettings.kSdevEdge);
		set_kSdevBg(defSettings.kSdevBg);
		set_kContrast(defSettings.kContrast);
		set_kMCsOffset(defSettings.kMCsOffset);
		set_kCosAngleMCs(defSettings.kCosAngleMCs);
		set_kSymDensity(defSettings.kSymDensity);
		setBlankThreshold(defSettings.threshold4Blank);
	}

	/// Saves *this to archive ar using current version of MatchSettings.
	template<class Archive>
	void save(Archive &ar, const unsigned/* version*/) const {
		ar << kSsim
			<< kSdevFg << kSdevEdge << kSdevBg
			<< kContrast
			<< kMCsOffset << kCosAngleMCs
			<< kSymDensity
			<< threshold4Blank;
	}
	BOOST_SERIALIZATION_SPLIT_MEMBER();
	friend class boost::serialization::access;

	/// creates 'initMatchSettings.cfg' with data from 'res/defaultMatchSettings.txt'
	void createUserDefaults();

public:
	/**
	Initializes the fields from initMatchSettings.cfg when this file exists and isn't obsolete,
	otherwise from res/defaultMatchSettings.txt.
	The latter file is used to conveniently alter the defaults during development.

	@param appLaunchPath the path to 'Pic2Sym.exe' and is used to determine
	the folder where to look for 'res/defaultMatchSettings.txt' and 'initMatchSettings.cfg'
	*/
	MatchSettings(const std::string &appLaunchPath);

	const boost::filesystem::path& getWorkDir() const { return workDir; }

	const double& get_kSsim() const { return kSsim; }
	void set_kSsim(double kSsim_);

	const double& get_kSdevFg() const { return kSdevFg; }
	void set_kSdevFg(double kSdevFg_);

	const double& get_kSdevEdge() const { return kSdevEdge; }
	void set_kSdevEdge(double kSdevEdge_);

	const double& get_kSdevBg() const { return kSdevBg; }
	void set_kSdevBg(double kSdevBg_);

	const double& get_kContrast() const { return kContrast; }
	void set_kContrast(double kContrast_);

	const double& get_kCosAngleMCs() const { return kCosAngleMCs; }
	void set_kCosAngleMCs(double kCosAngleMCs_);

	const double& get_kMCsOffset() const { return kMCsOffset; }
	void set_kMCsOffset(double kMCsOffset_);

	const double& get_kSymDensity() const { return kSymDensity; }
	void set_kSymDensity(double kSymDensity_);

	unsigned getBlankThreshold() const { return threshold4Blank; }
	void setBlankThreshold(unsigned threshold4Blank_);

	bool parseCfg(const boost::filesystem::path &cfgFile); ///< Loads the settings provided in cfgFile
	
	/// Overwrites current settings with those from initMatchSettings.cfg.
	void loadUserDefaults();
	void saveUserDefaults() const; ///< Overwrites initMatchSettings.cfg with current settings

	friend std::ostream& operator<<(std::ostream &os, const MatchSettings &c);
#ifdef UNIT_TESTING
	/// Constructor available only within UnitTesting project
	MatchSettings(double kSsim_ = 0.,
		   double kSdevFg_ = 0., double kSdevEdge_ = 0., double kSdevBg_ = 0.,
		   double kContrast_ = 0., double kMCsOffset_ = 0., double kCosAngleMCs_ = 0.,
		   double kSymDensity_ = 0., unsigned threshold4Blank_ = 0U);
#endif
};

BOOST_CLASS_VERSION(MatchSettings, MatchSettings::VERSION);

#endif
