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

#ifndef H_MATCH_SETTINGS
#define H_MATCH_SETTINGS

#include "matchSettingsBase.h"
#include "misc.h"

#pragma warning ( push, 0 )

#ifndef UNIT_TESTING
#	include "boost_filesystem_path.h"
#endif // UNIT_TESTING not defined

#ifndef AI_REVIEWER_CHECK
#	include <boost/archive/binary_oarchive.hpp>
#	include <boost/archive/binary_iarchive.hpp>
#	include <boost/serialization/split_member.hpp>
#	include <boost/serialization/version.hpp>
#endif // AI_REVIEWER_CHECK not defined

#pragma warning ( pop )

/// MatchSettings class controls the matching parameters for transforming one or more images.
class MatchSettings : public IMatchSettings {
protected:
#ifndef UNIT_TESTING
	static boost::filesystem::path defCfgPath;	///< Path of the original configuration file
	static boost::filesystem::path cfgPath;	///< Path of the user configuration file

	static void configurePaths();	///< initializes defCfgPath & cfgPath

	bool parseCfg();	///< Loads the settings provided in cfgFile

	/// creates 'initMatchSettings.cfg' with data from 'res/defaultMatchSettings.txt'
	void createUserDefaults();

#endif // UNIT_TESTING not defined

	double kSsim = 0.;				///< power of factor controlling structural similarity
	double kSdevFg = 0.;			///< power of factor for foreground glyph-patch correlation
	double kSdevEdge = 0.;			///< power of factor for contour glyph-patch correlation
	double kSdevBg = 0.;			///< power of factor for background glyph-patch correlation
	double kContrast = 0.;			///< power of factor for the resulted glyph contrast
	double kMCsOffset = 0.;			///< power of factor targeting smoothness (mass-center offset)
	double kCosAngleMCs = 0.;		///< power of factor targeting smoothness (mass-centers angle)
	double kSymDensity = 0.;		///< power of factor aiming fanciness, not correctness
	unsigned threshold4Blank = 0U;	///< Using Blank character replacement under this threshold
	bool hybridResultMode = false;	///< 'normal' means actual result; 'hybrid' cosmeticizes the result

#ifndef UNIT_TESTING
	bool initialized = false;		///< true after FIRST completed initialization
#endif // UNIT_TESTING not defined

public:
	// BUILD CLEAN WHEN THIS CHANGES!
	static const unsigned VERSION = 2U; ///< version of MatchSettings class

	/**
	Loading a MatchSettings object of a given version.
	It overwrites *this, reporting any changes

	@param ar the source of the object
	@param version what version is the loaded object

	@throw invalid_argument for an obsolete 'initMatchSettings.cfg'
	*/
	template<class Archive>
	void load(Archive &ar, const unsigned version) {
#ifndef UNIT_TESTING
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
			if(!initialized) // can happen only when loading an obsolete 'initMatchSettings.cfg'
				THROW_WITH_CONST_MSG("Obsolete version of 'initMatchSettings.cfg'!", invalid_argument);

			// Point reachable while reading Settings with an older version of MatchSettings field
		}
#endif // UNIT_TESTING not defined

		// It is useful to see which settings changed when loading =>
		// Loading data in a temporary object and comparing with existing values.
		MatchSettings defSettings(*this); // create as copy of previous values

		// read user default match settings
		if(version >= 2U) { // versions >= 2 use hybridResultMode
			ar >> defSettings.hybridResultMode;
		} else {
			defSettings.hybridResultMode = false;
		}
		if(version >= 1U) { // versions >= 1 use kSsim
			ar >> defSettings.kSsim;
		} else {
			defSettings.kSsim = 0.;
		}
		ar >> defSettings.kSdevFg >> defSettings.kSdevEdge >> defSettings.kSdevBg
			>> defSettings.kContrast
			>> defSettings.kMCsOffset >> defSettings.kCosAngleMCs
			>> defSettings.kSymDensity
			>> defSettings.threshold4Blank;

		// these show message when there are changes
		setResultMode(defSettings.hybridResultMode);
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
		ar << hybridResultMode
			<< kSsim
			<< kSdevFg << kSdevEdge << kSdevBg
			<< kContrast
			<< kMCsOffset << kCosAngleMCs
			<< kSymDensity
			<< threshold4Blank;
	}

	/**
	Initializes the object.

	When unit testing, it leaves it empty.
	Otherwise it will load its fields from disk.
	*/
	MatchSettings();

	const bool& isHybridResult() const override final { return hybridResultMode; }
	MatchSettings& setResultMode(bool hybridResultMode_) override;

	const double& get_kSsim() const override final { return kSsim; }
	MatchSettings& set_kSsim(double kSsim_) override;

	const double& get_kSdevFg() const override final { return kSdevFg; }
	MatchSettings& set_kSdevFg(double kSdevFg_) override;

	const double& get_kSdevEdge() const override final { return kSdevEdge; }
	MatchSettings& set_kSdevEdge(double kSdevEdge_) override;

	const double& get_kSdevBg() const override final { return kSdevBg; }
	MatchSettings& set_kSdevBg(double kSdevBg_) override;

	const double& get_kContrast() const override final { return kContrast; }
	MatchSettings& set_kContrast(double kContrast_) override;

	const double& get_kCosAngleMCs() const override final { return kCosAngleMCs; }
	MatchSettings& set_kCosAngleMCs(double kCosAngleMCs_) override;

	const double& get_kMCsOffset() const override final { return kMCsOffset; }
	MatchSettings& set_kMCsOffset(double kMCsOffset_) override;

	const double& get_kSymDensity() const override final { return kSymDensity; }
	MatchSettings& set_kSymDensity(double kSymDensity_) override;

	unsigned getBlankThreshold() const override final { return threshold4Blank; }
	MatchSettings& setBlankThreshold(unsigned threshold4Blank_) override;

#ifndef UNIT_TESTING
	/// loads user defaults or throws for obsolete / invalid file
	void replaceByUserDefaults() override;
	void saveAsUserDefaults() const override;	///< save these as user defaults
#endif // UNIT_TESTING not defined

	/// Provides a representation of these settings in a verbose manner or not
	const std::stringType toString(bool verbose) const override;

	/// @return a clone of current settings
	std::uniquePtr<IMatchSettings> clone() const override;

#ifndef AI_REVIEWER_CHECK
	BOOST_SERIALIZATION_SPLIT_MEMBER();
#endif // AI_REVIEWER_CHECK not defined
};

#ifndef AI_REVIEWER_CHECK
BOOST_CLASS_VERSION(MatchSettings, MatchSettings::VERSION);
#endif // AI_REVIEWER_CHECK not defined

#endif // H_MATCH_SETTINGS
