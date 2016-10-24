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

#ifndef H_MATCH_SETTINGS
#define H_MATCH_SETTINGS

#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/serialization/split_member.hpp>
#include <boost/serialization/version.hpp>

#ifndef UNIT_TESTING
class MatchSettingsManip; // forward declaration
#endif

/// MatchSettings class controls the matching parameters for transforming one or more images.
class MatchSettings {
public:
	// BUILD CLEAN WHEN THIS CHANGES!
	static const unsigned VERSION = 2U; ///< version of MatchSettings class

protected:
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
	friend class MatchSettingsManip; // to access initialized
#endif

	/**
	Loading a MatchSettings object of a given version.
	It overwrites *this, reporting any changes

	@param ar the source of the object
	@param version what version is the loaded object

	@throw invalid_argument when MatchSettingsManip::instance().load() throws
	*/
	template<class Archive>
	void load(Archive &ar, const unsigned version) {
#ifndef UNIT_TESTING
		MatchSettingsManip::instance().load(*this, ar, version);
#endif

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
	BOOST_SERIALIZATION_SPLIT_MEMBER();
	friend class boost::serialization::access;

public:
	/**
	Initializes the object.

	When unit testing, it leaves it empty.
	Otherwise it will ask MatchSettingsManip to load its fields from disk.
	*/
	MatchSettings();

	const bool& isHybridResult() const { return hybridResultMode; }
	MatchSettings& setResultMode(bool hybridResultMode_);

	const double& get_kSsim() const { return kSsim; }
	MatchSettings& set_kSsim(double kSsim_);

	const double& get_kSdevFg() const { return kSdevFg; }
	MatchSettings& set_kSdevFg(double kSdevFg_);

	const double& get_kSdevEdge() const { return kSdevEdge; }
	MatchSettings& set_kSdevEdge(double kSdevEdge_);

	const double& get_kSdevBg() const { return kSdevBg; }
	MatchSettings& set_kSdevBg(double kSdevBg_);

	const double& get_kContrast() const { return kContrast; }
	MatchSettings& set_kContrast(double kContrast_);

	const double& get_kCosAngleMCs() const { return kCosAngleMCs; }
	MatchSettings& set_kCosAngleMCs(double kCosAngleMCs_);

	const double& get_kMCsOffset() const { return kMCsOffset; }
	MatchSettings& set_kMCsOffset(double kMCsOffset_);

	const double& get_kSymDensity() const { return kSymDensity; }
	MatchSettings& set_kSymDensity(double kSymDensity_);

	unsigned getBlankThreshold() const { return threshold4Blank; }
	MatchSettings& setBlankThreshold(unsigned threshold4Blank_);

	friend std::ostream& operator<<(std::ostream &os, const MatchSettings &c);
};

BOOST_CLASS_VERSION(MatchSettings, MatchSettings::VERSION);

#endif
