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

#ifndef H_SYM_SETTINGS
#define H_SYM_SETTINGS

#include "symSettingsBase.h"

#pragma warning ( push, 0 )

#include "std_string.h"

#ifndef AI_REVIEWER_CHECK
#	include <boost/archive/binary_oarchive.hpp>
#	include <boost/archive/binary_iarchive.hpp>
#	include <boost/serialization/split_member.hpp>
#	include <boost/serialization/version.hpp>
#endif // AI_REVIEWER_CHECK not defined

#pragma warning ( pop )

/// Parameters concerning the symbols set used for approximating patches.
class SymSettings : public ISymSettings {
protected:
	std::stringType fontFile;	///< the file containing the used font family with the desired style
	std::stringType encoding;	///< the particular encoding of the used cmap
	unsigned fontSz;		///< size of the symbols

public:
	/**
	Loads a SymSettings object from ar overwriting *this and reporting the changes.

	@param ar source of the SymSettings to load
	@param version the version of the loaded object
	*/
	template<class Archive>
	void load(Archive &ar, const unsigned version) {
		UNREFERENCED_PARAMETER(version);

		// It is useful to see which settings changed when loading
		SymSettings defSettings(*this); // create as copy of previous values

		// read user default match settings
		string newFontFile, newEncoding;
		ar >> newFontFile >> newEncoding >> defSettings.fontSz;

		defSettings.fontFile = newFontFile;
		defSettings.encoding = newEncoding;

		// these show message when there are changes
		setFontFile(defSettings.fontFile);
		setEncoding(defSettings.encoding);
		setFontSz(defSettings.fontSz);
	}

	/// Saves *this to ar
	template<class Archive>
	void save(Archive &ar, const unsigned) const {
		ar << (string)fontFile << (string)encoding << fontSz;
	}

#ifndef AI_REVIEWER_CHECK
	BOOST_SERIALIZATION_SPLIT_MEMBER();
#endif // AI_REVIEWER_CHECK not defined

	/// Constructor takes an initial fontSz, just to present a valid slider value in Control Panel
	SymSettings(unsigned fontSz_) : fontSz(fontSz_) {}

	/// Reset font settings apart from the font size
	/// which should remain on its value from the Control Panel
	void reset() override;

	/// Report if these settings are initialized or not
	bool initialized() const override;

	const std::stringType& getFontFile() const override final { return fontFile; }
	void setFontFile(const std::stringType &fontFile_) override;

	const std::stringType& getEncoding() const override final { return encoding; }
	void setEncoding(const std::stringType &encoding_) override;

	const unsigned& getFontSz() const override final { return fontSz; }
	void setFontSz(unsigned fontSz_) override;

	/// @return a copy of these settings
	std::uniquePtr<ISymSettings> clone() const override;

	bool operator==(const SymSettings &other) const;
	bool operator!=(const SymSettings &other) const;
};

#ifndef AI_REVIEWER_CHECK
BOOST_CLASS_VERSION(SymSettings, 0)
#endif // AI_REVIEWER_CHECK not defined

#endif // H_SYM_SETTINGS
