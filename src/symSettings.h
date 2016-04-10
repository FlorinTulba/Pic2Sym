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

#ifndef H_SYM_SETTINGS
#define H_SYM_SETTINGS

#include <string>

#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/serialization/split_member.hpp>
#include <boost/serialization/version.hpp>

/// Parameters concerning the symbols set used for approximating patches.
class SymSettings {
protected:
	std::string fontFile;	///< the file containing the used font family with the desired style
	std::string encoding;	///< the particular encoding of the used cmap
	unsigned fontSz;		///< size of the symbols

	/**
	Loads a SymSettings object from ar overwriting *this and reporting the changes.

	@param ar source of the SymSettings to load
	@param version the version of the loaded object
	*/
	template<class Archive>
	void load(Archive &ar, const unsigned version) {
		// It is useful to see which settings changed when loading
		SymSettings defSettings(*this); // create as copy of previous values

		// read user default match settings
		ar >> defSettings.fontFile >> defSettings.encoding >> defSettings.fontSz;

		// these show message when there are changes
		setFontFile(defSettings.fontFile);
		setEncoding(defSettings.encoding);
		setFontSz(defSettings.fontSz);
	}

	/// Saves *this to ar
	template<class Archive>
	void save(Archive &ar, const unsigned) const {
		ar << fontFile << encoding << fontSz;
	}
	BOOST_SERIALIZATION_SPLIT_MEMBER();
	friend class boost::serialization::access;

public:
	/// Constructor takes an initial fontSz, just to present a valid slider value in Control Panel
	SymSettings(unsigned fontSz_) : fontSz(fontSz_) {}

	bool ready() const { return !fontFile.empty(); }

	const std::string& getFontFile() const { return fontFile; }
	void setFontFile(const std::string &fontFile_);

	const std::string& getEncoding() const { return encoding; }
	void setEncoding(const std::string &encoding_);

	unsigned getFontSz() const { return fontSz; }
	void setFontSz(unsigned fontSz_);

	bool operator==(const SymSettings &other) const;
	bool operator!=(const SymSettings &other) const;
	friend std::ostream& operator<<(std::ostream &os, const SymSettings &ss);
};

BOOST_CLASS_VERSION(SymSettings, 0)

#endif