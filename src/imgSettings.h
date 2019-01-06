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

#ifndef H_IMG_SETTINGS
#define H_IMG_SETTINGS

#include "imgSettingsBase.h"
#include "misc.h"

#pragma warning ( push, 0 )

#ifndef AI_REVIEWER_CHECK
#	include <boost/archive/binary_oarchive.hpp>
#	include <boost/archive/binary_iarchive.hpp>
#	include <boost/serialization/split_member.hpp>
#	include <boost/serialization/version.hpp>
#endif // AI_REVIEWER_CHECK not defined

#pragma warning ( pop )

/**
Contains max count of horizontal & vertical patches to process.

The image is resized appropriately before processing.
*/
class ImgSettings : public IfImgSettings {
protected:
	unsigned hMaxSyms;	///< Count of resulted horizontal symbols
	unsigned vMaxSyms;	///< Count of resulted vertical symbols

public:

	/// Constructor takes initial values just to present valid sliders positions in Control Panel
	ImgSettings(unsigned hMaxSyms_, unsigned vMaxSyms_) :
		hMaxSyms(hMaxSyms_), vMaxSyms(vMaxSyms_) {}

	unsigned getMaxHSyms() const override final { return hMaxSyms; }
	void setMaxHSyms(unsigned syms) override;

	unsigned getMaxVSyms() const override final { return vMaxSyms; }
	void setMaxVSyms(unsigned syms) override;

	std::uniquePtr<IfImgSettings> clone() const override;

private:
	friend class boost::serialization::access;
	/**
	Overwrites *this with the ImgSettings object read from ar.

	@param ar source of the object to load
	@param version the version of the loaded ImgSettings
	*/
	template<class Archive>
	void load(Archive &ar, const unsigned version) {
		UNREFERENCED_PARAMETER(version);

		// It is useful to see which settings changed when loading
		ImgSettings defSettings(*this); // create as copy of previous values

		// read user default match settings
#ifndef AI_REVIEWER_CHECK
		ar >> defSettings.hMaxSyms >> defSettings.vMaxSyms;
#endif // AI_REVIEWER_CHECK not defined

		// these show message when there are changes
		setMaxHSyms(defSettings.hMaxSyms);
		setMaxVSyms(defSettings.vMaxSyms);
	}

	/// Saves *this to ar
	template<class Archive>
	void save(Archive &ar, const unsigned) const {
#ifndef AI_REVIEWER_CHECK
		ar << hMaxSyms << vMaxSyms;
#endif // AI_REVIEWER_CHECK not defined
	}
#ifndef AI_REVIEWER_CHECK
	BOOST_SERIALIZATION_SPLIT_MEMBER();
#endif // AI_REVIEWER_CHECK not defined
};

#ifndef AI_REVIEWER_CHECK
BOOST_CLASS_VERSION(ImgSettings, 0)
#endif // AI_REVIEWER_CHECK not defined

#endif // H_IMG_SETTINGS
