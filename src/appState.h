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

#ifndef UNIT_TESTING

#ifndef H_APP_STATE
#define H_APP_STATE

typedef size_t AppStateType;

/**
Application distinct states

The actual state of the application might be a combination of them.
*/
enum struct AppState : AppStateType {
	Idle				= 0ULL,				///< Application is Idle
	UpdateImg			= 1ULL,				///< Updating the image to be transformed
	UpdateSymSettings	= UpdateImg<<1,		///< Updating settings related to the symbols to be used
	UpdateImgSettings	= UpdateImg<<2,		///< Updating settings related to the image to be transformed
	UpdateMatchSettings = UpdateImg<<3,		///< Updating settings related to the match aspects
	SaveMatchSettings	= UpdateImg<<4,		///< Saving only the settings about match aspects
	SaveAllSettings		= UpdateImg<<5,		///< Saving all 3 categories of settings
	LoadAllSettings		= UpdateImg<<6,		///< Loading all 3 categories of settings
	LoadMatchSettings	= UpdateImg<<7,		///< Loading only the settings about match aspects
	ImgTransform		= UpdateImg<<8		///< Performing an approximation of an image
};

/// More readable states
#define ST(State) \
	(AppStateType)AppState::State

/**
Every operation from the Control Panel must be authorized.
They must acquire such a permit when they start.
If they don't receive one, they must return.

In realization classes:
- the constructor will update application state to reflect the new [parallel] action.
- the destructor should be used to revert the application state change
*/
struct ActionPermit /*abstract*/ {
	virtual ~ActionPermit() = 0 {}
};

#endif // H_APP_STATE

#endif // UNIT_TESTING
