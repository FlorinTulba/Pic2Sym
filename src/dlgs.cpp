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

#ifndef UNIT_TESTING

#include "dlgs.h"
#include "misc.h"

#pragma warning ( push, 0 )

#include <cassert>
#include <unordered_map>
#include <regex>

#include "boost_filesystem_operations.h"

#pragma warning ( pop )

using namespace std;
using namespace boost::filesystem;

namespace {
	/**
	FontFinder encapsulates the logic to obtain the file path for a given font name.
	It has a single public and static method: pathFor.
	*/
	class FontFinder {
		/**
		RegistryHelper isolates Registry API from the business logic within FontFinder.
		It provides an iterator-like method: extractNextFont.
		*/
		class RegistryHelper {
			HKEY fontsKey = nullptr;
			vector<TCHAR> fontNameBuf;
			vector<BYTE> fontFileBuf;
			DWORD longestNameLen = 0, longestDataLen = 0;
			DWORD idx = 0;

		public:
			RegistryHelper() {
				// The mapping between the font name and the corresponding font file can be found
				// in the registry in:
				// Computer->HKEY_LOCAL_MACHINE->Software->Microsoft->Windows NT->CurrentVersion->Fonts
				static const LPTSTR fontRegistryPath = _T("Software\\Microsoft\\Windows NT\\CurrentVersion\\Fonts");

				if(RegOpenKeyEx(HKEY_LOCAL_MACHINE,	// predefined key
								fontRegistryPath,	// subkey
								0U,					// ulOptions - not an symbolic link
								KEY_READ,			// rights to query, enumerate
								&fontsKey			// returns the necessary key
								) != ERROR_SUCCESS)
					THROW_WITH_CONST_MSG("Couldn't find the Fonts mapping within Registry!", invalid_argument);

				// Get the required buffer size for font names and the names of the corresponding font files		
				if(RegQueryInfoKey(fontsKey,
						nullptr, // lpClass
						nullptr, // lpcClass
						nullptr, // lpReserved
						nullptr, // lpcSubKeys (There are no subkeys)
						nullptr, // lpcMaxSubKeyLen
						nullptr, // lpcMaxClassLen
						nullptr, // lpcValues (We just enumerate the values, no need to know their count)
						&longestNameLen, // returns required buffer size for font names
						&longestDataLen, // returns required buffer size for corresponding names of the font files
						nullptr, // lpcbSecurityDescriptor (Not necessary)
						nullptr // lpftLastWriteTime (Not interested in this)
						) != ERROR_SUCCESS)
					THROW_WITH_CONST_MSG("Couldn't interrogate the Fonts key!", invalid_argument);

				fontNameBuf.resize(longestNameLen+1); // reserve also for '\0'
				fontFileBuf.resize(longestDataLen+2); // reserve also for '\0'(wchar_t as BYTE)
			}
			~RegistryHelper() {
				RegCloseKey(fontsKey);
			}

			/*
			extractNextFont returns true if there was another font to be handled.
			In that case, it returns the font name and font file name within the parameters.
			*/
			bool extractNextFont(wstringType &fontName, wstringType &fontFileName) {
				DWORD lenFontName = longestNameLen+1, lenFontFileName = longestDataLen+2;
				const LONG ret = RegEnumValue(fontsKey,
										idx++, // which font index
										fontNameBuf.data(), // storage for the font names
										&lenFontName, // length of the returned font name
										nullptr, // lpReserved
										nullptr, // lpType (All are REG_SZ)
										fontFileBuf.data(), // storage for the font file names
										&lenFontFileName); // length of the returned font file name
				if(ERROR_NO_MORE_ITEMS == ret)
					return false;

				if(ERROR_MORE_DATA == ret)
					THROW_WITH_CONST_MSG(__FUNCTION__ " : Allocated buffer isn't large enough to fit the font name or font file name!", length_error);

				if(ERROR_SUCCESS != ret)
					THROW_WITH_CONST_MSG(__FUNCTION__ " : Couldn't enumerate the Fonts!", length_error);

				fontName.assign(fontNameBuf.data());
				fontFileName.assign((TCHAR*)fontFileBuf.data());

				return true;
			}
		};

		/**
		The font read from registry needs to contain the required name and also match bold&italic style.
		Otherwise it will return false.
		*/
		static bool relevantFontName(const wstringType &wCurFontName,
									 const wstringType &wFontName, bool isBold, bool isItalic) {
			const auto at = wCurFontName.find(wFontName); // fontName won't be necessary a prefix!!
			if(at == wstringType::npos)
				return false; // current font doesn't contain the desired prefix

			// Bold and Italic fonts typically append such terms to their key name.
	#pragma warning ( disable : WARN_THREAD_UNSAFE )
			static const wregex rexBold(L"Bold|Heavy|Black", regex_constants::icase);
			static const wregex rexItalic(L"Italic|Oblique", regex_constants::icase);

			static match_results<wstring::const_iterator> match;
	#pragma warning ( default : WARN_THREAD_UNSAFE )

			const wstringType wSuffixCurFontName =
				wCurFontName.substr(at+wFontName.length()); // extract the suffix

			if(isBold != regex_search((wstring)wSuffixCurFontName, match, rexBold))
				return false; // current font has different Bold status than expected

			if(isItalic != regex_search((wstring)wSuffixCurFontName, match, rexItalic))
				return false; // current font has different Italic status than expected

			return true;
		}

		/// Ensures the obtained font file name represents a valid path
		static stringType refineFontFileName(const wstringType &wCurFontFileName) {
			path curFontFile(stringType(BOUNDS(wCurFontFileName)));
			if(!curFontFile.has_parent_path()) {
				// The fonts are typically installed within c:\Windows\Fonts
	#pragma warning ( disable : WARN_DEPRECATED WARN_THREAD_UNSAFE )
				static const path typicalFontsDir =
					path(stringType(getenv("SystemRoot"))).append("Fonts");
	#pragma warning ( default : WARN_DEPRECATED WARN_THREAD_UNSAFE )

				path temp(typicalFontsDir);
				temp /= curFontFile;
				// If the curFontFile isn't a path already, prefix it with typicalFontsDir
				curFontFile = move(temp);
			}
			if(!exists(curFontFile))
				THROW_WITH_VAR_MSG(__FUNCTION__ " : There's no such font file: " + curFontFile.string(), 
									FontLocationFailure);

			return curFontFile.string();
		}

		/// When ambiguous results, lets the user select the correct one.
		static stringType extractResult(unordered_map<stringType, stringType> &choices) {
			assert(!choices.empty());

			const size_t choicesCount = choices.size();
			if(1ULL == choicesCount)
				return cbegin(choices)->second;

			// More than 1 file suits the selected font and the user should choose the appropriate one
			cout<<endl<<"More fonts within Windows Registry suit the selected Font type. Please select the appropriate one:"<<endl;
			size_t idx = 0ULL;
			for(const auto &choice : choices)
				cout<<idx++<<" : "<<choice.first<<" -> "<<choice.second<<endl;

			//idx is here choicesCount
			while(idx>=choicesCount) {
				cout<<"Enter correct index: ";
				cin>>idx;
			}

			return next(cbegin(choices), (ptrdiff_t)idx)->second;
		}

	public:
		/**
		pathFor static method finds the path for a provided fontName.
		Unfortunately, the provided fontName isn't decorated with Bold and/or Italic at all,
		so isBold and isItalic parameters were necessary, too.
		*/
		static stringType pathFor(const stringType &fontName, bool isBold, bool isItalic) {
			wstringType wCurFontName, wCurFontFileName;
			unordered_map<stringType, stringType> choices;
			wstringType wFontName(CBOUNDS(fontName));

			RegistryHelper rh;
			while(rh.extractNextFont(wCurFontName, wCurFontFileName))
				if(relevantFontName(wCurFontName, wFontName, isBold, isItalic))
					choices[wstr2str(wCurFontName)] =
								refineFontFileName(wCurFontFileName);

			if(choices.empty())
				THROW_WITH_CONST_MSG(__FUNCTION__ " : Couldn't find this font within registry!\n"
									"It might be there under a different name or "
									"it might appear only among the Windows Fonts as a shortcut to the actual file.\n"
									"The Font Dialog presents all corresponding Windows Fonts, "
									"unfortunately providing unreliable font name hints",
									FontLocationFailure);

			return extractResult(choices);
		}
	};
} // anonymous namespace

OpenSave::OpenSave(const TCHAR * const title, const TCHAR * const filter,
				   const TCHAR * const defExtension/* = nullptr*/,
				   bool toOpen_/* = true*/) : toOpen(toOpen_) {
	ZeroMemory(&ofn, sizeof(ofn));
	ofn.lStructSize = sizeof(ofn);
	ofn.hwndOwner = nullptr; // no owner
	fNameBuf[0ULL] = '\0';
	ofn.lpstrFile = fNameBuf;
	ofn.nMaxFile = sizeof(fNameBuf);
	ofn.lpstrFilter = filter;
	ofn.nFilterIndex = 1;
	ofn.lpstrFileTitle = nullptr;
	ofn.nMaxFileTitle = 0;
	ofn.lpstrInitialDir = nullptr;
	ofn.lpstrTitle = title;
	if(defExtension != nullptr)
		ofn.lpstrDefExt = defExtension;
	if(toOpen)
		ofn.Flags = OFN_PATHMUSTEXIST | OFN_FILEMUSTEXIST | OFN_NOCHANGEDIR;
	else
		ofn.Flags = OFN_OVERWRITEPROMPT | OFN_NOCHANGEDIR;
}

bool OpenSave::promptForUserChoice() {
	if(toOpen) {
		if(!GetOpenFileName(&ofn)) {
			reset();
			return false;
		}
	} else {
		if(!GetSaveFileName(&ofn)) {
			reset();
			return false;
		}
	}
	const wstringType wResult(ofn.lpstrFile);
	result.assign(CBOUNDS(wResult));
	return true;
}

ImgSelector::ImgSelector() :
	OpenSave(
		_T("Please select an image to process"),
		_T("Allowed Image Files\0*.bmp;*.dib;*.png;*.tif;*.tiff;*.jpg;*.jpe;*.jp2;*.jpeg;*.webp;*.pbm;*.pgm;*.ppm;*.sr;*.ras\0\0")) {}

SettingsSelector::SettingsSelector(bool toOpen_/* = true*/) :
	OpenSave(
		toOpen_ ?
			_T("Please select a settings file to load") :
			_T("Please specify where to save current settings"),
		_T("Allowed Settings Files\0*.p2s\0\0"),
		_T("p2s"),
		toOpen_) {}

SelectFont::SelectFont() {
	ZeroMemory(&cf, sizeof(cf));
	cf.lStructSize = sizeof(cf);
	ZeroMemory(&lf, sizeof(lf));
	cf.lpLogFont = &lf;
	cf.Flags = CF_FORCEFONTEXIST | CF_NOVERTFONTS | CF_FIXEDPITCHONLY | CF_SCALABLEONLY | CF_NOSIMULATIONS | CF_NOSCRIPTSEL;
}

bool SelectFont::promptForUserChoice() {
	if(!ChooseFont(&cf)) {
		reset();
		return false;
	}
		
	isBold = (cf.nFontType & 0x100) || (lf.lfWeight > FW_MEDIUM); // There are fonts with only a Medium style (no Regular one)
	isItalic = (cf.nFontType & 0x200) || (lf.lfItalic != (BYTE)0);
	const wstringType wResult(lf.lfFaceName);
	result.assign(CBOUNDS(wResult));

	cout<<"Selected ";
	if(isBold)
		cout<<"bold ";
	if(isItalic)
		cout<<"italic ";
	cout<<'\''<<result<<"'";
	
	result = FontFinder::pathFor(result, isBold, isItalic);

	cout<<" ["<<result<<']'<<endl;

	return true;
}

#endif // UNIT_TESTING not defined
