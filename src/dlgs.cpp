/**********************************************************
 Project:     Pic2Sym
 File:        dlgs.cpp

 Author:      Florin Tulba
 Created on:  2015-12-21
 
 Copyright (c) 2016 Florin Tulba
 **********************************************************/

#include "dlgs.h"
#include "misc.h"

#include <fstream>
#include <map>
#include <regex>

using namespace std;

FileOpen::FileOpen() {
	ZeroMemory(&ofn, sizeof(ofn));
	ofn.lStructSize = sizeof(ofn);
	ofn.hwndOwner = nullptr; // no owner
	ofn.lpstrFile = fNameBuf;
	// Set lpstrFile[0] to '\0' so that GetOpenFileName does not 
	// use the contents of fNameBuf to initialize itself.
	ofn.lpstrFile[0] = '\0';
	ofn.nMaxFile = sizeof(fNameBuf);
	ofn.lpstrFilter = _T("Allowed Image Files\0*.bmp;*.dib;*.png;*.tif;*.tiff;*.jpg;*.jpe;*.jp2;*.jpeg;*.webp;*.pbm;*.pgm;*.ppm;*.sr;*.ras\0\0");
	ofn.nFilterIndex = 1;
	ofn.lpstrFileTitle = nullptr;
	ofn.nMaxFileTitle = 0;
	ofn.lpstrInitialDir = nullptr;
	ofn.Flags = OFN_PATHMUSTEXIST | OFN_FILEMUSTEXIST;
	ofn.lpstrTitle = _T("Please select an image to process");
}

bool FileOpen::canceled() {
	if(!GetOpenFileName(&ofn)) {
		result = "";
		return true;
	}
	wstring wResult(ofn.lpstrFile);
	result.assign(BOUNDS(wResult));
	return false;
}

/*
FontFinder encapsulates the logic to obtain the file path for a given font name.
It has a single public and static method: pathFor.
*/
class FontFinder {
	/*
	RegistryHelper isolates Registry API from the business logic within FontFinder.
	It provides an iterator-like method: extractNextFont.
	*/
	class RegistryHelper {
		HKEY fontsKey = nullptr;
		DWORD longestNameLen = 0, longestDataLen = 0;
		LPTSTR fontNameBuf = nullptr;
		LPBYTE fontFileBuf = nullptr;
		DWORD idx = 0;

	public:
		RegistryHelper() {
			// The mapping between the font name and the corresponding font file can be found
			// in the registry in:
			// Computer->HKEY_LOCAL_MACHINE->Software->Microsoft->Windows NT->CurrentVersion->Fonts
			static const LPTSTR fontRegistryPath = _T("Software\\Microsoft\\Windows NT\\CurrentVersion\\Fonts");

			if(RegOpenKeyEx(HKEY_LOCAL_MACHINE, // predefined key
				fontRegistryPath, // subkey
				0U, // ulOptions - not an symbolic link
				KEY_READ, // rights to query, enumerate
				&fontsKey // returns the necessary key
				) != ERROR_SUCCESS)
				throw invalid_argument("Couldn't find the Fonts mapping within Registry!");

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
				throw invalid_argument("Couldn't interrogate the Fonts key!");

			fontNameBuf = new TCHAR[longestNameLen+1]; // reserve also for \0
			fontFileBuf = new BYTE[longestDataLen+2]; // reserve also for \0(wchar)
		}
		~RegistryHelper() {
			delete[] fontFileBuf;
			delete[] fontNameBuf;
			RegCloseKey(fontsKey);
		}

		/*
		extractNextFont returns true if there was another font to be handled.
		In that case, it returns the font name and font file name within the parameters.
		*/
		bool extractNextFont(wstring &fontName, wstring &fontFileName) {
			DWORD lenFontName = longestNameLen+1, lenFontFileName = longestDataLen+2;
			LONG ret = RegEnumValue(fontsKey,
									idx++, // which font index
									fontNameBuf, // storage for the font names
									&lenFontName, // length of the returned font name
									nullptr, // lpReserved
									nullptr, // lpType (All are REG_SZ)
									fontFileBuf, // storage for the font file names
									&lenFontFileName); // length of the returned font file name
			if(ERROR_NO_MORE_ITEMS == ret)
				return false;

			if(ERROR_MORE_DATA == ret) {
				cerr<<"Allocated buffer isn't large enough to fit the font name or font file name!"<<endl;
				throw length_error("Allocated buffer isn't large enough!");
			}
			if(ERROR_SUCCESS != ret)
				throw invalid_argument("Couldn't enumerate the Fonts!");

			fontName.assign(fontNameBuf);
			fontFileName.assign((TCHAR*)fontFileBuf);

			return true;
		}
	};

	/*
	The font read from registry needs to contain the required name and also match bold&italic style.
	Otherwise it will return false.
	*/
	static bool relevantFontName(const wstring &wCurFontName,
								 const wstring &wFontName, bool isBold, bool isItalic) {
		wstring::size_type at = wCurFontName.find(wFontName); // fontName won't be necessary a prefix!!
		if(at == wstring::npos)
			return false; // current font doesn't contain the desired prefix

		// Bold and Italic fonts typically append such terms to their key name.
		wstring wSuffixCurFontName = wCurFontName.substr(at+wFontName.length()); // extract the suffix

		static match_results<wstring::const_iterator> match;
		static const wregex rexBold(L"Bold|Heavy|Black", regex_constants::icase);
		if(isBold != regex_search(wSuffixCurFontName, match, rexBold))
			return false; // current font has different Bold status than expected

		static const wregex rexItalic(L"Italic|Oblique", regex_constants::icase);
		if(isItalic != regex_search(wSuffixCurFontName, match, rexItalic))
			return false; // current font has different Italic status than expected

		return true;
	}

	// Ensures the obtained font file name represents a valid path
	static string refineFontFileName(const wstring &wCurFontFileName) {
		string curFontFileName(BOUNDS(wCurFontFileName));
		if(curFontFileName.find('\\') == string::npos) {
			// The fonts are typically installed within c:\Windows\Fonts
#pragma warning(disable:4996) // getenv is safe (unless SystemRoot is really long)
			static const string normalFolder = string(getenv("SystemRoot")) + "\\Fonts\\";
#pragma warning(default:4996)
			curFontFileName = normalFolder + curFontFileName; // If the curFontFileName isn't a path already, add normal folder
		}
		if(!ifstream(curFontFileName)) {
			cerr<<"There's no such font file: "<<curFontFileName<<endl;
			throw domain_error("Wrong assumption for locating the font files!");
		}
		return curFontFileName;
	}

	// When ambiguous results, lets the user select the correct one.
	static string extractResult(map<string, string> &choices) {
		size_t choicesCount = choices.size();
		if(0U == choicesCount) {
			cerr<<"Couldn't find this font within registry!"<<endl;
			throw runtime_error("Couldn't locate font within registry!");
		}

		if(1U == choicesCount)
			return choices.begin()->second;

		// More than 1 file suits the selected font and the user should choose the appropriate one
		cout<<endl<<"More fonts within Registry suit the selected Font type. Please select the appropriate one:"<<endl;
		size_t idx = 0U;
		for(auto choice : choices)
			cout<<idx++<<" : "<<choice.first<<" -> "<<choice.second<<endl;

		//idx is here choicesCount
		while(idx>=choicesCount) {
			cout<<"Enter correct index: ";
			cin>>idx;
		}

		return next(choices.begin(), idx)->second;
	}

public:
	/*
	pathFor static method finds the path for a provided fontName.
	Unfortunately, the provided fontName isn't decorated with Bold and/or Italic at all,
	so isBold and isItalic parameters were necessary, too.
	*/
	static string pathFor(const string &fontName, bool isBold, bool isItalic) {
		wstring wCurFontName, wCurFontFileName;
		map<string, string> choices;
		wstring wFontName(BOUNDS(fontName));

		RegistryHelper rh;
		while(rh.extractNextFont(wCurFontName, wCurFontFileName)) {
			if(!relevantFontName(wCurFontName, wFontName, isBold, isItalic))
				continue;

			choices[string(BOUNDS(wCurFontName))] = refineFontFileName(wCurFontFileName);
		}

		return extractResult(choices);
	}
};

SelectFont::SelectFont() {
	ZeroMemory(&cf, sizeof(cf));
	cf.lStructSize = sizeof(cf);
	ZeroMemory(&lf, sizeof(lf));
	cf.lpLogFont = &lf;
	cf.Flags = CF_FORCEFONTEXIST | CF_NOVERTFONTS | CF_FIXEDPITCHONLY | CF_SCALABLEONLY | CF_NOSIMULATIONS | CF_NOSCRIPTSEL;
}

bool SelectFont::canceled() {
	if(!ChooseFont(&cf)) {
		result = "";
		return true;
	}
		
	isBold = (cf.nFontType & 0x100) || (lf.lfWeight > FW_MEDIUM); // There are fonts with only a Medium style (no Regular one)
	isItalic = (cf.nFontType & 0x200) || (lf.lfItalic != (BYTE)0);
	wstring wResult(lf.lfFaceName);
	result.assign(BOUNDS(wResult));

	cout<<"Selected ";
	if(isBold)
		cout<<"bold ";
	if(isItalic)
		cout<<"italic ";
	cout<<'\''<<result<<"'";

	result = FontFinder::pathFor(result, isBold, isItalic);

	cout<<" ["<<result<<']'<<endl;

	/*
	cout<<boolalpha;
	cout<<result<<":"<<endl;
	PRINTLN(cf.iPointSize);
	PRINTLN(cf.lpLogFont->lfHeight);
	PRINTLN(cf.lpLogFont->lfWeight);
	PRINTLN((bool)cf.lpLogFont->lfItalic);
	PRINTLN_H(cf.nFontType);
	PRINTLN_H((unsigned)cf.lpLogFont->lfQuality);
	PRINTLN_H((unsigned)cf.lpLogFont->lfClipPrecision);
	PRINTLN_H((unsigned)cf.lpLogFont->lfOutPrecision);
	PRINTLN_H((unsigned)cf.lpLogFont->lfPitchAndFamily);
	*/
	return false;
}
