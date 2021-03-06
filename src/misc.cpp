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

#include "misc.h"

#pragma warning ( push, 0 )

#define WIN32_LEAN_AND_MEAN
#include "Windows.h"

#pragma warning ( pop )

using namespace std;

namespace {

	/// Multiton supporting info/warn/errorMsg functions
	class MsgCateg final {
		const stringType categName;
		const UINT categVal;

		MsgCateg(const stringType &categName_, UINT categVal_) :
			categName(categName_), categVal(categVal_) {}
		MsgCateg(const MsgCateg&) = delete;
		void operator=(const MsgCateg&) = delete;

	public:
		static const MsgCateg INFO_CATEG, WARN_CATEG, ERR_CATEG;

		const stringType& name() const { return categName; }
		const UINT val() const { return categVal; }
	};

	const MsgCateg MsgCateg::INFO_CATEG("Information", MB_ICONINFORMATION);
	const MsgCateg MsgCateg::WARN_CATEG("Warning", MB_ICONWARNING);
	const MsgCateg MsgCateg::ERR_CATEG("Error", MB_ICONERROR);

#ifndef UNIT_TESTING
	/// When interacting with the user, the messages are nicer as popup windows
	void msg(const MsgCateg &msgCateg, const stringType &title_, const stringType &text) {
		stringType title = title_;
		if(title.empty())
			title = msgCateg.name();

		MessageBox(nullptr, str2wstr(text).c_str(),
				   str2wstr(title).c_str(),
				   MB_OK | MB_TASKMODAL | MB_SETFOREGROUND | msgCateg.val());
	}

#else // UNIT_TESTING defined
	/// When performing Unit Testing, the messages will appear on the console
	void msg(const MsgCateg &msgCateg, const stringType &title, const stringType &text) {
		cout.flush(); cerr.flush();
		ostream &os = (&msgCateg == &MsgCateg::ERR_CATEG) ? cerr : cout;
		os<<msgCateg.name();
		if(title.empty())
			os<<" ->"<<endl;
		else
			os<<" -> <<"<<title<<">>"<<endl;
		os<<text<<endl;
		os.flush();
	}

#endif // UNIT_TESTING

} // anonymous namespace


void infoMsg(const stringType &text, const stringType &title/* = ""*/) {
	msg(MsgCateg::INFO_CATEG, title, text);
}

void warnMsg(const stringType &text, const stringType &title/* = ""*/) {
	msg(MsgCateg::WARN_CATEG, title, text);
}

void errMsg(const stringType &text, const stringType &title/* = ""*/) {
	msg(MsgCateg::ERR_CATEG, title, text);
}

wstringType str2wstr(const stringType &str) {
	return wstringType(CBOUNDS(str)); // RVO
}

stringType wstr2str(const wstringType &wstr) {
	return stringType(CBOUNDS(wstr)); // RVO
}
