/**********************************************************
 Project:     Pic2Sym
 File:        misc.cpp

 Author:      Florin Tulba
 Created on:  2016-2-4
 
 Copyright (c) 2016 Florin Tulba
 **********************************************************/

#include "misc.h"

#include "Windows.h"

using namespace std;

namespace {

	class MsgCateg {
		const string categName;
		const long categVal;

		MsgCateg(const string &categName_, long categVal_) :
			categName(categName_), categVal(categVal_) {}
		MsgCateg(const MsgCateg&) = delete;
		void operator=(const MsgCateg&) = delete;

	public:
		static const MsgCateg INFO_CATEG, WARN_CATEG, ERR_CATEG;

		const string& name() const { return categName; }
		const long val() const { return categVal; }
	};

	const MsgCateg MsgCateg::INFO_CATEG("Information", MB_ICONINFORMATION);
	const MsgCateg MsgCateg::WARN_CATEG("Warning", MB_ICONWARNING);
	const MsgCateg MsgCateg::ERR_CATEG("Error", MB_ICONERROR);

#ifndef UNIT_TESTING

	void msg(const MsgCateg &msgCateg, const string &title_, const string &text) {
		string title = title_;
		if(title.empty())
			title = msgCateg.name();

		MessageBox(nullptr, wstring(CBOUNDS(text)).c_str(),
				   wstring(CBOUNDS(title)).c_str(),
				   MB_OK | MB_TASKMODAL | MB_SETFOREGROUND | msgCateg.val());
	}

#else // UNIT_TESTING is defined

	void msg(const MsgCateg &msgCateg, const string &title, const string &text) {
		cout.flush(); cerr.flush();
		auto &os = (&msgCateg == &MsgCateg::ERR_CATEG) ? cerr : cout;
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


void infoMsg(const string &text, const string &title/* = ""*/) {
	msg(MsgCateg::INFO_CATEG, title, text);
}

void warnMsg(const string &text, const string &title/* = ""*/) {
	msg(MsgCateg::WARN_CATEG, title, text);
}

void errMsg(const string &text, const string &title/* = ""*/) {
	msg(MsgCateg::ERR_CATEG, title, text);
}
