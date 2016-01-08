/**********************************************************
 Project:     Pic2Sym
 File:        misc.cpp

 Author:      Florin Tulba
 Created on:  2015-12-22
 
 Copyright (c) 2016 Florin Tulba
 **********************************************************/

#include "misc.h"

using namespace std;

bool boolPrompt(const string &msg) {
	string ans;
	cout<<msg<<" (<Enter>='Yes'; anything else='No') ";
	getline(cin, ans);
	return ans.empty();
}
