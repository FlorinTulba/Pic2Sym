/**********************************************************
 Project:     Pic2Sym
 File:        main.cpp

 Author:      Florin Tulba
 Created on:  2015-12-20
 
 Copyright (c) 2016 Florin Tulba
 **********************************************************/

#include "misc.h"
#include "config.h"
#include "transform.h"

using namespace std;

void main(int, char* argv[]) {
	Config cfg(argv[0]);
	Transformer t(cfg);

	do {
		t.reconfig();
		t.run();
	} while(boolPrompt("Do you want to do more transformations?"));

	cout<<"Leaving ..."<<endl;
}