/**********************************************************
 Project:     Pic2Sym
 File:        main.cpp

 Author:      Florin Tulba
 Created on:  2016-1-8
 
 Copyright (c) 2016 Florin Tulba
 **********************************************************/

#include "controller.h"

using namespace std;

void main(int, char* argv[]) {
	Config cfg(argv[0]);
	Controller c(cfg);
	c.handleRequests();
}
