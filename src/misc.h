/**********************************************************
 Project:     Pic2Sym
 File:        misc.h

 Author:      Florin Tulba
 Created on:  2015-12-20
 
 Copyright (c) 2016 Florin Tulba
 **********************************************************/

#ifndef H_MISC
#define H_MISC

#include <iostream>
#include <iomanip>
#include <string>

// Display an expression and its value
#define PRINT(expr)			std::cout<<#expr " : "<<(expr)
#define PRINTLN(expr)		PRINT(expr)<<std::endl
#define PRINT_H(expr)		std::cout<<#expr " : 0x"<<std::hex<<(expr)<<std::dec
#define PRINTLN_H(expr)		PRINT_H(expr)<<std::endl

// Oftentimes functions operating on ranges need the full range.
// Example: copy(x.begin(), x.end(), ..) => copy(BOUNDS(x), ..)
#define BOUNDS(iterable)	std::begin(iterable), std::end(iterable)
#define CBOUNDS(iterable)	std::cbegin(iterable), std::cend(iterable)

// Notifying the user
void infoMsg(const std::string &text, const std::string &title = "");
void warnMsg(const std::string &text, const std::string &title = "");
void errMsg(const std::string &text, const std::string &title = "");

#endif