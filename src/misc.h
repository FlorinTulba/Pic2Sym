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

// There are situations when a macro will receive parameters containing Comma (,)
// Normally such parameters are split and the parameters count increases.
// To prevent the undesired split, wrap the parameters containing Comma within SINGLE_ARG.
// Example:
//	PRINTLN(1,2,3)  - complains it got 3 parameters while expecting just one
//	PRINTLN(SINGLE_ARG(1,2,3)) - works (displays 3)
#define SINGLE_ARG(...) __VA_ARGS__

// Displays a message and waits for a Yes/No answer from user
bool boolPrompt(const std::string &msg);

#endif