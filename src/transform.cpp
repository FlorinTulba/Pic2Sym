/**********************************************************
 Project:     Pic2Sym
 File:        transform.cpp

 Author:      Florin Tulba
 Created on:  2016-1-6
 
 Copyright (c) 2016 Florin Tulba
 **********************************************************/

#include "controller.h"

#include "misc.h"

#include <sstream>
#include <numeric>

#if defined _DEBUG && !defined UNIT_TESTING
#	include <fstream>
#endif

#include <boost/filesystem/operations.hpp>

using namespace std;
using namespace boost::filesystem;

Transformer::Transformer(const Controller &ctrler_, const Settings &cfg_, MatchEngine &me_, Img &img_) :
		ctrler(ctrler_), cfg(cfg_), me(me_), img(img_) {
	createOutputFolder();
}

void Transformer::run() {
	me.updateSymbols(); // throws for invalid cmap/size

	// throws when no image
	const cv::Mat resized = img.resized(cfg.imgSettings(), cfg.symSettings().getFontSz());
	
	// keep it after img.resized, to display updated resized version as comparing image
	Controller::Timer timer(ctrler, Controller::Timer::ComputationType::IMG_TRANSFORM);

	const auto &ss = cfg.matchSettings();
	ostringstream oss;
	oss<<img.name()<<'_'
		<<me.getIdForSymsToUse()<<'_'
		<<ss.get_kSsim()<<'_'
		<<ss.get_kSdevFg()<<'_'<<ss.get_kSdevEdge()<<'_'<<ss.get_kSdevBg()<<'_'
		<<ss.get_kContrast()<<'_'<<ss.get_kMCsOffset()<<'_'<<ss.get_kCosAngleMCs()<<'_'
		<<ss.get_kSymDensity()<<'_'<<ss.getBlankThreshold()<<'_'
		<<resized.cols<<'_'<<resized.rows; // no extension yet
	const string studiedCase = oss.str(); // id included in the result & trace file names

	path resultFile(ss.getWorkDir());
	resultFile.append("Output").append(studiedCase).
		concat(".jpg");
	// generating a JPG result file (minor quality loss, but significant space requirements reduction)

	if(exists(resultFile)) {
		result = cv::imread(resultFile.string(), cv::ImreadModes::IMREAD_UNCHANGED);
		timer.release();
		ctrler.reportTransformationProgress(1.);

		infoMsg("This image has already been transformed under these settings.\n"
				"Displaying the available result");
		return;
	}

	me.getReady();

#if defined _DEBUG && !defined UNIT_TESTING
	static const wstring COMMA(L",\t");
	path traceFile(ss.getWorkDir());
	traceFile.append("data_").concat(studiedCase).
		concat(".csv"); // generating a CSV trace file
	wofstream ofs(traceFile.c_str());
	ofs<<"#Row"<<COMMA<<"#Col"<<COMMA<<BestMatch::HEADER<<endl;

	// Unicode symbols are logged in symbol format, while other encodings log their code
	const bool isUnicode = me.usesUnicode();
#endif
	
	const unsigned sz = cfg.symSettings().getFontSz();
	result = cv::Mat(resized.rows, resized.cols, resized.type());

	for(unsigned r = 0U, h = (unsigned)resized.rows; r<h; r += sz) {
		ctrler.reportTransformationProgress((double)r/h);

		for(unsigned c = 0U, w = (unsigned)resized.cols; c<w; c += sz) {
			const cv::Mat patch(resized, cv::Range(r, r+sz), cv::Range(c, c+sz));

#if defined _DEBUG && !defined UNIT_TESTING
			BestMatch best(isUnicode);
#else
			BestMatch best;
#endif
			const cv::Mat approximation = me.approxPatch(patch, best);
			approximation.copyTo(cv::Mat(result, cv::Range(r, r+sz), cv::Range(c, c+sz)));

#if defined _DEBUG && !defined UNIT_TESTING
			ofs<<r/sz<<COMMA<<c/sz<<COMMA<<best<<endl;
#endif
		}
#if defined _DEBUG && !defined UNIT_TESTING
		ofs.flush(); // flush after processing a full row (of height sz) of the image
#endif
	}

#if defined _DEBUG && !defined UNIT_TESTING
	// Flushing and closing the trace file, to be also ready when inspecting the resulted image
	ofs.close();
#endif

#ifndef UNIT_TESTING
	cout<<"Writing result to "<<resultFile<<endl<<endl;
	imwrite(resultFile.string(), result);
#endif
}

#ifndef UNIT_TESTING // Unit Testing module has different implementations for these methods
void Transformer::createOutputFolder() {
	// Ensure there is an Output folder
	path outputFolder = cfg.matchSettings().getWorkDir();
	if(!exists(outputFolder.append("Output")))
		create_directory(outputFolder);
}
#endif