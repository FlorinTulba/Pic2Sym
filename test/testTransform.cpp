/**********************************************************
 Project:     UnitTesting
 File:        testTransform.cpp

 Author:      Florin Tulba
 Created on:  2016-1-17
 
 Copyright (c) 2016 Florin Tulba
 **********************************************************/

#include "testMain.h"
#include <fstream>

#include "transform.cpp" // including CPP, to recompile #if(n)def UNIT_TESTING regions
#include "match.cpp" // including CPP, to recompile #if(n)def UNIT_TESTING regions
#include "ui.cpp"
#include "misc.cpp"

BOOST_AUTO_TEST_SUITE(Transform_Tests)
BOOST_AUTO_TEST_CASE(Check_symbols_set) {
	Controller c;
	Img img(c);
	FontEngine fe(c);
	Config cfg(c, 10U, 500U, 300U, 0U, 1., 1., 1., 0., 0., 0., 0.);
	MatchEngine me(c, cfg, fe);
	Transformer t(c, cfg, me, img);

	bool correct = false;
	const unsigned sz = cfg.getFontSz();
	const double sz2 = (double)sz*sz, sz_1 = sz - 1.;

	BOOST_REQUIRE_NO_THROW(correct = fe.newFont("res\\BPmonoBold.ttf"));
	//BOOST_REQUIRE_NO_THROW(correct = fe.newFont("C:\\Windows\\Fonts\\courbd.ttf"));
	BOOST_REQUIRE(correct);

	BOOST_REQUIRE_NO_THROW(correct = fe.setEncoding("APPLE_ROMAN"));
	BOOST_REQUIRE(correct);

	BOOST_REQUIRE_NO_THROW(fe.setFontSz(sz));

// 	me.updateSymbols();
// 
// 	Matcher matcher(sz, fe.smallGlyphsCoverage()), expectedMatcher(matcher);
// 	Mat consec(1, sz, CV_64FC1);
// 	iota(consec.begin<double>(), consec.end<double>(), 0.);
// 	vector<vector<const Mat>> symsSet;
// 	double minVal, maxVal;
// 	static const double STILL_BG = .025,
// 		STILL_FG = 1. - STILL_BG;
// 	for(const auto &pmc : fe.symsSet()) {
// 		const Mat glyph = toMat(pmc, sz), negGlyph = 1. - glyph;
// 
// 		// for very small fonts, minVal might be > 0 and maxVal might be < 255
// 		minMaxIdx(glyph, &minVal, &maxVal);
// 
// 		const Mat nonZero = (glyph != 0.), nonOne = (glyph != 1.),
// 			fgMask = (glyph > (minVal + STILL_FG * (maxVal-minVal))),
// 			bgMask = (glyph < (minVal + STILL_BG * (maxVal-minVal)));
// 
// 		;
// 		symsSet.emplace_back(vector<const Mat> { glyph, negGlyph, nonZero, nonOne, fgMask, bgMask });
// 	}
// 
// 	const auto itFeBegin = fe.symsSet().cbegin();
// 	auto itFe = itFeBegin;
// 	BestMatch best(fe.getEncoding().compare("UNICODE") == 0), // holds the best grayscale match
// 		expectedBest(best);
// 	me.getReady();
// 
// 	vector<tuple<Mat, BestMatch, BestMatch>> errors;
// 	for(size_t i = 0UL, len = symsSet.size(); i<len; ++i, ++itFe) {
//  		const Mat glyph = 255.*symsSet[i][0], negGlyph = 255.-glyph;
// 
// 		me.findBestMatch(glyph, matcher, best);
// 		if(i != (size_t)best.symIdx) {
// 			const double score =
// 				assessGlyphMatch(cfg, symsSet[i], glyph, negGlyph,
// 								expectedMatcher, itFe, sz_1, sz2);
// 			expectedBest.reset(score, (unsigned)i, itFe->symCode, expectedMatcher.params);
// 			errors.emplace_back(glyph, best, expectedBest);
// 		}
// 
// 		me.findBestMatch(negGlyph, matcher, best);
// 		if(i != (size_t)best.symIdx) {
// 			const double score =
// 				assessGlyphMatch(cfg, symsSet[i], negGlyph, glyph,
// 								expectedMatcher, itFe, sz_1, sz2);
// 			expectedBest.reset(score, (unsigned)i, itFe->symCode, expectedMatcher.params);
// 			errors.emplace_back(glyph, best, expectedBest);
// 		}
// 
// 		cout<<fixed<<setprecision(5)<<100.*i/len<<"% ("<<setw(6)<<errors.size()<<" unexpected matches)\r";
// 	}
// 	cout<<endl;
// 	int errCount = (int)errors.size();
// 	cout<<"There were "<<errCount<<" unexpected matches."<<endl;
// 	if(errCount!=0UL) {
// 		int itemsPerSide = (int)ceil(sqrt(errCount)), squareSide = sz*itemsPerSide;
// 		Mat expected(squareSide, squareSide, CV_8UC1, Scalar(127)),
// 			resulted(squareSide, squareSide, CV_8UC1, Scalar(127));
// 		wofstream ofs("debug.csv");
// 		ofs<<BestMatch::HEADER<<endl;
// 		for(int r = 0, i = 0; i<errCount && r<itemsPerSide; ++r) {
// 			Range rowRange(r*sz, (r+1)*sz);
// 			for(int c = 0; i<errCount && c<itemsPerSide; ++c, ++i) {
// 				Range colRange(c*sz, (c+1)*sz);
// 				
// 				get<0>(errors[i]).convertTo(Mat(expected, rowRange, colRange), CV_8UC1);
// 
// 				best = get<1>(errors[i]);
// 				Mat &reportedMatch = symsSet[best.symIdx][0];
// 				reportedMatch.convertTo(Mat(resulted, rowRange, colRange), CV_8UC1,
// 										(best.params.fg - best.params.bg),
// 										best.params.bg);
// 				ofs<<best<<endl;
// 				best = get<2>(errors[i]);
// 				ofs<<best<<endl;
// 			}
// 		}
// 		ofs.close();
// 		Comparator comp(c);
// 		comp.permitResize();
// 		comp.setReference(expected);
// 		comp.setResult(resulted);
// 		while(27!=cv::waitKey());
// 	}
}
BOOST_AUTO_TEST_SUITE_END() // FontEngine_Tests
