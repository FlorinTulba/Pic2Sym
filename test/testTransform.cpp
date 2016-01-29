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
#include "ui.cpp"

BOOST_AUTO_TEST_SUITE(Transform_Tests)
BOOST_AUTO_TEST_CASE(Check_symbols_set) {
	Controller c;
	bool correct = false;
// 	Config cfg(c, 10U, 500U, 300U, 0U, .4, .9, 0., 0., 0., 0.);
	//Config cfg(c, 10U, 500U, 300U, 0U, 1., 2.25, 0., 0.0001, 0., 0.);
	Config cfg(c, 10U, 500U, 300U, 0U, 1., 1.41, 0., 0., 0., 0.);
	const unsigned sz = cfg.getFontSz();
	const double sz2 = (double)sz*sz, sz_1 = sz - 1.;
	FontEngine fe(c);
	//BOOST_REQUIRE_NO_THROW(correct = fe.newFont("res\\BPmonoBold.ttf"));
	BOOST_REQUIRE_NO_THROW(correct = fe.newFont("C:\\Windows\\Fonts\\courbd.ttf"));
	BOOST_REQUIRE(correct);

// 	BOOST_REQUIRE_NO_THROW(correct = fe.setEncoding("APPLE_ROMAN"));
// 	BOOST_REQUIRE(correct);

	BOOST_REQUIRE_NO_THROW(fe.setFontSz(sz));

	Matcher matcher(sz, fe.smallGlyphsCoverage()), expectedMatcher(matcher);
	Mat consec(1, sz, CV_64FC1);
	iota(consec.begin<double>(), consec.end<double>(), 0.);
	vector<pair<Mat, Mat>> symsSet;
	for(auto &pmc : fe.symsSet()) {
		Mat glyph = toMat(pmc, sz), negGlyph = 1. - glyph;
		symsSet.emplace_back(glyph, negGlyph);
	}

	auto itFeBegin = fe.symsSet().cbegin(), itFe = itFeBegin;
	BestMatch best(fe.getEncoding().compare("UNICODE") == 0), // holds the best grayscale match
		expectedBest(best);
	Mat glyph;
	
	vector<tuple<Mat, BestMatch, BestMatch>> errors;
	for(size_t i = 0UL, len = symsSet.size(); i<len; ++i, ++itFe) {
		glyph = symsSet[i].first;
		
		findBestMatch(cfg, symsSet, glyph, matcher, best, itFeBegin, sz2, consec);
		if(i != (size_t)best.symIdx) {
			const double score =
				assessGlyphMatch(cfg, symsSet[i], glyph, *sum(glyph).val, expectedMatcher, itFe, sz_1, sz2);
			expectedBest.reset(score, (unsigned)i, itFe->symCode, expectedMatcher.params);
			errors.emplace_back(glyph, best, expectedBest);
		}

		cout<<fixed<<setprecision(5)<<100.*i/len<<"% ("<<setw(6)<<errors.size()<<" unexpected matches)\r";
	}
	cout<<endl;
	int errCount = (int)errors.size();
	cout<<"There were "<<errCount<<" unexpected matches."<<endl;
	if(errCount!=0UL) {
		int itemsPerSide = (int)ceil(sqrt(errCount)), squareSide = sz*itemsPerSide;
		Mat expected(squareSide, squareSide, CV_8UC1, Scalar(127)),
			resulted(squareSide, squareSide, CV_8UC1, Scalar(127));
		wofstream ofs("debug.csv");
		ofs<<BestMatch::HEADER<<endl;
		for(int r = 0, i = 0; i<errCount && r<itemsPerSide; ++r) {
			Range rowRange(r*sz, (r+1)*sz);
			for(int c = 0; i<errCount && c<itemsPerSide; ++c, ++i) {
				Range colRange(c*sz, (c+1)*sz);
				
				get<0>(errors[i]).convertTo(Mat(expected, rowRange, colRange), CV_8UC1, 255.);

				best = get<1>(errors[i]);
				Mat &reportedMatch = symsSet[best.symIdx].first;
				reportedMatch.convertTo(Mat(resulted, rowRange, colRange), CV_8UC1,
										255.*(best.params.miuFg - best.params.miuBg),
										255.*best.params.miuBg);
				ofs<<best<<endl;
				best = get<2>(errors[i]);
				ofs<<best<<endl;
			}
		}
		ofs.close();
		Comparator comp(c);
		comp.permitResize();
		comp.setReference(expected);
		comp.setResult(resulted);
		while(27!=cv::waitKey());
	}
}
BOOST_AUTO_TEST_SUITE_END() // FontEngine_Tests
