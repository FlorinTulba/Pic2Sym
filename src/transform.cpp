/**********************************************************
 Project:     Pic2Sym
 File:        transform.cpp

 Author:      Florin Tulba
 Created on:  2016-1-6
 
 Copyright (c) 2016 Florin Tulba
 **********************************************************/

#include "transform.h"

#include "misc.h"
#include "dlgs.h"

#include <cstdio>
#include <sstream>

#ifdef _DEBUG
#include <fstream>
#endif

using namespace std;
using namespace cv;

namespace {
	// Let the user pick a new image to process or continue to use the existing one
	void selectImage(Img &img) {
		static FileOpen fo;
		bool readNewImg = true;
		if(!fo.selection().empty()) {
			ostringstream oss;
			oss<<endl<<"Current image is '"<<fo.selection()<<"'.\nKeep working with it?";
			readNewImg = !boolPrompt(oss.str());
		}
		if(readNewImg) {
			cout<<"Please select a new image ..."<<endl;
			while(fo.canceled() || fo.selection().empty() ||
				  !img.reset(fo.selection()))
				  fo.reset();
		}
	}

	// Let the user pick a new font to use or continue to utilize the existing one
	void selectFont(FontEngine &fe, Config &cfg) {
		static SelectFont sf;
		bool readNewFont = true;
		if(!sf.selection().empty()) {
			ostringstream oss;
			oss<<endl<<"Current font is '"<<sf.selection()<<"'.\nKeep working with it?";
			readNewFont = !boolPrompt(oss.str());
		}
		if(readNewFont) {
			cout<<"Please select a new font ..."<<endl;
			FT_Face face;
			while(sf.canceled() || sf.selection().empty() ||
				  !fe.checkFontFile(sf.selection(), face))
				  sf.reset();
			fe.setFace(face);
		}

		fe.selectEncoding();
		fe.setFontSz(cfg.getFontSz());

		//fe.generateCharmapCharts();
	}

	// Conversion PixMapChar -> Mat using font size <sz>
	Mat toMat(const PixMapChar &pmc, unsigned sz) {
		Mat result((int)sz, (int)sz, CV_8UC1, Scalar(0U));

		int firstRow = (int)sz-(int)pmc.top-1;
		Mat region(result,
				   Range(firstRow, firstRow+(int)pmc.rows),
				   Range((int)pmc.left, (int)(pmc.left+pmc.cols)));

		Mat pmcData((int)pmc.rows, (int)pmc.cols, CV_8UC1, (void*)pmc.data);
		pmcData.copyTo(region);

		return result;
	}
	
	// Holds mean and standard deviation (grayscale matching) for foreground/background pixels
	struct MatchParams {
		double miuFg, miuBg;
		double sdevFg, sdevBg;
	};

	// Holds the best grayscale match found at a given time
	struct BestMatch {
		double score = numeric_limits<double>::infinity();
		unsigned charIdx = UINT_MAX; // no best yet
		MatchParams params;
	};

	// Returns a small positive value for better correlations
	double evalFit(const MatchParams &bm) {
		return (bm.sdevFg + bm.sdevBg)/(abs(bm.miuBg - bm.miuFg) + 1e-3);
	}
} // anonymous namespace

Transformer::Transformer(Config &cfg_) : cfg(cfg_) {}

void Transformer::reconfig() {
	string initCharsetId = fe.fontId(); // remember initial charset

	cfg.update(); // Allow setting new parameters for the transformation
	selectFont(fe, cfg); // Configure font set

	if(fe.fontId() != initCharsetId) {
		unsigned sz = cfg.getFontSz();
		charset.clear();
		charset.reserve(fe.charset().size());

		for(auto &pmc : fe.charset())
			charset.emplace_back(toMat(pmc, sz));
	}
}

void Transformer::run() {
#ifdef _DEBUG
	ofstream ofs("data.csv");
	ofs<<"#ChosenScore,\tmiuFg,\tmiuBg,\tsdevFg,\tsdevBg"<<endl;
#endif

	selectImage(img);

	auto itFeBegin = fe.charset().cbegin();
	unsigned sz = cfg.getFontSz();
	const unsigned long sz2 = (unsigned long)sz*sz,
		sz2_255 = sz2*255UL; // is less than ULONG_MAX for fontSz <= 50
	Mat temp, gray, resized = img.resized(cfg, &gray);
	Mat result(resized.rows, resized.cols, resized.type());

	for(unsigned r = 0U, h = (unsigned)gray.rows; r<h; r += sz) {
		// Reporting progress
		printf("%6.2f%%  ", r*100./h); // simpler to format percents with printf

		Mat row(gray, Range(r, r+sz)),
			rowResult(result, Range(r, r+sz));
		for(unsigned c = 0U, w = (unsigned)gray.cols; c<w; c += sz) {
			Mat patch(row, Range::all(), Range(c, c+sz)), patchDouble,
				patchResult(rowResult, Range::all(), Range(c, c+sz));
			patch.convertTo(patchDouble, CV_64FC1);

			unsigned long pSum = (unsigned long)sum(patch)[0],
				pSum_255 = pSum*255UL; // is less than ULONG_MAX for fontSz <= 50

			BestMatch best; // holds the best grayscale match found at a given time

			auto itFe = fe.charset().cbegin();
			for(auto &cs : charset) {
				Mat csDouble;
				cs.convertTo(csDouble, CV_64FC1);

				unsigned long chSum = itFe->cachedSum,
					chInvSum = sz2_255 - chSum;
				assert(chSum != 0U && chInvSum != 0U); // Empty/Full chars were already discarded
				double dotP = patch.dot(cs);
				
				MatchParams mp;
				mp.miuFg = dotP/chSum;
				mp.miuBg = (pSum_255 - dotP)/chInvSum;
				assert(mp.miuBg >= 0.);

				temp = (patchDouble-mp.miuFg).mul(csDouble); // Elem-wise multiply
				mp.sdevFg = sqrt(temp.dot(temp)/(255.*chSum)); // not subtracting the 1 from denominator

				temp = (patchDouble-mp.miuBg).mul(255.-csDouble); // Elem-wise multiply
				mp.sdevBg = sqrt(temp.dot(temp)/(255.*chInvSum)); // not subtracting the 1 from denominator

				double ratio = evalFit(mp);
				if(ratio < best.score) {
					best.score = ratio;
					best.charIdx = (unsigned)distance(itFeBegin, itFe);
					best.params = mp;
				}

				++itFe;
			}

#ifdef _DEBUG
			ofs<<best.score<<",\t"
				<<best.params.miuFg<<",\t"<<best.params.miuBg<<",\t"
				<<best.params.sdevFg<<",\t"<<best.params.sdevBg<<endl;
#endif

			// write match to patchResult
			Mat match = *next(charset.begin(), best.charIdx),
				matchDouble;
			if(img.isRGB()) {
				Mat patchRGB(resized, Range(r, r+sz), Range(c, c+sz));
				unsigned long chSum = next(itFeBegin, best.charIdx)->cachedSum,
					chInvSum = sz2_255 - chSum;

				vector<Mat> channels;
				split(patchRGB, channels);
				assert(channels.size() == 3);

				double diffFgBg = 0.;
				for(auto &ch : channels) {
					double dotP = ch.dot(match);
					double miuFg = dotP/chSum,
						miuBg = ch.dot((unsigned char)255 - match)/chInvSum;
					assert(miuBg >= 0.);

					diffFgBg += abs(miuFg - miuBg);

					match.convertTo(matchDouble, CV_64FC1,
									(miuFg - miuBg) / 255.,
									miuBg);
					matchDouble.convertTo(ch, CV_8UC1);
					dotP = dotP;
				}

				if(diffFgBg < 3*cfg.getBlankThreshold())
					patchResult = mean(patchRGB);
				else
					merge(channels, patchResult);
			} else { // grayscale result
				if(abs(best.params.miuFg - best.params.miuBg) < cfg.getBlankThreshold())
					patchResult = mean(patch);
				else {
					match.convertTo(matchDouble, CV_64FC1,
									(best.params.miuFg - best.params.miuBg) / 255.,
									best.params.miuBg);
					matchDouble.convertTo(patchResult, CV_8UC1);
				}
			}
			cout<<'.';
		}
		cout<<endl;
	}
	cout<<"100%"<<endl<<endl;

	ostringstream oss;
	oss<<img.name()<<'_'<<cfg.getFontSz()<<".bmp";
	imwrite(oss.str(), result);
}
