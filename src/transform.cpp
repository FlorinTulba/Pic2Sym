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
using namespace boost::filesystem;

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
			cout<<endl<<"Please select a new image ..."<<endl;
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
		
		//fe.generateCharmapCharts(cfg.getWorkDir());
	}

	// Conversion PixMapChar -> Mat of type double with range [0..1] instead of [0..255]
	Mat toMat(const PixMapChar &pmc, unsigned fontSz) {
		Mat result((int)fontSz, (int)fontSz, CV_8UC1, Scalar(0U));

		int firstRow = (int)fontSz-(int)pmc.top-1;
		Mat region(result,
				   Range(firstRow, firstRow+(int)pmc.rows),
				   Range((int)pmc.left, (int)(pmc.left+pmc.cols)));

		Mat pmcData((int)pmc.rows, (int)pmc.cols, CV_8UC1, (void*)pmc.data);
		pmcData.copyTo(region);

		static const double INV_255 = 1./255;
		result.convertTo(result, CV_64FC1, INV_255); // convert to double

		return result;
	}
	
	// Holds mean and standard deviation (grayscale matching) for foreground/background pixels
	struct MatchParams {
		double miuFg, miuBg;
		double aseFg, aseBg;

		/*
		Returns a small positive value for better correlations.
		Good correlation is a subjective matter.

		Several considered factors are mentioned below.
		* = mandatory;  + = nice to have

		A) Separately, each patch needs to:
		* 1. minimize the difference between the selected foreground glyph
		and the covered part from the original patch

		* 2. minimize the difference between the remaining background around the selected glyph
		and the corresponding part from the original patch

		* 3. have enough contrast between foreground selected glyph & background

		+ 4. have the color of the largest part (foreground / background)
		as close as possible to the mean of the entire original patch.
		Gain: Smoothness;		Drawback: Less Contrast(A3)

		B) Together, the patches should also preserve region's gradients:
		+ 5. Prefer glyphs that respect the gradient of their corresponding patch.
		Gain: More natural;	Drawback: More computations

		+ 6. Consider the gradient within a slightly extended patch
		Gain: Smoothness;		Drawback: Complexity


		Point A3 just maximizes the difference between the means of the fg & bg.
		Points A1&2 minimizes the standard deviation (or a similar measure) on each region.

		The remaining points might be addressed in a future version.
		*/
		double score() const {
			register const double divider = abs(miuBg - miuFg) + 1e-3;
			// return (aseFg * aseBg)/divider;
			// return (aseFg * aseBg)/(divider*divider);
			// return (aseFg + aseBg)/divider;
			return (sqrt(aseFg) + sqrt(aseBg))/divider; // preferred one so far
		}
	};

	// Holds the best grayscale match found at a given time
	struct BestMatch {
		double score = numeric_limits<double>::infinity();
		unsigned charIdx = UINT_MAX; // no best yet
		MatchParams params;
	};
} // anonymous namespace

Transformer::Transformer(Config &cfg_) : cfg(cfg_) {
	// Ensure there is an Output folder
	path outputFolder = cfg.getWorkDir();
	if(!exists(outputFolder.append("Output")))
	   create_directory(outputFolder);
}

void Transformer::reconfig() {
	string initCharsetId = fe.fontId(); // remember initial charset

	cfg.update(); // Allow setting new parameters for the transformation
	selectFont(fe, cfg); // Configure font set

	if(fe.fontId() != initCharsetId) {
		unsigned sz = cfg.getFontSz();
		charset.clear();
		charset.reserve(fe.charset().size());

		for(auto &pmc : fe.charset()) {
			Mat glyph = toMat(pmc, sz), negGlyph = 1. - glyph;
			charset.emplace_back(glyph, negGlyph);
		}
	}
}

void Transformer::run() {

	selectImage(img);

	ostringstream oss; oss<<img.name()<<'_'<<fe.fontId(); // no extension yet
	const string studiedCase = oss.str(); // id included in the result & trace file names

#ifdef _DEBUG
	path traceFile(cfg.getWorkDir());
	traceFile.append("data_").concat(studiedCase).
		concat(".csv"); // generating a CSV trace file
	ofstream ofs(traceFile.c_str());
	ofs<<"#ChosenScore,\t#miuFg,\t#miuBg,\t#aseFg,\t#aseBg"<<endl;
#endif

	auto itFeBegin = fe.charset().cbegin();
	unsigned sz = cfg.getFontSz();
	const double sz2 = (double)sz*sz;
	Mat temp, gray;
	Mat resized = img.resized(cfg, &gray);
	Mat result(resized.rows, resized.cols, resized.type());
	gray.convertTo(gray, CV_64FC1);

	for(unsigned r = 0U, h = (unsigned)gray.rows; r<h; r += sz) {
		// Reporting progress
		printf("%6.2f%%\r", r*100./h); // simpler to format percent values with printf

		for(unsigned c = 0U, w = (unsigned)gray.cols; c<w; c += sz) {
			Mat patch(gray, Range(r, r+sz), Range(c, c+sz)),
				patchResult(result, Range(r, r+sz), Range(c, c+sz));

			double patchSum = *sum(patch).val;

			BestMatch best; // holds the best grayscale match found at a given time

			Mat glyph, negGlyph;
			auto itFe = fe.charset().cbegin();
			for(auto &glyphAndNegative : charset) {
				tie(glyph, negGlyph) = glyphAndNegative;

				MatchParams mp;

				// Computing foreground and background means plus the average of squared errors.
				// 'mean' & 'meanStdDev' provided by OpenCV have binary masking.
				// Our glyphs have gradual transitions between foreground & background.
				// These transition need less than the full weighting provided by 'mean' & 'meanStdDev'.
				// Implemented below edge weighting.

				// First the means:
				double glyphSum = itFe->cachedSum / 255.,
					negGlyphSum = sz2 - glyphSum;

				double dotP = patch.dot(glyph);
				mp.miuFg = dotP / glyphSum; // 'mean' would divide by more (countNonZero)
				mp.miuBg = (patchSum - dotP) / negGlyphSum; // 'mean' would divide by more (countLessThan1)

				// Now the average of squared errors (cheaper to compute than standard deviations):
				// 'meanStdDev' would subtract from patch elements different means
				// and divide by the larger values described above
				temp = (patch-mp.miuFg).mul(glyph); // Elem-wise (mask only fg, weight contours)
				mp.aseFg = temp.dot(temp) / glyphSum;

				temp = (patch-mp.miuBg).mul(negGlyph); // Elem-wise (mask only bg, weight contours)
				mp.aseBg = temp.dot(temp) / negGlyphSum;

				double score = mp.score();
				if(score < best.score) {
					best.score = score;
					best.charIdx = (unsigned)distance(itFeBegin, itFe);
					best.params = mp;
				}

				++itFe;
			}

#ifdef _DEBUG
			ofs<<best.score<<",\t"
				<<best.params.miuFg<<",\t"<<best.params.miuBg<<",\t"
				<<best.params.aseFg<<",\t"<<best.params.aseBg<<endl;
#endif

			// write match to patchResult
			tie(glyph, negGlyph) = *next(charset.begin(), best.charIdx);
			if(img.isRGB()) {
				Mat patchRGB(resized, Range(r, r+sz), Range(c, c+sz));
				double chSum = next(itFeBegin, best.charIdx)->cachedSum / 255.,
					chInvSum = sz2 - chSum;

				vector<Mat> channels;
				split(patchRGB, channels);
				assert(channels.size() == 3);

				double diffFgBg = 0.;
				for(auto &ch : channels) {
					ch.convertTo(ch, CV_64FC1); // processing double values

					double miuFg = ch.dot(glyph) / chSum,
						miuBg = ch.dot(negGlyph) / chInvSum,
						newDiff = miuFg - miuBg;

					glyph.convertTo(ch, CV_8UC1,
									newDiff,
									miuBg);

					diffFgBg += abs(newDiff);
				}

				if(diffFgBg < 3.*cfg.getBlankThreshold())
					patchResult = mean(patchRGB);
				else
					merge(channels, patchResult);

			} else { // grayscale result
				if(abs(best.params.miuFg - best.params.miuBg) < cfg.getBlankThreshold())
					patchResult = mean(patch);
				else {
					glyph.convertTo(patchResult, CV_8UC1,
									best.params.miuFg - best.params.miuBg,
									best.params.miuBg);
				}
			}
		}
	}

	path resultFile(cfg.getWorkDir());
	resultFile.append("Output").append(studiedCase).
		concat(".bmp"); // generating a BMP result file
	
	cout<<"Writing result to "<<resultFile<<endl<<endl;
	imwrite(resultFile.string(), result);
	system(resultFile.string().c_str());
}
