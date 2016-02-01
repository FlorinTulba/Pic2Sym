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
#include "controller.h"
#include "match.h"

#include <sstream>
#include <numeric>

#ifdef _DEBUG
#include <fstream>
#endif

#include <boost/filesystem/operations.hpp>

using namespace std;
using namespace cv;
using namespace boost::filesystem;

namespace {
	// Conversion PixMapSym -> Mat of type double with range [0..1] instead of [0..255]
	Mat toMat(const PixMapSym &pms, unsigned fontSz) {
		Mat result((int)fontSz, (int)fontSz, CV_8UC1, Scalar(0U));

		int firstRow = (int)fontSz-(int)pms.top-1;
		Mat region(result,
				   Range(firstRow, firstRow+(int)pms.rows),
				   Range((int)pms.left, (int)(pms.left+pms.cols)));

		const Mat pmsData((int)pms.rows, (int)pms.cols, CV_8UC1, (void*)pms.pixels.data());
		pmsData.copyTo(region);

		static const double INV_255 = 1./255;
		result.convertTo(result, CV_64FC1, INV_255); // convert to double

		return result;
	}
	
	pair<double, double> averageFgBg(const Mat &patch, const Mat &fgMask, const Mat &bgMask) {
		const Scalar miuFg = mean(patch, fgMask),
				miuBg = mean(patch, bgMask);
		return make_pair(*miuFg.val, *miuBg.val);
	}

	double assessGlyphMatch(const Config &cfg,
							const vector<const Mat> &glyphAndMasks,
							const Mat &patch, const Mat &negPatch,
							Matcher &matcher,
							vector<PixMapSym>::const_iterator itFe,
							const double sz_1, const double sz2) {
		MatchParams &mp = matcher.params;

		const Mat &glyph = glyphAndMasks[0], &negGlyph = glyphAndMasks[1],
				&nonZero = glyphAndMasks[2], &nonOne = glyphAndMasks[3],
				&fgMask = glyphAndMasks[4], &bgMask = glyphAndMasks[5];
		Scalar miu, sdev;
		Mat temp;

		tie(mp.fg, mp.bg) = averageFgBg(patch, fgMask, bgMask);

		if(mp.fg > mp.bg) {
			divide(patch, glyph, temp);
			meanStdDev(temp, miu, sdev, nonZero);
			mp.sdevFg = *sdev.val;

			temp.release();
			divide(negPatch, negGlyph, temp);
			meanStdDev(temp, miu, sdev, nonOne);
			mp.sdevBg = *sdev.val;
		} else {
			divide(negPatch, glyph, temp);
			meanStdDev(temp, miu, sdev, nonZero);
			mp.sdevFg = *sdev.val;

			temp.release();
			divide(patch, negGlyph, temp);
			meanStdDev(temp, miu, sdev, nonOne);
			mp.sdevBg = *sdev.val;
		}

		mp.glyphWeight = itFe->glyphSum / sz2;

		// Obtaining glyph's mass center
		const double k = mp.glyphWeight * (mp.fg-mp.bg),
			delta = .5 * mp.bg * sz_1;
		if(k+mp.bg == 0.)
			mp.mcGlyph = Point2d(sz_1, sz_1) * .5;
		else
			mp.mcGlyph = (k * itFe->mc + Point2d(delta, delta)) / (k + mp.bg);

		return matcher.score(cfg);
	}

	// Determines best match of 'patch' compared to the elements from 'symsSet'
	void findBestMatch(const Config &cfg, const vector<vector<const Mat>> &symsSet,
					   const Mat &patch, Matcher &matcher, BestMatch &best,
					   vector<PixMapSym>::const_iterator itFeBegin,
					   const double sz2, const Mat &consec) {
		best.reset();

		const double patchSum = *sum(patch).val,
				sz_1 = (double)cfg.getFontSz() - 1.;
		Mat temp, temp1;
		reduce(patch, temp, 0, CV_REDUCE_SUM);	// sum all rows
		reduce(patch, temp1, 1, CV_REDUCE_SUM);	// sum all columns

		MatchParams &mp = matcher.params;
		mp.mcPatch = Point2d(temp.dot(consec), temp1.t().dot(consec)) / patchSum; // mass center

		const Mat negPatch = 255. - patch;
		auto itFe = itFeBegin;
		for(const auto &glyphAndMasks : symsSet) {
			const double score =
				assessGlyphMatch(cfg, glyphAndMasks,
								patch, negPatch, matcher, itFe, sz_1, sz2);
			if(score > best.score)
				best.reset(score, (unsigned)distance(itFeBegin, itFe), itFe->symCode, mp);

			++itFe;
		}
	}

	// Writes symsSet[best.symIdx] to the appropriate part (r,c) from result
	void commitMatch(const Config &cfg, const vector<vector<const Mat>> &symsSet,
					 const BestMatch &best, Mat &result,
					 const Mat &resized, const Mat &patch, unsigned r, unsigned c,
					 bool isColor) {
		const auto sz = cfg.getFontSz();
		const vector<const Mat> &glyphMatrices = symsSet[best.symIdx];
		const Mat &glyph = glyphMatrices[0];
		Mat patchResult(result, Range(r, r+sz), Range(c, c+sz));

		if(isColor) {
			const Mat &fgMask = glyphMatrices[4],
					&bgMask = glyphMatrices[5];
			Mat patchColor(resized, Range(r, r+sz), Range(c, c+sz));

			vector<Mat> channels;
			split(patchColor, channels);
			assert(channels.size() == 3);

			double miuFg, miuBg, newDiff, diffFgBg = 0.;
			for(auto &ch : channels) {
				ch.convertTo(ch, CV_64FC1); // processing double values

				tie(miuFg, miuBg) = averageFgBg(ch, fgMask, bgMask);
				newDiff = miuFg - miuBg;

				glyph.convertTo(ch, CV_8UC1, newDiff, miuBg);

				diffFgBg += abs(newDiff);
			}

			if(diffFgBg < 3.*cfg.getBlankThreshold())
				patchResult = mean(patchColor);
			else
				merge(channels, patchResult);

		} else { // grayscale result
			if(abs(best.params.fg - best.params.bg) < cfg.getBlankThreshold())
				patchResult = mean(patch);
			else
				glyph.convertTo(patchResult, CV_8UC1,
								best.params.fg - best.params.bg,
								best.params.bg);
		}
	}
} // anonymous namespace

Transformer::Transformer(Controller &ctrler_, const string &cmd) : ctrler(ctrler_), cfg(ctrler_, cmd), fe(ctrler_), img(ctrler_) {
	// Ensure there is an Output folder
	path outputFolder = cfg.getWorkDir();
	if(!exists(outputFolder.append("Output")))
	   create_directory(outputFolder);
}

string Transformer::getIdForSymsToUse() {
	const unsigned sz = cfg.getFontSz();
	if(!Config::isFontSizeOk(sz)) {
		cerr<<"Invalid font size to use: "<<sz<<endl;
		throw logic_error("Invalid font size for getIdForSymsToUse");
	}

	ostringstream oss;
	oss<<fe.getFamily()<<'_'<<fe.getStyle()<<'_'<<fe.getEncoding()<<'_'<<sz;
	// this also throws logic_error if no family/style

	return oss.str();
}

Transformer::VVMatCItPair Transformer::getSymsRange(unsigned from, unsigned count) const {
	const unsigned sz = (unsigned)symsSet.size();
	const VVMatCIt itEnd = symsSet.cend();
	if(from >= sz)
		return make_pair(itEnd, itEnd);

	const VVMatCIt itStart = next(symsSet.cbegin(), from);
	if(from + count >= sz)
		return make_pair(itStart, itEnd);

	return make_pair(itStart, next(itStart, count));
}

void Transformer::updateSymbols() {
	const string idForSymsToUse = getIdForSymsToUse(); // throws for invalid cmap/size
	if(symsIdReady.compare(idForSymsToUse) == 0)
		return; // already up to date

	static const double STILL_BG = .025,			// darkest shades
						STILL_FG = 1. - STILL_BG;	// brightest shades
	symsSet.clear();
	symsSet.reserve(fe.symsSet().size());

	double minVal, maxVal;
	const unsigned sz = cfg.getFontSz();
	const int szGlyph[] = {2, sz, sz},
			szMasks[] = {4, sz, sz};
	for(const auto &pms : fe.symsSet()) {
		const Mat glyph = toMat(pms, sz), negGlyph = 1. - glyph;

		// for very small fonts, minVal might be > 0 and maxVal might be < 255
		minMaxIdx(glyph, &minVal, &maxVal);

		const Mat nonZero = (glyph != 0.), nonOne = (glyph != 1.),
				fgMask = (glyph > (minVal + STILL_FG * (maxVal-minVal))),
				bgMask = (glyph < (minVal + STILL_BG * (maxVal-minVal)));

		symsSet.emplace_back(vector<const Mat>
				{ glyph, negGlyph, nonZero, nonOne, fgMask, bgMask });
	}

	symsIdReady = idForSymsToUse; // ready to use the new cmap&size
}

void Transformer::run() {
	updateSymbols(); // throws for invalid cmap/size

	Mat gray;
	const Mat resized = img.resized(cfg, &gray); // throws when no image
	ctrler.reportTransformationProgress(0.); // keep it after img.resized, to display updated resized version as comparing image

	ostringstream oss;
	oss<<img.name()<<'_'
		<<getIdForSymsToUse()<<'_'
		<<cfg.get_kContrast()<<'_'<<cfg.get_kSdevFg()<<'_'<<cfg.get_kSdevBg()<<'_'
		<<cfg.get_kCosAngleMCs()<<'_'<<cfg.get_kMCsOffset()<<'_'
		<<cfg.get_kGlyphWeight()<<'_'<<cfg.getBlankThreshold()<<'_'
		<<resized.cols<<'_'<<resized.rows; // no extension yet
	const string studiedCase = oss.str(); // id included in the result & trace file names

	path resultFile(cfg.getWorkDir());
	resultFile.append("Output").append(studiedCase).
		concat(".jpg");
	// generating a JPG result file (minor quality loss, but significant space requirements reduction)

	if(exists(resultFile)) {
		result = imread(resultFile.string(), ImreadModes::IMREAD_UNCHANGED);
		ctrler.reportTransformationProgress(1.);

		MessageBox(nullptr, L"This image has already been transformed under these settings.\n" \
				   L"Displaying the available result",
				   L"Information", MB_ICONINFORMATION | MB_OK | MB_TASKMODAL | MB_SETFOREGROUND);
		return;
	}

	oss.str(""); oss.clear();
	oss<<resultFile; // contains also the double quotes needed when the path contains Spaces
	string quotedResultFile(oss.str());

#ifdef _DEBUG
	path traceFile(cfg.getWorkDir());
	traceFile.append("data_").concat(studiedCase).
		concat(".csv"); // generating a CSV trace file
	wofstream ofs(traceFile.c_str());
	ofs<<"#Row,\t#Col,\t"<<BestMatch::HEADER<<endl;
#endif

	const unsigned sz = cfg.getFontSz();
	const double sz2 = (double)sz*sz;

	Mat consec(1, sz, CV_64FC1);
	iota(consec.begin<double>(), consec.end<double>(), 0.);

	Mat temp, temp1;
	Matcher matcher(sz, fe.smallGlyphsCoverage());
	MatchParams &mp = matcher.params;
	BestMatch best(fe.getEncoding().compare("UNICODE") == 0); // holds the best grayscale match found at a given time
	result = Mat(resized.rows, resized.cols, resized.type());
	gray.convertTo(gray, CV_64FC1);
	
	const auto itFeBegin = fe.symsSet().cbegin();
	for(unsigned r = 0U, h = (unsigned)gray.rows; r<h; r += sz) {
		ctrler.reportTransformationProgress((double)r/h);

		for(unsigned c = 0U, w = (unsigned)gray.cols; c<w; c += sz) {
			const Mat patch(gray, Range(r, r+sz), Range(c, c+sz));

			findBestMatch(cfg, symsSet, patch, matcher, best, itFeBegin, sz2, consec);

#ifdef _DEBUG
			ofs<<r/sz<<",\t"<<c/sz<<",\t"<<best<<endl;
#endif

			commitMatch(cfg, symsSet, best, result, resized, patch, r, c, img.isColor());
		}
#ifdef _DEBUG
		ofs.flush(); // flush after processing a full row (of height sz) of the image
#endif
	}

#ifdef _DEBUG
	// Flushing and closing the trace file, to be also ready when inspecting the resulted image
	ofs.close();
#endif

	cout<<"Writing result to "<<resultFile<<endl<<endl;
	imwrite(resultFile.string(), result);
	
	ctrler.reportTransformationProgress(1.);
}
