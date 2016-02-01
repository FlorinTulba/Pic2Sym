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
	// Holds relevant data during patch&glyph matching
	struct MatchParams {
		static const wstring HEADER; // table header when values are serialized

		Point2d mcPatch;			// mass center for the patch
		Point2d mcGlyph;			// glyph's mass center
		double glyphWeight;			// % of the box covered by the glyph (0..1)
		double fg, bg;				// color for fg / bg (range 0..255)

		// standard deviations for fg / bg
		// 0 .. 255^2/(2*sqrt(2)); best when 0
		double sdevFg, sdevBg;

		friend wostream& operator<<(wostream &os, const MatchParams &mp) {
			os<<mp.mcGlyph.x<<",\t"<<mp.mcGlyph.y<<",\t"<<mp.mcPatch.x<<",\t"<<mp.mcPatch.y<<",\t"
				<<mp.fg<<",\t"<<mp.bg<<",\t"<<mp.sdevFg<<",\t"<<mp.sdevBg<<",\t"<<mp.glyphWeight;
			return os;
		}
	};

	const wstring MatchParams::HEADER(L"#mcGlyphX,\t#mcGlyphY,\t#mcPatchX,\t#mcPatchY,\t"
									 L"#fg,\t#bg,\t#sdevFg,\t#sdevBg,\t#fg/all");

	/*
	Class for assessing a match based on various criteria.
	*/
	class Matcher {
		const unsigned fontSz_1;		// size of the font - 1
		const double smallGlyphsCoverage; // max ratio of glyph area / containing area for small symbols
		const double PREFERRED_RADIUS;	// allowed distance between the mc-s of patch & chosen glyph
		const double MAX_MCS_OFFSET;	// max distance possible between the mc-s of patch & chosen glyph
		// happens for the extremes of a diagonal

		const Point2d centerPatch;		// center of the patch

	public:
		MatchParams params;

		Matcher(unsigned fontSz, double smallGlyphsCoverage_) :
			fontSz_1(fontSz - 1U), smallGlyphsCoverage(smallGlyphsCoverage_),
			PREFERRED_RADIUS(3U * fontSz / 8.), MAX_MCS_OFFSET(sqrt(2) * fontSz_1),
			centerPatch(fontSz_1/2., fontSz_1/2.) {}

		/*
		Returns a larger positive value for better correlations.
		Good correlation is a subjective matter.

		Several considered factors are mentioned below.
		* = mandatory;  + = nice to have

		A) Separately, each patch needs to:
		* 1. minimize the difference between the selected foreground glyph
		and the covered part from the original patch

		* 2. minimize the difference between the remaining background around the selected glyph
		and the corresponding part from the original patch

		* 3. have enough contrast between foreground & background of the selected glyph.

		B) Together, the patches should also preserve region's gradients:
		* 1. Prefer glyphs that respect the gradient of their corresponding patch.

		+ 2. Consider the gradient within a slightly extended patch
		Gain: Smoothness;		Drawback: Complexity

		C) Balance the accuracy with more artistic aspects:
		+ 1. use largest possible 'matching' glyph, not just dots, quotes, commas


		Points A1&2 minimize the standard deviation (or a similar measure) on each region.

		Point A3 encourages larger differences between the means of the fg & bg.

		Point B1 ensures the weight centers of the glyph and patch are close to each other
		as distance and as direction.

		Point C1 favors larger glyphs.

		The remaining points might be addressed in a future version.
		*/
		double score(const Config &cfg) const {
			static const double SQRT2 = sqrt(2), TWO_SQRT2 = 2. - SQRT2;

			// for a histogram with just 2 equally large bins on 0 and 255^2 =>
			// mean = 255^2/2. = 32512.5; sdev = 255^2/(2*sqrt(2)) ~ 22989.81
			static const double SDEV_MAX = 255*255/(2.*SQRT2);
			static const double MIN_CONTRAST_BRIGHT = 2., // less contrast needed for bright tones
								MIN_CONTRAST_DARK = 5.; // more contrast needed for dark tones
			static const double CONTRAST_RATIO = (MIN_CONTRAST_DARK - MIN_CONTRAST_BRIGHT) / (2.*255);
			static const Point2d ORIGIN; // (0, 0)

			/////////////// CORRECTNESS FACTORS (Best Matching & Good Contrast) ///////////////
			// Range 0..1, acting just as penalty for bad standard deviations.
			// Closer to 1 for good sdev of fg;  tends to 0 otherwise.
			register const double fSdevFg = pow(1. - params.sdevFg / SDEV_MAX,
												cfg.get_kSdevFg());
			register const double fSdevBg = pow(1. - params.sdevBg / SDEV_MAX,
												cfg.get_kSdevBg());

			const double minimalContrast = // minimal contrast for the average brightness
				MIN_CONTRAST_BRIGHT + CONTRAST_RATIO * (params.fg + params.bg);
			// range 0 .. 255, best when large
			const double contrast = abs(params.bg - params.fg);
			// Encourage contrasts larger than minimalContrast:
			// <1 for low contrast;  1 for minimalContrast;  >1 otherwise
			register double fMinimalContrast = pow(contrast / minimalContrast,
												   cfg.get_kContrast());

			/////////////// SMOOTHNESS FACTORS (Similar gradient) ///////////////
			// best glyph location is when mc-s are near to each other
			// range 0 .. 1.42*fontSz_1, best when 0
			const double mcsOffset = norm(params.mcPatch - params.mcGlyph);
			// <=1 for mcsOffset >= PREFERRED_RADIUS;  >1 otherwise
			register const double fMinimalMCsOffset =
				pow(1. + (PREFERRED_RADIUS - mcsOffset) / MAX_MCS_OFFSET,
					cfg.get_kMCsOffset());

			const Point2d relMcPatch = params.mcPatch - centerPatch;
			const Point2d relMcGlyph = params.mcGlyph - centerPatch;

			// best gradient orientation when angle between mc-s is 0 => cos = 1
			// Maintaining the cosine of the angle is ok, as it stays near 1 for small angles.
			// -1..1 range, best when 1
			double cosAngleMCs = 0.;
			if(relMcGlyph != ORIGIN && relMcPatch != ORIGIN) // avoid DivBy0
				cosAngleMCs = relMcGlyph.dot(relMcPatch) /
								(norm(relMcGlyph) * norm(relMcPatch));

			// <=1 for |angleMCs| >= 45;  >1 otherwise
			// lessen the importance for small mcsOffset-s (< PREFERRED_RADIUS)
			register const double fMCsAngleLessThan45 = pow((1. + cosAngleMCs) * TWO_SQRT2,
				cfg.get_kCosAngleMCs() * min(mcsOffset, PREFERRED_RADIUS) / PREFERRED_RADIUS);

			/////////////// FANCINESS FACTOR (Larger glyphs) ///////////////
			// <=1 for glyphs considered small;   >1 otherwise
			register const double fGlyphWeight = pow(params.glyphWeight + 1. - smallGlyphsCoverage,
				cfg.get_kGlyphWeight());

			const double result = fSdevFg * fSdevBg * fMinimalContrast *
				fMinimalMCsOffset * fMCsAngleLessThan45 * fGlyphWeight;

			return result;
		}
	};

	// Holds the best grayscale match found at a given time
	struct BestMatch {
		double score;			// score of the best
		unsigned symIdx;		// index within vector<PixMapSym>
		unsigned long symCode;	// glyph code
		const bool unicode;		// is the charmap in Unicode?

		MatchParams params;	// parameters of the match for the best glyph

		BestMatch(bool isUnicode = true) : unicode(isUnicode) { reset(); }
		BestMatch(const BestMatch&) = default;
		BestMatch& operator=(const BestMatch &other) {
			if(this != &other) {
				score = other.score;
				symIdx = other.symIdx;
				symCode = other.symCode;
				*const_cast<bool*>(&unicode) = other.unicode;
				params = other.params;
			}
			return *this;
		}

		void reset() {
			score = numeric_limits<double>::lowest();
			symIdx = UINT_MAX; // no best yet
			symCode = 32UL; // Space
		}

		void reset(double score_, unsigned symIdx_, unsigned long symCode_, const MatchParams &params_) {
			score = score_;
			symIdx = symIdx_;
			symCode = symCode_;
			params = params_;
		}

		static const wstring HEADER;

		friend wostream& operator<<(wostream &os, const BestMatch &bm) {
			unsigned long symCode = bm.symCode;
			if(bm.unicode) {
				if(symCode == (unsigned long)',')
					os<<L"COMMA";
				else if(symCode == (unsigned long)'(')
					os<<L"OPEN_PAR";
				else if(symCode == (unsigned long)')')
					os<<L"CLOSE_PAR";
				else if(os<<(wchar_t)symCode)
					os<<'('<<symCode<<')';
				else {
					os.clear();
					os<<symCode;
				}
			} else
				os<<symCode;

			os<<",\t"<<bm.score<<",\t"<<bm.params;
			return os;
		}
	};

	const wstring BestMatch::HEADER(wstring(L"#GlyphCode,\t#ChosenScore,\t") + MatchParams::HEADER);

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
