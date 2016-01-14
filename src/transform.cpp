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
#include <numeric>

#ifdef _DEBUG
#include <fstream>
#endif

using namespace std;
using namespace cv;
using namespace boost::filesystem;

namespace {
	// Holds relevant data during patch&glyph matching
	struct MatchParams {
		Point2d cogPatch;			// center of gravity of patch
		Point2d cogGlyph;			// center of gravity of glyph
		double glyphWeight;			// % of the box covered by the glyph (0..1)
		double miuFg, miuBg;		// average color for fg / bg (range 0..255)

		// average squared error for fg / bg
		// subrange of 0..127.5^2; best when 0
		double aseFg, aseBg;

#ifdef _DEBUG
		static const wstring HEADER;

		friend wostream& operator<<(wostream &os, const MatchParams &mp) {
			os<<mp.cogGlyph.x<<",\t"<<mp.cogGlyph.y<<",\t"<<mp.cogPatch.x<<",\t"<<mp.cogPatch.y<<",\t"
				<<mp.miuFg<<",\t"<<mp.miuBg<<",\t"<<mp.aseFg<<",\t"<<mp.aseBg<<",\t"<<mp.glyphWeight;
			return os;
		}
#endif // _DEBUG
	};

#ifdef _DEBUG
	const wstring MatchParams::HEADER(L"#cogFgX,\t#cogFgY,\t#cogBgX,\t#cogBgY,\t"
									 L"#miuFg,\t#miuBg,\t#aseFg,\t#aseBg,\t#fg/all");
#endif // _DEBUG

	class Matcher {
		const unsigned fontSz_1;		// size of the font - 1
		const double smallGlyphsCoverage; // max ratio of glyph area / containing area for small chars
		const double PREFERRED_RADIUS;	// allowed distance between the cog-s of patch & chosen glyph
		const double MAX_COG_OFFSET;	// max distance possible between the cog-s of patch & chosen glyph
		// happens for the extremes of a diagonal

		const Point2d centerPatch;		// center of the patch

	public:
		MatchParams params;

		Matcher(unsigned fontSz, double smallGlyphsCoverage_) :
			fontSz_1(fontSz - 1U), smallGlyphsCoverage(smallGlyphsCoverage_),
			PREFERRED_RADIUS(3U * fontSz / 8.), MAX_COG_OFFSET(sqrt(2) * fontSz_1),
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

		+ 3. have enough contrast between foreground & background of the selected glyph

		+ 4. use largest possible matching glyph

		B) Together, the patches should also preserve region's gradients:
		* 1. Prefer glyphs that respect the gradient of their corresponding patch.

		+ 2. Consider the gradient within a slightly extended patch
		Gain: Smoothness;		Drawback: Complexity


		Points A1&2 minimizes the standard deviation (or a similar measure) on each region.
		Point A3 just penalizes small differences between the means of the fg & bg.
		Point A4 favors larger glyphs.
		Point B1 ensures the weight centers of the glyph and patch are close to each other
		as distance and as direction.

		The remaining points might be addressed in a future version.
		*/
		double score(const Config &cfg) const {
			// for a histogram with just 2 equally large bins on 0 and 255 => mean = sdev = 127.5
			static const double SDEV_MAX = 255/2.;
			static const double SQRT2 = sqrt(2), TWO_SQRT2 = 2. - SQRT2;
			static const double MIN_CONTRAST_BRIGHT = 2., // less contrast needed for bright tones
								MIN_CONTRAST_DARK = 5.; // more contrast needed for dark tones
			static const Point2d ORIGIN;

			/////////////// CORRECTNESS FACTORS (Best Matching) ///////////////
			// range 0..1, acting just as penalty for bad standard deviations
			register const double fSdevFg = 1. - sqrt(params.aseFg) / SDEV_MAX;
			register const double fSdevBg = 1. - sqrt(params.aseBg) / SDEV_MAX;

			/////////////// SMOOTHNESS FACTORS (Similar gradient) ///////////////
			// best glyph location is when cog-s are near to each other
			// range 0 .. 1.42*fontSz_1, best when 0
			const double cogOffset = norm(params.cogPatch - params.cogGlyph);
			// 0.266 .. 1 for cogOffset >= PREFERRED_RADIUS;    1 .. 1.266 for less
			register const double fMinimalCogOffset =
				1. + (PREFERRED_RADIUS - cogOffset) / MAX_COG_OFFSET;

			const Point2d relCogPatch = params.cogPatch - centerPatch;
			const Point2d relCogGlyph = params.cogGlyph - centerPatch;

			// best gradient orientation when angle between cog-s is 0 => cos = 1
			// Maintaining the cosine of the angle is ok, as it stays near 1 for small angles.
			// -1..1 range, best when 1
			double cosAngleCogs = 0.;
			if(relCogGlyph != ORIGIN && relCogPatch != ORIGIN) // avoid DivBy0
				cosAngleCogs = relCogGlyph.dot(relCogPatch) /
								(norm(relCogGlyph) * norm(relCogPatch));

			// 0..1 for |cogAngle| >= 45;   >1 for |cogAngle| < 45
			// max 1.17 for cogAngle == 0
			register const double fCogAngleLessThan45 = (1. + cosAngleCogs) * TWO_SQRT2;

			/////////////// FANCINESS FACTORS (Larger glyphs & contrast) ///////////////
			// range 0 .. 255, best when large; less important than the other factors
			const double contrast = abs(params.miuBg - params.miuFg);
			// just penalize severely low contrast for the average glyph brightness
			double minimalContrast =
				MIN_CONTRAST_BRIGHT + (MIN_CONTRAST_DARK - MIN_CONTRAST_BRIGHT) *
				(params.miuFg + params.miuBg) * .5;
			register double fMinimalContrast = contrast;
			if(contrast > minimalContrast)
				fMinimalContrast = minimalContrast;
			fMinimalContrast /= minimalContrast;

			// <=1 for glyphs considered small;   >1 otherwise
			register const double fGlyphWeight = params.glyphWeight + 1. - smallGlyphsCoverage;

			double result =
				/////////////// CORRECTNESS FACTORS (Best Matching) ///////////////
				// closer to 1 for good sdev of fg;  tends to 0 otherwise
				pow(fSdevFg, cfg.get_kSdevFg())

				// closer to 1 for good sdev of bg;  tends to 0 otherwise
				* pow(fSdevBg, cfg.get_kSdevBg())

				/////////////// SMOOTHNESS FACTORS (Similar gradient) ///////////////
				// <=1 for cogOffset >= PREFERRED_RADIUS;  >1 otherwise
				* pow(fMinimalCogOffset, cfg.get_kCogOffset())

				// <=1 for |angleCogs| >= 45;  >1 otherwise
				// lessen the importance for small cogOffset-s (< PREFERRED_RADIUS)
				* pow(fCogAngleLessThan45,
					cfg.get_kCosAngleCogs() *
						min(cogOffset, PREFERRED_RADIUS)/
							PREFERRED_RADIUS)

				/////////////// FANCINESS FACTORS (Larger glyphs & contrast) ///////////////
				// <1 for poor contrast; 1 otherwise
				* pow(fMinimalContrast, cfg.get_kContrast())

				// <=1 for glyphs considered small;  >1 otherwise
				* pow(fGlyphWeight, cfg.get_kGlyphWeight());

			return result;
		}
	};

	// Holds the best grayscale match found at a given time
	struct BestMatch {
		double score;			// score of the best
		unsigned charIdx;		// index within vector<PixMapChar>
		unsigned long charCode;	// glyph code
		const bool unicode;		// is the charmap in unicode

		MatchParams params;	// parameters of the match for the best glyph

		BestMatch(bool isUnicode = true) : unicode(isUnicode) { reset(); }

		void reset() {
			score = numeric_limits<double>::lowest();
			charIdx = UINT_MAX; // no best yet
			charCode = 32; // Space
		}

#ifdef _DEBUG
		static const wstring HEADER;

		friend wostream& operator<<(wostream &os, const BestMatch &bm) {
			unsigned long chCode = bm.charCode;
			if(bm.unicode) {
				if(chCode == (unsigned long)',')
					os<<L"COMMA";
				else if(chCode == (unsigned long)'(')
					os<<L"OPEN_PAR";
				else if(chCode == (unsigned long)')')
					os<<L"CLOSE_PAR";
				else if(os<<(wchar_t)chCode)
					os<<'('<<chCode<<')';
				else {
					os.clear();
					os<<chCode;
				}
			} else
				os<<chCode;

			os<<",\t"<<bm.score<<",\t"<<bm.params;
			return os;
		}
#endif // _DEBUG
	};

#ifdef _DEBUG
	const wstring BestMatch::HEADER(wstring(L"#GlyphCode,\t#ChosenScore,\t") + MatchParams::HEADER);
#endif // _DEBUG

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
	
	void findBestMatch(const Config &cfg, const vector<pair<Mat, Mat>> &charset,
					   const Mat &patch, Matcher &matcher, BestMatch &best,
					   vector<PixMapChar>::const_iterator itFeBegin,
					   const double sz2, const Mat &consec) {
		best.reset();

		double patchSum = *sum(patch).val;
		Mat glyph, negGlyph, temp, temp1;
		reduce(patch, temp, 0, CV_REDUCE_SUM);	// sum all rows
		reduce(patch, temp1, 1, CV_REDUCE_SUM);	// sum all columns

		MatchParams &mp = matcher.params;
		mp.cogPatch = Point2d(temp.dot(consec), temp1.t().dot(consec)) / patchSum; // center of gravity

		auto itFe = itFeBegin;
		for(auto &glyphAndNegative : charset) {
			tie(glyph, negGlyph) = glyphAndNegative;

			double glyphSum = itFe->glyphSum,
				negGlyphSum = itFe->negGlyphSum;

			mp.glyphWeight = glyphSum / sz2;

			// Computing foreground and background means plus the average of squared errors.
			// 'mean' & 'meanStdDev' provided by OpenCV have unfortunately binary masking.
			// Our glyphs have gradual transitions between foreground & background.
			// These transition need less than the full weighting provided by OpenCV.

			// Implemented below edge weighting:

			// First the means:
			double dotP = patch.dot(glyph);
			mp.miuFg = dotP / glyphSum; // 'mean' would divide by more (countNonZero)
			mp.miuBg = (patchSum - dotP) / negGlyphSum; // 'mean' would divide by more (countLessThan1)

			// Now the average of squared errors (cheaper to compute than standard deviations):
			// 'meanStdDev' would subtract from patch elements different means
			// and divide by the larger values described above
			temp = (patch-mp.miuFg).mul(glyph); // Elem-wise (mask only fg, weigh contours)
			mp.aseFg = temp.dot(temp) / glyphSum;

			temp = (patch-mp.miuBg).mul(negGlyph); // Elem-wise (mask only bg, weigh contours)
			mp.aseBg = temp.dot(temp) / negGlyphSum;

			// Obtaining center of gravity for glyph
			mp.cogGlyph = (mp.miuFg * itFe->cogFg + mp.miuBg * itFe->cogBg) /
				(mp.miuFg + mp.miuBg);

			double score = matcher.score(cfg);
			if(score > best.score) {
				best.score = score;
				best.charCode = itFe->chCode;
				best.charIdx = (unsigned)distance(itFeBegin, itFe);
				best.params = mp;
			}

			++itFe;
		}
	}

	void commitMatch(const Config &cfg, const vector<pair<Mat, Mat>> &charset,
					 const BestMatch &best, Mat &result,
					 const Mat &resized, const Mat &patch, unsigned r, unsigned c,
					 bool isColor, vector<PixMapChar>::const_iterator itFeBegin) {
		auto sz = cfg.getFontSz();
		Mat patchResult(result, Range(r, r+sz), Range(c, c+sz)), glyph, negGlyph;
		tie(glyph, negGlyph) = *next(charset.begin(), best.charIdx);
		if(isColor) {
			Mat patchColor(resized, Range(r, r+sz), Range(c, c+sz));
			auto itFeBest = next(itFeBegin, best.charIdx);
			double glyphBestSum = itFeBest->glyphSum,
				negGlyphBestSum = itFeBest->negGlyphSum;

			vector<Mat> channels;
			split(patchColor, channels);
			assert(channels.size() == 3);

			double diffFgBg = 0.;
			for(auto &ch : channels) {
				ch.convertTo(ch, CV_64FC1); // processing double values

				double miuFg = ch.dot(glyph) / glyphBestSum,
					miuBg = ch.dot(negGlyph) / negGlyphBestSum,
					newDiff = miuFg - miuBg;

				glyph.convertTo(ch, CV_8UC1, newDiff, miuBg);

				diffFgBg += abs(newDiff);
			}

			if(diffFgBg < 3.*cfg.getBlankThreshold())
				patchResult = mean(patchColor);
			else
				merge(channels, patchResult);

		} else { // grayscale result
			if(abs(best.params.miuFg - best.params.miuBg) < cfg.getBlankThreshold())
				patchResult = mean(patch);
			else
				glyph.convertTo(patchResult, CV_8UC1,
								best.params.miuFg - best.params.miuBg,
								best.params.miuBg);
		}
	}
} // anonymous namespace

Transformer::Transformer(Config &cfg_) : cfg(cfg_) {
	// Ensure there is an Output folder
	path outputFolder = cfg.getWorkDir();
	if(!exists(outputFolder.append("Output")))
	   create_directory(outputFolder);
}

void Transformer::reconfig() {
	string initCharsetId = fe.fontId(); // remember initial charset

	newSettings = cfg.update(); // Allow setting new parameters for the transformation
	selectFont(fe, cfg); // Configure font set

	if(fe.fontId() != initCharsetId) {
		newSettings = true;

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
	auto oldImg = img.name();
	selectImage(img);

	ostringstream oss; oss<<img.name()<<'_'<<fe.fontId(); // no extension yet
	const string studiedCase = oss.str(); // id included in the result & trace file names

	path resultFile(cfg.getWorkDir());
	resultFile.append("Output").append(studiedCase).
		concat(".bmp"); // generating a BMP result file

	oss.str(""); oss.clear();
	oss<<resultFile; // contains also the double quotes needed when the path contains Spaces
	string quotedResultFile(oss.str());

	if(img.name().compare(oldImg) == 0 && !newSettings) {
		cout<<"Image already processed under these settings."<<endl;
		system(quotedResultFile.c_str());
		return;
	}

#ifdef _DEBUG
	path traceFile(cfg.getWorkDir());
	traceFile.append("data_").concat(studiedCase).
		concat(".csv"); // generating a CSV trace file
	wofstream ofs(traceFile.c_str());
	ofs<<"#Row,\t#Col,\t"<<BestMatch::HEADER<<endl;
#endif

	unsigned sz = cfg.getFontSz();
	const double sz2 = (double)sz*sz;

	Mat consec(1, sz, CV_64FC1);
	iota(consec.begin<double>(), consec.end<double>(), 0.);

	Mat temp, temp1, gray;
	const Mat resized = img.resized(cfg, &gray);
	gray.convertTo(gray, CV_64FC1);

	Matcher matcher(sz, fe.smallGlyphsCoverage());
	MatchParams &mp = matcher.params;
	BestMatch best(fe.getEncoding().compare("UNICODE") == 0); // holds the best grayscale match found at a given time
	Mat result(resized.rows, resized.cols, resized.type());
	auto itFeBegin = fe.charset().cbegin();
	for(unsigned r = 0U, h = (unsigned)gray.rows; r<h; r += sz) {
		// Reporting progress
		printf("%6.2f%%\r", r*100./h); // simpler to format percent values with printf

		for(unsigned c = 0U, w = (unsigned)gray.cols; c<w; c += sz) {
			Mat patch(gray, Range(r, r+sz), Range(c, c+sz));

			findBestMatch(cfg, charset, patch, matcher, best, itFeBegin, sz2, consec);

#ifdef _DEBUG
			ofs<<r/sz<<",\t"<<c/sz<<",\t"<<best<<endl;
#endif

			commitMatch(cfg, charset, best, result, resized, patch, r, c, img.isColor(), itFeBegin);
		}
	}

#ifdef _DEBUG
	// Flushing and closing the trace file, to be also ready when inspecting the resulted image
	ofs.close();
#endif

	newSettings = false;

	cout<<"Writing result to "<<resultFile<<endl<<endl;
	imwrite(resultFile.string(), result);
	system(quotedResultFile.c_str()); // inspecting the resulted image
}
