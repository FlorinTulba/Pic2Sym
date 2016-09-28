/****************************************************************************************
 The application Pic2Sym approximates images by a
 grid of colored symbols with colored backgrounds.

 This file belongs to the UnitTesting project.

 Copyrights from the libraries used by the program:
 - (c) 2015 Boost (www.boost.org)
   License: <http://www.boost.org/LICENSE_1_0.txt>
            or doc/licenses/Boost.lic
 - (c) 2015 The FreeType Project (www.freetype.org)
   License: <http://git.savannah.gnu.org/cgit/freetype/freetype2.git/plain/docs/FTL.TXT>
	        or doc/licenses/FTL.txt
 - (c) 2015 OpenCV (www.opencv.org)
   License: <http://opencv.org/license.html>
            or doc/licenses/OpenCV.lic
 - (c) 1997-2002 OpenMP Architecture Review Board (www.openmp.org)
   (c) Microsoft Corporation (Visual C++ implementation for OpenMP C/C++ Version 2.0 March 2002)
   See: <https://msdn.microsoft.com/en-us/library/8y6825x5(v=vs.140).aspx>
 
 (c) 2016 Florin Tulba <florintulba@yahoo.com>

 This program is free software: you can use its results,
 redistribute it and/or modify it under the terms of the GNU
 Affero General Public License version 3 as published by the
 Free Software Foundation.

 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 GNU Affero General Public License for more details.

 You should have received a copy of the GNU Affero General Public License
 along with this program ('agpl-3.0.txt').
 If not, see <http://www.gnu.org/licenses/agpl-3.0.txt>.
 ****************************************************************************************/

#define BOOST_TEST_MODULE Tests for Pic2Sym project

#include "testMain.h"

// Including the CPP files allows parsing UNIT_TESTING guarded regions.
// Disadvantage: namespace pollution
#include "misc.cpp"
#include "timing.cpp"
#include "fontEngine.cpp"
#include "matchEngine.cpp"
#include "clusterEngine.cpp"
#include "symData.cpp"
#include "match.cpp"
#include "patch.cpp"
#include "transform.cpp"
#include "controller.cpp"
#include "img.cpp"
#include "matchSettings.cpp"
#include "pixMapSym.cpp"
#include "varConfig.cpp"
#include "presentation.cpp"
#include "matchParams.cpp"
#include "symSettings.cpp"
#include "cachedData.cpp"
#include "structuralSimilarity.cpp"
#include "propsReader.cpp"
#include "transformTrace.cpp"
#include "matchAspectsFactory.cpp"
#include "symFilter.cpp"
#include "filledRectanglesFilter.cpp"
#include "gridBarsFilter.cpp"
#include "bulkySymsFilter.cpp"
#include "unreadableSymsFilter.cpp"
#include "sievesSymsFilter.cpp"
#include "symsSerialization.cpp"
#include "clusterAlg.cpp"
#include "noClustering.cpp"
#include "partitionClustering.cpp"
#include "ttsasClustering.cpp"

#include <ctime>
#include <random>

#include <boost/optional/optional.hpp>

const unsigned SymsBatch_defaultSz = 25U;

#pragma warning(disable:4273) // inconsistent DLL linkage
int __cdecl omp_get_thread_num(void) {
	return 0;
}
#pragma warning(default:4273)

bool checkCancellationRequest() {
	return false;
}

namespace cv {
	// waitKey is called within Controller::symbolsChanged(),
	// but UnitTesting shouldn't depend on highgui, so it defines a dummy waitKey right here.
	int __cdecl waitKey(int) { return 0; }
}

namespace ut {
	bool Controller::initImg = false;
	bool Controller::initFontEngine = false;
	bool Controller::initMatchEngine = false;
	bool Controller::initTransformer = false;
	bool Controller::initComparator = false;
	bool Controller::initControlPanel = false;

	Fixt::Fixt() {
		// reinitialize all these fields
		Controller::initImg = Controller::initFontEngine = Controller::initMatchEngine =
		Controller::initTransformer = Controller::initComparator = Controller::initControlPanel =
			true;
	}

	Fixt::~Fixt() {
	}

	namespace {
		/// Returns the path to 'Pic2Sym.exe'
		boost::optional<const boost::filesystem::path> pathToPic2Sym() {
#ifdef _DEBUG
#		define CONFIG_TYPE "Debug"
#else
#		define CONFIG_TYPE "Release"
#endif
			boost::filesystem::path dirOfPic2Sym(absolute("."));
			if(exists("x64")) { // Solution root is the current folder
				// Pic2Sym.exe is in x64/<CONFIG_TYPE>/ folder
				dirOfPic2Sym.append("x64").append(CONFIG_TYPE);

			} else { // UnitTesting.exe launched directly from its own folder
				// Pic2Sym.exe is in ../ folder
				dirOfPic2Sym = dirOfPic2Sym
					.parent_path()	// parent of '<CONFIG_TYPE>/UnitTesting/.' is '<CONFIG_TYPE>/UnitTesting'
					.parent_path();	// to get to <CONFIG_TYPE> we need one more step
			}
#undef CONFIG_TYPE

			boost::filesystem::path pic2SymPath(boost::filesystem::path(dirOfPic2Sym)
												.append("Pic2Sym.exe"));
			if(!exists(pic2SymPath)) {
				cerr<<"Couldn't locate Pic2Sym.exe"<<endl;
				return boost::none;
			}

			return pic2SymPath;
		}

		/// Starts detached process to view particular issues detected by Unit Testing
		void createVisualizer(const boost::filesystem::path &pic2SymPath,
							  const boost::filesystem::path &dirOfPic2Sym,
							  const wstring &commandLine,
							  const string &testCateg
							  ) {
			PROCESS_INFORMATION pi { 0 }; STARTUPINFO si { 0 };
			if(!CreateProcess(pic2SymPath.wstring().c_str(),
								(LPWSTR)commandLine.c_str(),
								NULL, NULL, FALSE, CREATE_NEW_CONSOLE, NULL,
								dirOfPic2Sym.wstring().c_str(),
								&si, &pi)) {
				DWORD er = GetLastError();

				LPVOID lpMsgBuf;
				FormatMessage(FORMAT_MESSAGE_ALLOCATE_BUFFER |  FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS, nullptr, er, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), (LPTSTR)&lpMsgBuf, 0, nullptr);

				wstring reason((TCHAR*)lpMsgBuf);
				ostringstream oss;
				oss<<"(ErrCode=0x"<<hex<<setfill('0')<<setw(6)<<er<<dec<<") - "<<wstr2str(reason);

				LocalFree(lpMsgBuf);

				cerr<<"Couldn't display the "<<testCateg<<" due to:"<<endl;
				cerr<<oss.str()<<endl;
			}
		}
	} // anonymous namespace

	void showMismatches(const string &testTitle,
						const vector<const BestMatch> &mismatches) {
		if(mismatches.empty())
			return;
		
		const unsigned cases = (unsigned)mismatches.size();
		const auto &firstItem = mismatches.front();
		const auto &firstRefM = firstItem.patch.orig;
		const unsigned tileSz = (unsigned)firstRefM.rows;
		const bool isColor = firstRefM.channels() > 1;
		cerr<<"There were "<<cases<<" unexpected matches"<<endl;

		auto pic2SymPath = pathToPic2Sym();
		if(!pic2SymPath)
			return;

		const auto dirOfPic2Sym = pic2SymPath->parent_path();

		// Ensure there is a folder x64\<ConfigType>\UnitTesting\Mismatches
		boost::filesystem::path mismatchesFolder(dirOfPic2Sym);
		if(!exists(mismatchesFolder.append("UnitTesting").append("Mismatches")))
			create_directory(mismatchesFolder);

		time_t suffix = time(nullptr);
		const string uniqueTestTitle = testTitle + "_" + to_string(suffix);
		wostringstream woss;
		woss<<"Pic2Sym.exe mismatches \""<<uniqueTestTitle<<'"';
		wstring commandLine(woss.str());

		// Tiling the references/results in 2 rectangles with approx 800 x 600 aspect ratio
		const double ar = 800/600.,
			widthD = round(sqrt(cases*ar)), heightD = ceil(cases/widthD);
		const unsigned widthTiles = (unsigned)widthD, heightTiles = (unsigned)heightD;
		const unsigned width = widthTiles*tileSz, height = heightTiles*tileSz;
		Mat m(2*height, width, (isColor?CV_8UC3:CV_8UC1), Scalar::all(127U)), // combined
			mRef(m, Range(0, height)), // upper half contains the references
			mRes(m, Range(height, 2*height)); // lower half for results
		
		// Tile the references & results
		for(unsigned idx = 0U, r = 0U; r<heightTiles && idx<cases; ++r) {
			Range rowRange(r*tileSz, (r+1)*tileSz);
			for(unsigned c = 0U; c<widthTiles && idx<cases; ++c, ++idx) {
				Range colRange(c*tileSz, (c+1)*tileSz);
				const auto &item = mismatches[idx];
				item.patch.orig.copyTo(Mat(mRef, rowRange, colRange)); // reference
				item.bestVariant.approx.copyTo(Mat(mRes, rowRange, colRange)); // result

				// optionally present info about corresponding BestMatch get<2>(item)
			}
		}

		// write the combined matrices as <uniqueTestTitle>.jpg
		boost::filesystem::path destFile(boost::filesystem::path(mismatchesFolder)
										 .append(uniqueTestTitle).concat(".jpg"));
		if(false == imwrite(destFile.string().c_str(), m))
				cerr<<"Couldn't write the image generated by Unit Tests!"<<endl
					<<'['<<destFile<<']'<<endl;

		// start detached process to display a comparator with the references & results
		createVisualizer(pic2SymPath.value(), dirOfPic2Sym, commandLine, "mismatches");
	}

	void showMisfiltered(const string &testTitle,
						 const vector<std::shared_ptr<PixMapSym>> &misfiltered) {
		if(misfiltered.empty())
			return;

		auto pic2SymPath = pathToPic2Sym();
		if(!pic2SymPath)
			return;

		const auto dirOfPic2Sym = pic2SymPath->parent_path();

		// Ensure there is a folder x64\<ConfigType>\UnitTesting\Misfiltered
		boost::filesystem::path misfilteredFolder(dirOfPic2Sym);
		if(!exists(misfilteredFolder.append("UnitTesting").append("Misfiltered")))
			create_directory(misfilteredFolder);

		time_t suffix = time(nullptr);
		string uniqueTestTitle = testTitle + "_" + to_string(suffix); // Append an unique suffix
		for(const char toReplace : string(" :")) // Replace all spaces and colons with underscores
			replace(BOUNDS(uniqueTestTitle), toReplace, '_');
		wostringstream woss;
		woss<<"Pic2Sym.exe misfiltered \""<<uniqueTestTitle<<'"';
		wstring commandLine(woss.str());

		// Construct the image to display, which should contain all miscategorized symbols from one group
		static const Scalar GridColor(255U, 200U, 200U), BgColor(200U, 200U, 255U);
		const int borderRows = 2, vertSplitsCount = 1 + (int)misfiltered.size();
		int rowsMislabeledSyms = 0,
			colsMislabeledSyms = 1,
			tallestSym = 0, vertSplitIdx = 1;
		vector<int> posVertSplits(vertSplitsCount, 0);
		for(const auto &sym : misfiltered) {
			const Mat symAsMat = sym->asNarrowMat(); // narrow versions are now the whole symbols
			if(symAsMat.rows > tallestSym)
				tallestSym = symAsMat.rows;
			colsMislabeledSyms += symAsMat.cols + 1;
			posVertSplits[vertSplitIdx++] = colsMislabeledSyms - 1;
		}
		rowsMislabeledSyms = borderRows + tallestSym;
		Mat mislabeledSyms(rowsMislabeledSyms, colsMislabeledSyms, CV_8UC3, BgColor);
		mislabeledSyms.row(0).setTo(GridColor);
		mislabeledSyms.row(rowsMislabeledSyms - 1).setTo(GridColor);
		for(const int posVertSplit : posVertSplits)
			mislabeledSyms.col(posVertSplit).setTo(GridColor);
		vertSplitIdx = 0;
		for(const auto &sym : misfiltered) {
			const Mat symMat = sym->asNarrowMat(); // narrow versions are now the whole symbols
			const vector<Mat> symChannels(3, symMat);
			Mat symAsIfColor,
				region(mislabeledSyms, Range(1, symMat.rows + 1),
				Range(posVertSplits[vertSplitIdx]+1, posVertSplits[vertSplitIdx+1]));
			merge(symChannels, symAsIfColor);
			symAsIfColor.copyTo(region);
			++vertSplitIdx;
		}

		// write the misfiltered symbols as <uniqueTestTitle>.jpg
		boost::filesystem::path destFile(boost::filesystem::path(misfilteredFolder)
										 .append(uniqueTestTitle).concat(".jpg"));
		if(false == imwrite(destFile.string().c_str(), mislabeledSyms))
			cerr<<"Couldn't write the image generated by Unit Tests!"<<endl
			<<'['<<destFile<<']'<<endl;

		createVisualizer(pic2SymPath.value(), dirOfPic2Sym, commandLine, "misfiltered symbols");
	}

	unsigned randUnifUint() {
		static random_device rd;
		static mt19937 gen(rd());
		static uniform_int_distribution<unsigned> uid;
		return uid(gen);
	}

	unsigned char randUnsignedChar(unsigned char minIncl/* = 0U*/, unsigned char maxIncl/* = 255U*/) {
		return (unsigned char)(minIncl + randUnifUint()%((unsigned)(maxIncl - minIncl) + 1U));
	}
}

MatchSettings::MatchSettings() {}

#define GET_FIELD(FieldType, ...) \
	static std::shared_ptr<FieldType> pField; \
	if(ut::Controller::init##FieldType || !pField) { \
		pField = std::make_shared<FieldType>(__VA_ARGS__); \
		ut::Controller::init##FieldType = false; \
	} \
	return *pField

Img& Controller::getImg() {
	GET_FIELD(Img, nullptr); // Here's useful the hack mentioned at Img's constructor declaration
}

Comparator& Controller::getComparator() {
	GET_FIELD(Comparator, nullptr); // Here's useful the hack mentioned at Comparator's constructor declaration
}

FontEngine& Controller::getFontEngine(const SymSettings &ss_) const {
	GET_FIELD(FontEngine, *this, ss_);
}

MatchEngine& Controller::getMatchEngine(const Settings &cfg_) const {
	GET_FIELD(MatchEngine, cfg_, getFontEngine(cfg_.ss));
}

Transformer& Controller::getTransformer(const Settings &cfg_) const {
	GET_FIELD(Transformer, *this, cfg_, getMatchEngine(cfg_), getImg());
}

ControlPanel& Controller::getControlPanel(Settings &cfg_) {
	GET_FIELD(ControlPanel, *this, cfg_);
}

#undef GET_FIELD

Controller::~Controller() {}

void Controller::handleRequests() {}

void Controller::hourGlass(double, const string&) const {}

void Controller::reportGlyphProgress(double) const {}

void Controller::updateSymsDone(double) const {}

void Controller::reportTransformationProgress(double, bool) const {}

void Controller::presentTransformationResults(double) const {}

bool Controller::newImage(const Mat &imgMat) {
	bool result = img.reset(imgMat);

	if(result) {
		cout<<"Using Matrix instead of a "<<(img.isColor()?"color":"grayscale")<<" image"<<endl;
		if(!imageOk)
			imageOk = true;

		// For valid matrices of size sz x sz, ignore MIN_H_SYMS & MIN_V_SYMS =>
		// Testing an image containing a single patch
		if(imgMat.cols == cfg.ss.getFontSz() && imgMat.rows == cfg.ss.getFontSz()) {
			if(1U != cfg.is.getMaxHSyms())
				cfg.is.setMaxHSyms(1U);
			if(1U != cfg.is.getMaxVSyms())
				cfg.is.setMaxVSyms(1U);
		}
	}

	return result;
}

const SymData* Controller::pointedSymbol(int x, int y) const { return nullptr; }

void Controller::displaySymCode(unsigned long symCode) const {}

void Controller::enlistSymbolForInvestigation(const SymData &sd) const {}

void Controller::symbolsReadyToInvestigate() const {}

TaskMonitor::TaskMonitor(const string&, AbsJobMonitor&) : AbsTaskMonitor("") {}

void TaskMonitor::setTotalSteps(size_t) {}

void TaskMonitor::taskAdvanced(size_t/* = 1U*/) {}

void TaskMonitor::taskDone() {}

void TaskMonitor::taskAborted() {}

JobMonitor::JobMonitor(...) : AbsJobMonitor("") {}

unsigned JobMonitor::monitorNewTask(AbsTaskMonitor &newActivity) { return 0U; }

void JobMonitor::setTasksDetails(const vector<double>&) {}

void JobMonitor::taskAdvanced(double, unsigned) {}

void JobMonitor::taskDone(unsigned) {}

void JobMonitor::taskAborted(unsigned) {}

SymData::SymData(unsigned long code_, double minVal_, double diffMinMax_, double pixelSum_,
				 const Point2d &mc_, const SymData::IdxMatMap &relevantMats) :
				 code(code_), minVal(minVal_), diffMinMax(diffMinMax_),
				 pixelSum(pixelSum_), mc(mc_),
				 symAndMasks(SymData::MatArray { { Mat(), Mat(), Mat(), Mat(), Mat(), Mat(), Mat() } }) {
	for(const auto &idxAndMat : relevantMats)
		const_cast<Mat&>(symAndMasks[idxAndMat.first]) = idxAndMat.second;
}

PixMapSym::PixMapSym(const vector<unsigned char> &data, const Mat &consec, const Mat &revConsec) : 
		pixels(data) {
	const unsigned sz = (unsigned)consec.cols;
	assert(sz == (unsigned)revConsec.rows);
	assert(sz*sz == (unsigned)data.size());

	rows = cols = (unsigned char)sz;
	top = (unsigned char)(rows - 1U);

	computeMcAndGlyphSum(sz, data, rows, cols, 0U, top,
						 consec, revConsec,
						 mc, glyphSum,
						 &colSums, &rowSums);
}

static const Mat blurredVersionOf(const Mat &orig_) {
	Mat blurred;
	extern const Size BlurWinSize;
	extern const double BlurStandardDeviation;
	GaussianBlur(orig_, blurred, BlurWinSize, BlurStandardDeviation, BlurStandardDeviation, BORDER_REPLICATE);
	return blurred;
}

Patch::Patch(const Mat &orig_) : Patch(orig_, blurredVersionOf(orig_), orig_.channels()>1) {}

