/******************************************************************************
 The application Pic2Sym approximates images by a
 grid of colored symbols with colored backgrounds.

 Copyrights from the libraries used by the program:
 - (c) 2003 Boost (www.boost.org)
     License: doc/licenses/Boost.lic
     http://www.boost.org/LICENSE_1_0.txt
 - (c) 2015-2016 OpenCV (www.opencv.org)
     License: doc/licenses/OpenCV.lic
     http://opencv.org/license/
 - (c) 1996-2002, 2006 The FreeType Project (www.freetype.org)
     License: doc/licenses/FTL.txt
     http://git.savannah.gnu.org/cgit/freetype/freetype2.git/plain/docs/FTL.TXT
 - (c) 1997-2002 OpenMP Architecture Review Board (www.openmp.org)
   (c) Microsoft Corporation (implementation for OpenMP C/C++ v2.0 March 2002)
     See: https://msdn.microsoft.com/en-us/library/8y6825x5.aspx
 - (c) 1995-2017 zlib software (Jean-loup Gailly and Mark Adler - www.zlib.net)
     License: doc/licenses/zlib.lic
     http://www.zlib.net/zlib_license.html


 (c) 2016-2019 Florin Tulba <florintulba@yahoo.com>

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
 If not, see: http://www.gnu.org/licenses/agpl-3.0.txt .
 *****************************************************************************/

#include "precompiled.h"

#define BOOST_TEST_MODULE Tests for project Pic2Sym

#include "clusterEngine.h"
#include "clusterSupport.h"
#include "cmapPerspective.h"
#include "controlPanelActions.h"
#include "controller.h"
#include "fontEngine.h"
#include "glyphsProgressTracker.h"
#include "img.h"
#include "imgSettingsBase.h"
#include "jobMonitorBase.h"
#include "matchEngine.h"
#include "matchSettings.h"
#include "matchSupport.h"
#include "misc.h"
#include "patch.h"
#include "picTransformProgressTracker.h"
#include "pixMapSym.h"
#include "presentCmapBase.h"
#include "resizedImgBase.h"
#include "selectBranch.h"
#include "selectSymbols.h"
#include "settingsBase.h"
#include "symSettingsBase.h"
#include "symbolsSupport.h"
#include "testMain.h"
#include "transform.h"
#include "transformSupport.h"
#include "updateSymSettingsBase.h"
#include "views.h"
#include "warnings.h"

#pragma warning(push, 0)

#include <ctime>
#include <filesystem>
#include <fstream>
#include <optional>
#include <random>

#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#pragma warning(pop)

using namespace std;
using namespace cv;
using namespace boost;
using namespace std::filesystem;

#pragma warning(disable : WARN_INCONSISTENT_DLL_LINKAGE)
int __cdecl omp_get_thread_num(void) {
  return 0;
}
#pragma warning(default : WARN_INCONSISTENT_DLL_LINKAGE)

bool checkCancellationRequest() noexcept {
  return false;
}

namespace cv {
// waitKey is called within Controller::symbolsChanged(),
// but UnitTesting shouldn't depend on highgui, so it defines a dummy waitKey
// right here.
int __cdecl waitKey(int) {
  return 0;
}
}  // namespace cv

namespace ut {
Fixt::Fixt() noexcept {
  // reinitialize all these fields
  Controller::initImg = Controller::initFontEngine =
      Controller::initMatchEngine = Controller::initTransformer =
          Controller::initPreselManager = Controller::initComparator =
              Controller::initControlPanel = true;
}

namespace {
/// Returns the path to 'Pic2Sym.exe'
std::optional<const std::filesystem::path> pathToPic2Sym() {
#ifdef _DEBUG
#define CONFIG_TYPE "Debug"
#else  // _DEBUG not defined
#define CONFIG_TYPE "Release"
#endif  // _DEBUG
  std::filesystem::path dirOfPic2Sym(absolute("."));
  if (exists("x64")) {  // Solution root is the current folder
    // Pic2Sym.exe is in x64/<CONFIG_TYPE>/ folder
    dirOfPic2Sym.append("x64").append(CONFIG_TYPE);

  } else {  // UnitTesting.exe launched directly from its own folder
    // Pic2Sym.exe is in ../ folder
    dirOfPic2Sym =
        dirOfPic2Sym
            .parent_path()   // parent of '<CONFIG_TYPE>/UnitTesting/.' is
                             // '<CONFIG_TYPE>/UnitTesting'
            .parent_path();  // to get to <CONFIG_TYPE> we need one more step
  }
#undef CONFIG_TYPE

  std::filesystem::path pic2SymPath(
      std::filesystem::path(dirOfPic2Sym).append("Pic2Sym.exe"));
  if (!exists(pic2SymPath)) {
    cerr << "Couldn't locate Pic2Sym.exe" << endl;
    return std::nullopt;
  }

  return pic2SymPath;
}

/**
Appends a new issues visualization request to 'issues.bat'.
That file can be launched at the end of the unit tests.
It is located in dirOfPic2Sym = '$(TargetDir)..' when considering the
UnitTesting project.
*/
void createVisualizer(const std::filesystem::path& dirOfPic2Sym,
                      const string& commandLine,
                      const string& testCateg) {
  ofstream ofs(absolute(dirOfPic2Sym).append("issues.bat").string(), ios::app);
  if (!ofs) {
    cerr << "Couldn't open `issues.bat` in append mode for registering the "
         << testCateg << endl;
    return;
  }

  ofs << "start " << commandLine << endl;
  if (!ofs)
    cerr << "Couldn't register the " << testCateg << endl;
}
}  // anonymous namespace

void showMismatches(const string& testTitle,
                    const vector<unique_ptr<BestMatch>>& mismatches) {
  if (mismatches.empty())
    return;

  const unsigned cases = (unsigned)mismatches.size();
  const auto& firstItem = *mismatches.front();
  const auto& firstRefM = firstItem.getPatch().getOrig();
  const unsigned tileSz = (unsigned)firstRefM.rows;
  const bool isColor = firstRefM.channels() > 1;
  cerr << "There were " << cases << " unexpected matches" << endl;

  auto pic2SymPath = pathToPic2Sym();
  if (!pic2SymPath)
    return;

  const auto dirOfPic2Sym = pic2SymPath->parent_path();

  // Ensure there is a folder x64\<ConfigType>\UnitTesting\Mismatches
  std::filesystem::path mismatchesFolder(dirOfPic2Sym);
  if (!exists(mismatchesFolder.append("UnitTesting").append("Mismatches")))
    create_directory(mismatchesFolder);

  time_t suffix = time(nullptr);
  const string uniqueTestTitle = testTitle + "_" + to_string(suffix);
  ostringstream oss;
  oss << "Pic2Sym.exe mismatches \"" << uniqueTestTitle << '"';
  string commandLine(oss.str());

  // Tiling the references/results in 2 rectangles with approx 800 x 600 aspect
  // ratio
  static constexpr double ar = 800 / 600.;

  const double widthD = round(sqrt(cases * ar)), heightD = ceil(cases / widthD);
  const unsigned widthTiles = (unsigned)widthD, heightTiles = (unsigned)heightD;
  const unsigned width = widthTiles * tileSz, height = heightTiles * tileSz;
  Mat m(2 * height, width, (isColor ? CV_8UC3 : CV_8UC1),
        Scalar::all(127U)),                // combined
      mRef(m, Range(0, height)),           // upper half contains the references
      mRes(m, Range(height, 2 * height));  // lower half for results

  // Tile the references & results
  for (unsigned idx = 0U, r = 0U; r < heightTiles && idx < cases; ++r) {
    Range rowRange(r * tileSz, (r + 1) * tileSz);
    for (unsigned c = 0U; c < widthTiles && idx < cases; ++c, ++idx) {
      Range colRange(c * tileSz, (c + 1) * tileSz);
      const auto& item = *mismatches[idx];
      item.getPatch().getOrig().copyTo(
          Mat(mRef, rowRange, colRange));                      // reference
      item.getApprox().copyTo(Mat(mRes, rowRange, colRange));  // result
    }
  }

  // write the combined matrices as <uniqueTestTitle>.jpg
  std::filesystem::path destFile(std::filesystem::path(mismatchesFolder)
                                     .append(uniqueTestTitle)
                                     .concat(".jpg"));
  if (false == imwrite(destFile.string().c_str(), m))
    cerr << "Couldn't write the image generated by Unit Tests!\n"
         << '[' << destFile << ']' << endl;

  // start detached process to display a comparator with the references &
  // results
  createVisualizer(dirOfPic2Sym, commandLine, "mismatches");
}

void showMisfiltered(
    const string& testTitle,
    const vector<std::unique_ptr<const IPixMapSym>>& misfiltered) {
  if (misfiltered.empty())
    return;

  auto pic2SymPath = pathToPic2Sym();
  if (!pic2SymPath)
    return;

  const auto dirOfPic2Sym = pic2SymPath->parent_path();

  // Ensure there is a folder x64\<ConfigType>\UnitTesting\Misfiltered
  std::filesystem::path misfilteredFolder(dirOfPic2Sym);
  if (!exists(misfilteredFolder.append("UnitTesting").append("Misfiltered")))
    create_directory(misfilteredFolder);

  time_t suffix = time(nullptr);
  string uniqueTestTitle =
      testTitle + "_" + to_string(suffix);  // Append an unique suffix
  for (const char toReplace :
       string(" :"))  // Replace all spaces and colons with underscores
    replace(BOUNDS(uniqueTestTitle), toReplace, '_');
  ostringstream oss;
  oss << "Pic2Sym.exe misfiltered \"" << uniqueTestTitle << '"';
  string commandLine(oss.str());

  // Construct the image to display, which should contain all miscategorized
  // symbols from one group
  static const Scalar GridColor(255U, 200U, 200U), BgColor(200U, 200U, 255U);

  const int borderRows = 2, vertSplitsCount = 1 + (int)misfiltered.size();
  int rowsMislabeledSyms = 0, colsMislabeledSyms = 1, tallestSym = 0,
      vertSplitIdx = 1;
  vector<int> posVertSplits(vertSplitsCount, 0);
  for (const auto& sym : misfiltered) {
    const Mat symAsMat =
        sym->asNarrowMat();  // narrow versions are now the whole symbols
    if (symAsMat.rows > tallestSym)
      tallestSym = symAsMat.rows;
    colsMislabeledSyms += symAsMat.cols + 1;
    posVertSplits[vertSplitIdx++] = colsMislabeledSyms - 1;
  }
  rowsMislabeledSyms = borderRows + tallestSym;
  Mat mislabeledSyms(rowsMislabeledSyms, colsMislabeledSyms, CV_8UC3, BgColor);
  mislabeledSyms.row(0).setTo(GridColor);
  mislabeledSyms.row(rowsMislabeledSyms - 1).setTo(GridColor);
  for (const int posVertSplit : posVertSplits)
    mislabeledSyms.col(posVertSplit).setTo(GridColor);
  vertSplitIdx = 0;
  for (const auto& sym : misfiltered) {
    const Mat symMat =
        sym->asNarrowMat();  // narrow versions are now the whole symbols
    const vector<Mat> symChannels(3, symMat);
    Mat symAsIfColor, region(mislabeledSyms, Range(1, symMat.rows + 1),
                             Range(posVertSplits[vertSplitIdx] + 1,
                                   posVertSplits[vertSplitIdx + 1ULL]));
    merge(symChannels, symAsIfColor);
    symAsIfColor.copyTo(region);
    ++vertSplitIdx;
  }

  // write the misfiltered symbols as <uniqueTestTitle>.jpg
  std::filesystem::path destFile(std::filesystem::path(misfilteredFolder)
                                     .append(uniqueTestTitle)
                                     .concat(".jpg"));
  if (false == imwrite(destFile.string().c_str(), mislabeledSyms))
    cerr << "Couldn't write the image generated by Unit Tests!\n"
         << '[' << destFile << ']' << endl;

  createVisualizer(dirOfPic2Sym, commandLine, "misfiltered symbols");
}

unsigned randUnifUint() {
  static random_device rd;
  static mt19937 gen(rd());
  static uniform_int_distribution<unsigned> uid;

  return uid(gen);
}

unsigned char randUnsignedChar(unsigned char minIncl /* = 0U*/,
                               unsigned char maxIncl /* = 255U*/) {
  return (unsigned char)(minIncl +
                         randUnifUint() % ((unsigned)(maxIncl - minIncl) + 1U));
}
}  // namespace ut

MatchSettings::MatchSettings() {}

#define GET_FIELD(FieldType, ...)                      \
  static std::unique_ptr<FieldType> pField;            \
  if (ut::Controller::init##FieldType || !pField) {    \
    pField = std::make_unique<FieldType>(__VA_ARGS__); \
    ut::Controller::init##FieldType = false;           \
  }                                                    \
  return *pField

Img& ControlPanelActions::getImg() noexcept {
  GET_FIELD(Img);
}

IControlPanel& ControlPanelActions::getControlPanel(
    const ISettingsRW& cfg_) noexcept {
  GET_FIELD(ControlPanel, *this, cfg_);
}

bool ControlPanelActions::newImage(const Mat& imgMat) noexcept {
  bool result = img.reset(imgMat);

  if (result) {
    cout << "Using Matrix instead of a "
         << (img.isColor() ? "color" : "grayscale") << " image" << endl;
    if (!imageOk)
      imageOk = true;

    // For valid matrices of size sz x sz, ignore MIN_H_SYMS & MIN_V_SYMS =>
    // Testing an image containing a single patch
    if ((unsigned)imgMat.cols == cfg.getSS().getFontSz() &&
        (unsigned)imgMat.rows == cfg.getSS().getFontSz()) {
      if (1U != cfg.getIS().getMaxHSyms())
        cfg.refIS().setMaxHSyms(1U);
      if (1U != cfg.getIS().getMaxVSyms())
        cfg.refIS().setMaxVSyms(1U);
    }
  }

  return result;
}

IComparator& Controller::getComparator() noexcept {
  GET_FIELD(Comparator);
}

IFontEngine& Controller::getFontEngine(const ISymSettings& ss_) noexcept {
  GET_FIELD(FontEngine, *this, ss_);
}

IMatchEngine& Controller::getMatchEngine(const ISettings& cfg_) noexcept {
  GET_FIELD(MatchEngine, cfg_, getFontEngine(cfg_.getSS()), *cmP);
}

ITransformer& Controller::getTransformer(const ISettings& cfg_) noexcept {
  GET_FIELD(Transformer, *this, cfg_, getMatchEngine(cfg_),
            ControlPanelActions::getImg());
}

#undef GET_FIELD

Controller::~Controller() {}

void Controller::handleRequests() noexcept {}

void Controller::hourGlass(double, const string&, bool) const {}

void PicTransformProgressTracker::transformFailedToStart() noexcept {}

void PicTransformProgressTracker::reportTransformationProgress(double,
                                                               bool) const
    noexcept {}

void PicTransformProgressTracker::presentTransformationResults(double) const
    noexcept {}

void GlyphsProgressTracker::updateSymsDone(double) const noexcept {}

void Controller::updateStatusBarCmapInspect(unsigned,
                                            const string&,
                                            bool) const {}

void Controller::reportDuration(const string&, double) const {}

bool Controller::updateResizedImg(const IResizedImg&) noexcept {
  return true;
}

void Controller::showResultedImage(double) const noexcept {}

const ISymData* SelectSymbols::pointedSymbol(int, int) const noexcept {
  return nullptr;
}

void SelectSymbols::displaySymCode(unsigned long) const noexcept {}

void SelectSymbols::enlistSymbolForInvestigation(const ISymData&) const
    noexcept {}

void SelectSymbols::symbolsReadyToInvestigate() const noexcept {}

namespace {
ICmapPerspective::VPSymDataCIt dummyIt;
ICmapPerspective::VPSymDataCItPair dummyFontFaces(dummyIt, dummyIt);
set<unsigned> dummyClusterOffsets;
}  // anonymous namespace

ICmapPerspective::VPSymDataCItPair CmapPerspective::getSymsRange(...) const
    noexcept {
  return dummyFontFaces;
}

const set<unsigned>& CmapPerspective::getClusterOffsets() const noexcept {
  return dummyClusterOffsets;
}

void CmapPerspective::reset(
    const VSymData&,
    const std::vector<std::vector<unsigned>>&) noexcept {}

TinySym::TinySym(const Mat& negSym_,
                 const Point2d& mc_ /* = Point2d(.5, .5)*/,
                 double avgPixVal_ /* = 0.*/) noexcept
    : SymData(mc_, avgPixVal_),
      backslashDiagAvgProj(1, 2 * negSym_.rows - 1, CV_64FC1),
      slashDiagAvgProj(1, 2 * negSym_.rows - 1, CV_64FC1) {
  assert(!negSym_.empty());
  negSym = negSym_;
  Mat tinySymMat = 1 - negSym * INV_255;
  SymData::computeFields(tinySymMat, *this, true);

  mat = masks[(size_t)MaskType::GroundedSym].clone();
  // computing average projections
  cv::reduce(mat, hAvgProj, 0, cv::REDUCE_AVG);
  cv::reduce(mat, vAvgProj, 1, cv::REDUCE_AVG);

  Mat flippedMat;
  flip(mat, flippedMat, 1);  // flip around vertical axis
  const int tinySymSz = negSym_.rows;
  const double invTinySymSz = 1. / tinySymSz,
               invTinySymArea = invTinySymSz * invTinySymSz,
               invDiagsCountTinySym = 1. / (2. * tinySymSz - 1.);
  for (int diagIdx = -tinySymSz + 1, i = 0; diagIdx < tinySymSz;
       ++diagIdx, ++i) {
    const Mat backslashDiag = mat.diag(diagIdx);
    backslashDiagAvgProj.at<double>(i) = *mean(backslashDiag).val;

    const Mat slashDiag = flippedMat.diag(-diagIdx);
    slashDiagAvgProj.at<double>(i) = *mean(slashDiag).val;
  }

  // Ensuring the sum of all elements of the following matrices is in [0..1]
  // range
  mat *= invTinySymArea;
  hAvgProj *= invTinySymSz;
  vAvgProj *= invTinySymSz;
  backslashDiagAvgProj *= invDiagsCountTinySym;
  slashDiagAvgProj *= invDiagsCountTinySym;
}

SymData::SymData(unsigned long code_,
                 size_t symIdx_,
                 double minVal_,
                 double diffMinMax_,
                 double avgPixVal_,
                 double normSymMiu0_,
                 const Point2d& mc_,
                 const SymData::IdxMatMap& relevantMats,
                 const Mat& negSym_ /* = Mat()*/,
                 const Mat& symMiu0_ /* = Mat()*/) noexcept
    : code(code_),
      symIdx(symIdx_),
      minVal(minVal_),
      diffMinMax(diffMinMax_),
      avgPixVal(avgPixVal_),
      normSymMiu0(normSymMiu0_),
      mc(mc_),
      negSym(negSym_),
      symMiu0(symMiu0_) {
  for (const auto& idxAndMat : relevantMats)
    masks[idxAndMat.first] = idxAndMat.second;
}

unique_ptr<const SymData> SymData::clone(size_t symIdx_) const noexcept {
  return make_unique<const SymData>(negSym, symMiu0, code, symIdx_, minVal,
                                    diffMinMax, avgPixVal, normSymMiu0, mc,
                                    masks);
}

PixMapSym::PixMapSym(const vector<unsigned char>& data,
                     const Mat& consec,
                     const Mat& revConsec) noexcept
    : pixels(data) {
  const unsigned sz = (unsigned)consec.cols;
  const double maxGlyphSum = 255. * sz * sz;
  assert(sz == (unsigned)revConsec.rows);
  assert(sz * sz == (unsigned)data.size());

  rows = cols = (unsigned char)sz;
  top = (unsigned char)(rows - 1U);

  computeMcAndAvgPixVal(sz, maxGlyphSum, data, rows, cols, 0U, top, consec,
                        revConsec, mc, avgPixVal, &colSums, &rowSums);
}

static const Mat blurredVersionOf(const Mat& orig_) noexcept {
  Mat blurred;
  extern const Size BlurWinSize;
  extern const double BlurStandardDeviation;
  GaussianBlur(orig_, blurred, BlurWinSize, BlurStandardDeviation,
               BlurStandardDeviation, BORDER_REPLICATE);
  return blurred;
}

Patch::Patch(const Mat& orig_) noexcept
    : Patch(orig_, blurredVersionOf(orig_), orig_.channels() > 1) {}

Patch& Patch::setMatrixToApprox(const Mat& m) noexcept {
  const_cast<Mat&>(grayD) = m;
  return *this;
}

void Patch::forceApproximation() noexcept {
  const_cast<bool&>(needsApproximation) = true;
}

bool ClusterEngine::clusteredAlready(const string&,
                                     const string&,
                                     std::filesystem::path&) noexcept {
  return false;
}

bool FontEngine::isTinySymsDataSavedOnDisk(const string&,
                                           std::filesystem::path&) noexcept {
  return false;
}

void AbsJobMonitor::getReady(ITimerResult&) noexcept {}
