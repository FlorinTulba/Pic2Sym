/******************************************************************************
 The application Pic2Sym approximates images by a
 grid of colored symbols with colored backgrounds.

 Copyrights from the libraries used by the program:
 - (c) 2003-2021 Boost (www.boost.org)
     License: doc/licenses/Boost.lic
     http://www.boost.org/LICENSE_1_0.txt
 - (c) 2015-2021 OpenCV (www.opencv.org)
     License: doc/licenses/OpenCV.lic
     http://opencv.org/license/
 - (c) 1996-2021 The FreeType Project (www.freetype.org)
     License: doc/licenses/FTL.txt
     http://git.savannah.gnu.org/cgit/freetype/freetype2.git/plain/docs/FTL.TXT
 - (c) 1997-2021 OpenMP Architecture Review Board (www.openmp.org)
   (c) Microsoft Corporation (implementation for OpenMP C/C++ v2.0 March 2002)
     See: https://msdn.microsoft.com/en-us/library/8y6825x5.aspx
 - (c) 1995-2021 zlib software (Jean-loup Gailly and Mark Adler - www.zlib.net)
     License: doc/licenses/zlib.lic
     http://www.zlib.net/zlib_license.html
 - (c) 2015-2021 Microsoft Guidelines Support Library - github.com/microsoft/GSL
     License: doc/licenses/MicrosoftGSL.lic
     https://raw.githubusercontent.com/microsoft/GSL/main/LICENSE


 (c) 2016-2021 Florin Tulba <florintulba@yahoo.com>

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
// This keeps precompiled.h first; Otherwise header sorting might move it

#define BOOST_TEST_MODULE Tests for project Pic2Sym

#include "testMain.h"

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
#include "transform.h"
#include "transformSupport.h"
#include "updateSymSettingsBase.h"
#include "views.h"
#include "warnings.h"

#pragma warning(push, 0)

#include <ctime>
#include <filesystem>
#include <fstream>
#include <future>
#include <optional>
#include <random>
#include <ranges>

#include <gsl/gsl>

#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#pragma warning(pop)

using namespace std;
using namespace cv;
using namespace gsl;
using namespace boost;
using namespace std::filesystem;

#pragma warning(disable : WARN_INCONSISTENT_DLL_LINKAGE)
int __cdecl omp_get_thread_num(void) {
  return 0;
}
#pragma warning(default : WARN_INCONSISTENT_DLL_LINKAGE)

namespace cv {
// waitKey is called within Controller::symbolsChanged(),
// but UnitTesting shouldn't depend on highgui, so it defines a dummy waitKey
// right here.
int __cdecl waitKey(int) {
  return 0;
}
}  // namespace cv

namespace pic2sym {

extern const Size BlurWinSize;
extern const double BlurStandardDeviation;

void checkCancellationRequest(std::future<void>&, std::atomic_flag&) noexcept {}

namespace ut {
Fixt::~Fixt() noexcept {
  Component<p2s::ui::ControlPanel>::comp.reset();
  Component<p2s::ui::Comparator>::comp.reset();
  Component<p2s::transform::Transformer>::comp.reset();
  Component<p2s::syms::FontEngine>::comp.reset();
  Component<p2s::input::Img>::comp.reset();
  Component<p2s::match::MatchEngine>::comp.reset();
}

namespace {
/// Returns the path to 'Pic2Sym.exe'
std::optional<const std::filesystem::path> pathToPic2Sym() {
#ifdef _DEBUG
#define CONFIG_TYPE "Debug"
#else  // _DEBUG not defined
#define CONFIG_TYPE "Release"
#endif  // _DEBUG
  std::filesystem::path dirOfPic2Sym{absolute(".")};
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

  std::filesystem::path pic2SymPath{
      std::filesystem::path(dirOfPic2Sym).append("Pic2Sym.exe")};
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
  ofstream ofs{absolute(dirOfPic2Sym).append("issues.bat").string(), ios::app};
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

void showMismatches(
    const string& testTitle,
    const vector<unique_ptr<p2s::match::BestMatch>>& mismatches) {
  if (mismatches.empty())
    return;

  const unsigned cases{narrow_cast<unsigned>(std::size(mismatches))};
  const auto& firstItem = *mismatches.front();
  const auto& firstRefM = firstItem.getPatch().getOrig();
  const int tileSz{firstRefM.rows};
  const bool isColor{firstRefM.channels() > 1};
  cerr << "There were " << cases << " unexpected matches" << endl;

  auto pic2SymPath = pathToPic2Sym();
  if (!pic2SymPath)
    return;

  const auto dirOfPic2Sym = pic2SymPath->parent_path();

  // Ensure there is a folder x64\<ConfigType>\UnitTesting\Mismatches
  std::filesystem::path mismatchesFolder{dirOfPic2Sym};
  if (!exists(mismatchesFolder.append("UnitTesting").append("Mismatches")))
    create_directory(mismatchesFolder);

  time_t suffix{time(nullptr)};
  const string uniqueTestTitle{testTitle + "_" + to_string(suffix)};
  ostringstream oss;
  oss << "Pic2Sym.exe mismatches " << quoted(uniqueTestTitle);
  string commandLine{oss.str()};

  // Tiling the references/results in 2 rectangles with approx 800 x 600 aspect
  // ratio
  static constexpr double ar{800 / 600.};

  const double widthD{round(sqrt(cases * ar))};
  const double heightD{ceil(cases / widthD)};
  const unsigned widthTiles{narrow_cast<unsigned>(widthD)};
  const unsigned heightTiles{narrow_cast<unsigned>(heightD)};
  const int width{(int)widthTiles * tileSz};
  const int height{(int)heightTiles * tileSz};

  // combined
  Mat m{2 * height, width, (isColor ? CV_8UC3 : CV_8UC1), Scalar::all(127.)};
  Mat mRef{m, Range{0, height}};           // upper half contains the references
  Mat mRes{m, Range{height, 2 * height}};  // lower half for results

  // Tile the references & results
  for (unsigned idx{}, r{}; r < heightTiles && idx < cases; ++r) {
    Range rowRange{(int)r * tileSz, int(r + 1U) * tileSz};
    for (unsigned c{}; c < widthTiles && idx < cases; ++c, ++idx) {
      Range colRange{(int)c * tileSz, int(c + 1) * tileSz};
      const auto& item = *mismatches[idx];
      item.getPatch().getOrig().copyTo(
          Mat{mRef, rowRange, colRange});                      // reference
      item.getApprox().copyTo(Mat{mRes, rowRange, colRange});  // result
    }
  }

  // write the combined matrices as <uniqueTestTitle>.jpg
  std::filesystem::path destFile{std::filesystem::path(mismatchesFolder)
                                     .append(uniqueTestTitle)
                                     .concat(".jpg")};
  if (false == imwrite(destFile.string().c_str(), m))
    cerr << "Couldn't write the image generated by Unit Tests!\n"
         << '[' << destFile << ']' << endl;

  // start detached process to display a comparator with the references &
  // results
  createVisualizer(dirOfPic2Sym, commandLine, "mismatches");
}

void showMisfiltered(
    const string& testTitle,
    const vector<std::unique_ptr<const p2s::syms::IPixMapSym>>& misfiltered) {
  if (misfiltered.empty())
    return;

  auto pic2SymPath = pathToPic2Sym();
  if (!pic2SymPath)
    return;

  const auto dirOfPic2Sym = pic2SymPath->parent_path();

  // Ensure there is a folder x64\<ConfigType>\UnitTesting\Misfiltered
  std::filesystem::path misfilteredFolder{dirOfPic2Sym};
  if (!exists(misfilteredFolder.append("UnitTesting").append("Misfiltered")))
    create_directory(misfilteredFolder);

  time_t suffix{time(nullptr)};

  // Append an unique suffix
  string uniqueTestTitle{testTitle + "_" + to_string(suffix)};
  for (const char toReplace : " :"s)
    // Replace all spaces and colons with underscores
    ranges::replace(uniqueTestTitle, toReplace, '_');
  ostringstream oss;
  oss << "Pic2Sym.exe misfiltered " << quoted(uniqueTestTitle);
  string commandLine{oss.str()};

  // Construct the image to display, which should contain all miscategorized
  // symbols from one group
  static const Scalar GridColor{255U, 200U, 200U};
  static const Scalar BgColor{200U, 200U, 255U};

  const int borderRows{2};
  const int vertSplitsCount{1 + narrow_cast<int>(std::ssize(misfiltered))};
  int rowsMislabeledSyms{};
  int colsMislabeledSyms{1};
  int tallestSym{};
  int vertSplitIdx{1};
  vector<int> posVertSplits(vertSplitsCount, 0);
  for (const auto& sym : misfiltered) {
    // narrow versions are now the whole symbols
    const Mat symAsMat{sym->asNarrowMat()};
    if (symAsMat.rows > tallestSym)
      tallestSym = symAsMat.rows;
    colsMislabeledSyms += symAsMat.cols + 1;
    posVertSplits[vertSplitIdx++] = colsMislabeledSyms - 1;
  }
  rowsMislabeledSyms = borderRows + tallestSym;
  Mat mislabeledSyms{rowsMislabeledSyms, colsMislabeledSyms, CV_8UC3, BgColor};
  mislabeledSyms.row(0).setTo(GridColor);
  mislabeledSyms.row(rowsMislabeledSyms - 1).setTo(GridColor);
  for (const int posVertSplit : posVertSplits)
    mislabeledSyms.col(posVertSplit).setTo(GridColor);
  vertSplitIdx = 0;
  for (const auto& sym : misfiltered) {
    // narrow versions are now the whole symbols
    const Mat symMat{sym->asNarrowMat()};
    const vector<Mat> symChannels(3, symMat);
    Mat symAsIfColor;
    Mat region{mislabeledSyms, Range{1, symMat.rows + 1},
               Range{posVertSplits[vertSplitIdx] + 1,
                     posVertSplits[vertSplitIdx + 1ULL]}};
    merge(symChannels, symAsIfColor);
    symAsIfColor.copyTo(region);
    ++vertSplitIdx;
  }

  // write the misfiltered symbols as <uniqueTestTitle>.jpg
  std::filesystem::path destFile{std::filesystem::path(misfilteredFolder)
                                     .append(uniqueTestTitle)
                                     .concat(".jpg")};
  if (false == imwrite(destFile.string().c_str(), mislabeledSyms))
    cerr << "Couldn't write the image generated by Unit Tests!\n"
         << '[' << destFile << ']' << endl;

  createVisualizer(dirOfPic2Sym, commandLine, "misfiltered symbols");
}

unsigned randUnifUint() {
  static random_device rd;
  static mt19937 gen{rd()};
  static uniform_int_distribution<unsigned> uid;

  return uid(gen);
}

unsigned char randUnsignedChar(unsigned char minIncl /* = 0U*/,
                               unsigned char maxIncl /* = 255U*/) {
  return narrow_cast<unsigned char>(
      minIncl + randUnifUint() % ((unsigned)(maxIncl - minIncl) + 1U));
}
}  // namespace ut

cfg::MatchSettings::MatchSettings() {}

input::Img& ControlPanelActions::getImg() noexcept {
  return ut::Fixt::Component<input::Img>::get();
}

ui::IControlPanel& ControlPanelActions::getControlPanel(
    const cfg::ISettingsRW& cfg_) noexcept {
  return ut::Fixt::Component<ui::ControlPanel>::get(*this, cfg_);
}

bool ControlPanelActions::newImage(const Mat& imgMat) noexcept {
  bool result{img->reset(imgMat)};

  if (result) {
    cout << "Using Matrix instead of a "
         << (img->isColor() ? "color" : "grayscale") << " image" << endl;
    if (!imageOk)
      imageOk = true;

    // For valid matrices of size sz x sz, ignore MIN_H_SYMS & MIN_V_SYMS =>
    // Testing an image containing a single patch
    if ((unsigned)imgMat.cols == cfg->getSS().getFontSz() &&
        (unsigned)imgMat.rows == cfg->getSS().getFontSz()) {
      if (1U != cfg->getIS().getMaxHSyms())
        cfg->refIS().setMaxHSyms(1U);
      if (1U != cfg->getIS().getMaxVSyms())
        cfg->refIS().setMaxVSyms(1U);
    }
  }

  return result;
}

ui::IComparator& Controller::getComparator() noexcept {
  return ut::Fixt::Component<ui::Comparator>::get();
}

syms::IFontEngine& Controller::getFontEngine(
    const cfg::ISymSettings& ss_) noexcept {
  return ut::Fixt::Component<syms::FontEngine>::get(*this, ss_);
}

match::IMatchEngine& Controller::getMatchEngine(
    const cfg::ISettings& cfg_) noexcept {
  return ut::Fixt::Component<match::MatchEngine>::get(
      cfg_, getFontEngine(cfg_.getSS()), *cmP);
}

transform::ITransformer& Controller::getTransformer(
    const cfg::ISettings& cfg_) noexcept {
  return ut::Fixt::Component<transform::Transformer>::get(
      *this, cfg_, getMatchEngine(cfg_), ControlPanelActions::getImg());
}

Controller::~Controller() noexcept {}

void Controller::hourGlass(double, const string&, bool) const {}

void PicTransformProgressTracker::transformFailedToStart() noexcept {}

void PicTransformProgressTracker::reportTransformationProgress(double, bool)
    const noexcept {}

void PicTransformProgressTracker::presentTransformationResults(
    double) const noexcept {}

void GlyphsProgressTracker::updateSymsDone(double) const noexcept {}

void Controller::updateStatusBarCmapInspect(unsigned,
                                            const string&,
                                            bool) const {}

void Controller::reportDuration(string_view, double) const {}

bool Controller::updateResizedImg(const input::IResizedImg&) noexcept {
  return true;
}

void Controller::showResultedImage(double) const noexcept {}

const syms::ISymData* SelectSymbols::pointedSymbol(int, int) const noexcept {
  return nullptr;
}

void SelectSymbols::displaySymCode(unsigned long) const noexcept {}

void SelectSymbols::enlistSymbolForInvestigation(
    const syms::ISymData&) const noexcept {}

void SelectSymbols::symbolsReadyToInvestigate() const noexcept {}

namespace {
ui::ICmapPerspective::VPSymDataCIt dummyIt;
ui::ICmapPerspective::VPSymDataRange dummyFontFaces{dummyIt, dummyIt};
set<unsigned> dummyClusterOffsets;
}  // anonymous namespace

ui::ICmapPerspective::VPSymDataRange ui::CmapPerspective::getSymsRange(
    ...) const noexcept {
  return dummyFontFaces;
}

const set<unsigned>& ui::CmapPerspective::getClusterOffsets() const noexcept {
  return dummyClusterOffsets;
}

void ui::CmapPerspective::reset(
    const syms::VSymData&,
    const std::vector<std::vector<unsigned>>&) noexcept {}

syms::TinySym::TinySym(const Mat& negSym_,
                       const Point2d& mc_ /* = Point2d(.5, .5)*/,
                       double avgPixVal_ /* = 0.*/) noexcept
    : syms::SymData{mc_, avgPixVal_},

      // 0. parameter prevents using the initializer_list ctor of Mat
      backslashDiagAvgProj{1, 2 * negSym_.rows - 1, CV_64FC1, 0.},
      slashDiagAvgProj{1, 2 * negSym_.rows - 1, CV_64FC1, 0.} {
  assert(!negSym_.empty());
  negSym = negSym_;
  Mat tinySymMat{1 - negSym * Inv255};
  syms::SymData::computeFields(tinySymMat, *this, true);

  mat = masks[(size_t)MaskType::GroundedSym].clone();
  // computing average projections
  cv::reduce(mat, hAvgProj, 0, cv::REDUCE_AVG);
  cv::reduce(mat, vAvgProj, 1, cv::REDUCE_AVG);

  Mat flippedMat;
  flip(mat, flippedMat, 1);  // flip around vertical axis
  const int tinySymSz{negSym_.rows};
  const double invTinySymSz{1. / tinySymSz};
  const double invTinySymArea{invTinySymSz * invTinySymSz};
  const double invDiagsCountTinySym{1. / (2. * tinySymSz - 1.)};
  for (int diagIdx{-tinySymSz + 1}, i{}; diagIdx < tinySymSz; ++diagIdx, ++i) {
    const Mat backslashDiag{mat.diag(diagIdx)};
    backslashDiagAvgProj.at<double>(i) = *mean(backslashDiag).val;

    const Mat slashDiag{flippedMat.diag(-diagIdx)};
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

syms::SymData::SymData(unsigned long code_,
                       size_t symIdx_,
                       double minVal_,
                       double diffMinMax_,
                       double avgPixVal_,
                       double normSymMiu0_,
                       const Point2d& mc_,
                       const syms::SymData::IdxMatMap& relevantMats,
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

unique_ptr<const syms::SymData> syms::SymData::clone(
    size_t symIdx_) const noexcept {
  return make_unique<const syms::SymData>(negSym, symMiu0, code, symIdx_,
                                          minVal, diffMinMax, avgPixVal,
                                          normSymMiu0, mc, masks);
}

syms::PixMapSym::PixMapSym(const vector<unsigned char>& data,
                           const Mat& consec,
                           const Mat& revConsec) noexcept
    : pixels(data) {
  const unsigned sz{(unsigned)consec.cols};
  const double maxGlyphSum{255. * sz * sz};
  assert(sz == (unsigned)revConsec.rows);
  assert(sz * sz == narrow_cast<unsigned>(std::size(data)));

  rows = cols = narrow_cast<unsigned char>(sz);
  top = narrow_cast<unsigned char>(rows - 1U);

  computeMcAndAvgPixVal(sz, maxGlyphSum, data, rows, cols, 0U, top, consec,
                        revConsec, mc, avgPixVal, &colSums, &rowSums);
}

static const Mat blurredVersionOf(const Mat& orig_) noexcept {
  Mat blurred;
  GaussianBlur(orig_, blurred, BlurWinSize, BlurStandardDeviation,
               BlurStandardDeviation, BORDER_REPLICATE);
  return blurred;
}

namespace input {

Patch::Patch(const Mat& orig_) noexcept
    : Patch{orig_, blurredVersionOf(orig_), orig_.channels() > 1} {}

Patch& Patch::setMatrixToApprox(const Mat& m) noexcept {
  const_cast<Mat&>(grayD) = m;
  return *this;
}

void Patch::forceApproximation() noexcept {
  const_cast<bool&>(needsApproximation) = true;
}

}  // namespace input

bool syms::cluster::ClusterEngine::clusteredAlready(
    const string&,
    const string&,
    std::filesystem::path&) noexcept {
  return false;
}

bool syms::FontEngine::isTinySymsDataSavedOnDisk(
    const string&,
    std::filesystem::path&) noexcept {
  return false;
}

void ui::AbsJobMonitor::getReady(ITimerResult&) noexcept {}

}  // namespace pic2sym
