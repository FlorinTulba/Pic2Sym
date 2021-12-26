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

#include "fontEngine.h"

#include "controllerBase.h"
#include "fontErrorsHelper.h"
#include "glyphsProgressTracker.h"
#include "jobMonitorBase.h"
#include "misc.h"
#include "ompTrace.h"
#include "pmsCont.h"
#include "presentCmap.h"
#include "settingsBase.h"
#include "symFilter.h"
#include "symFilterCache.h"
#include "symSettingsBase.h"
#include "taskMonitor.h"

#pragma warning(push, 0)

#include FT_TRUETYPE_IDS_H

#include <iomanip>

#pragma warning(pop)

using namespace std;
using namespace std::filesystem;
using namespace boost::bimaps;
using namespace gsl;
using namespace cv;

namespace pic2sym {

extern const bool ParallelizePixMapStatistics;

namespace syms {

namespace {
/// Creates a bimap using initializer_list. Needed in 'encodingsMap' below
template <typename L, typename R>
bimap<L, R> make_bimap(
    initializer_list<typename bimap<L, R>::value_type> il) noexcept {
  return bimap<L, R>{CBOUNDS(il)};
}

/**
@return mapping between encodings codes and their corresponding names.
It's non-const just to allow accessing the map with operator[].
*/
const bimap<FT_Encoding, string>& encodingsMap() noexcept {
  // Defines pairs like { FT_ENCODING_ADOBE_STANDARD, "ADOBE_STANDARD" }
#define ENC(encValue) \
  { encValue, string(#encValue).substr(12) }

  static const bimap<FT_Encoding, string> encMap(
      make_bimap<FT_Encoding, string>(
          {// known encodings
           ENC(FT_ENCODING_NONE), ENC(FT_ENCODING_UNICODE),
           ENC(FT_ENCODING_MS_SYMBOL), ENC(FT_ENCODING_ADOBE_LATIN_1),
           ENC(FT_ENCODING_OLD_LATIN_2), ENC(FT_ENCODING_SJIS),
           ENC(FT_ENCODING_GB2312), ENC(FT_ENCODING_BIG5),
           ENC(FT_ENCODING_WANSUNG), ENC(FT_ENCODING_JOHAB),
           ENC(FT_ENCODING_ADOBE_STANDARD), ENC(FT_ENCODING_ADOBE_EXPERT),
           ENC(FT_ENCODING_ADOBE_CUSTOM), ENC(FT_ENCODING_APPLE_ROMAN)}));
#undef ENC

  return encMap;
}

/// Required data for a symbol to be resized twice
struct DataForSymToResize {
  FT_ULong symCode;  ///< code of the symbol
  size_t symIdx;     ///< index within charmap
  double hRatio;     ///< horizontal resize ratio
  double vRatio;     ///< vertical resize ratio
};
}  // anonymous namespace

#pragma warning(disable \
                : WARN_DYNAMIC_CAST_MIGHT_FAIL WARN_THROWS_ALTHOUGH_NOEXCEPT)
FontEngine::FontEngine(IController& ctrler_,
                       const p2s::cfg::ISymSettings& ss_) noexcept
    : IFontEngine(),
      symSettingsUpdater(&ctrler_.getUpdateSymSettings()),
      cmapPresenter(&ctrler_.getPresentCmap()),
      ss(&ss_),
      symsCont(make_unique<PmsCont>(ctrler_)) {
  const FT_Error error{FT_Init_FreeType(&library)};
  EXPECTS_OR_REPORT_AND_THROW(
      error == FT_Err_Ok, runtime_error,
      "Couldn't initialize FreeType! Error: " + FtErrors[(size_t)error]);
}
#pragma warning(default \
                : WARN_DYNAMIC_CAST_MIGHT_FAIL WARN_THROWS_ALTHOUGH_NOEXCEPT)

FontEngine::~FontEngine() noexcept {
  FT_Done_Face(face);
  FT_Done_FreeType(library);
}

void FontEngine::invalidateFont() noexcept {
  FT_Done_Face(face);
  face = nullptr;
  disposeTinySyms();
  uniqueEncs.clear();
  symsCont->reset();
  symsUnableToLoad.clear();
  encodingIndex = symsCount = 0U;
}

bool FontEngine::checkFontFile(const path& fontPath,
                               FT_Face& face_) const noexcept {
  if (!exists(fontPath)) {
    cerr << "No such file: '" << fontPath.string() << '\'' << endl;
    return false;
  }

  if (const FT_Error error{
          FT_New_Face(library, fontPath.string().c_str(), 0, &face_)};
      error != FT_Err_Ok) {
    cerr << "Invalid font file: '" << fontPath.string()
         << "'  Error: " << FtErrors[(size_t)error] << endl;
    return false;
  }
  /*
  // Some faces not providing this flag 'squeeze' basic ASCII characters to the
  // left of the square

  if (!FT_IS_FIXED_WIDTH(face_)) {
    cerr << "The font file '" << fontPath.string()
         << "' isn't a fixed-width (monospace) font! Flags: 0x" << hex
         << face_->face_flags << dec << endl;
    return false;
  }
  */

  if (!FT_IS_SCALABLE(face_)) {
    cerr << "The font file '" << fontPath.string() << "' isn't a scalable font!"
         << endl;
    return false;
  }

  return true;
}

#pragma warning(disable : WARN_THROWS_ALTHOUGH_NOEXCEPT)
bool FontEngine::setNthUniqueEncoding(unsigned idx) noexcept(!UT) {
  EXPECTS_OR_REPORT_AND_THROW_CONST_MSG(
      face, logic_error,
      "Call FontEngine::newFont() before calling "s + HERE.function_name());

  if (idx == encodingIndex)
    return true;  // same encoding

  if (idx >= uniqueEncodings())
    return false;

  if (const FT_Error error{FT_Set_Charmap(
          face, face->charmaps[next(uniqueEncs.right.begin(), idx)->first])};
      error != FT_Err_Ok) {
    cerr << "Couldn't set new cmap! Error: " << FtErrors[(size_t)error] << endl;
    return false;
  }

  encodingIndex = idx;
  const string& encName =
      encodingsMap().left.find(face->charmap->encoding)->second;
  cout << "Using encoding " << quoted(encName, '\'') << " (index "
       << encodingIndex << ')' << endl;

  tinySyms.clear();
  symsCont->reset();
  symsCount = 0U;
  symsUnableToLoad.clear();

  symSettingsUpdater->newFontEncoding(encName);

  return true;
}
#pragma warning(default : WARN_THROWS_ALTHOUGH_NOEXCEPT)

#pragma warning(disable : WARN_THROWS_ALTHOUGH_NOEXCEPT)
bool FontEngine::setEncoding(const string& encName,
                             bool forceUpdate /* = false*/) noexcept(!UT) {
  EXPECTS_OR_REPORT_AND_THROW_CONST_MSG(
      face, logic_error,
      "Call FontEngine::newFont() before calling "s + HERE.function_name());

  if (encName == ss->getEncoding() && !forceUpdate)
    return true;  // same encoding

  const auto& encMapR = encodingsMap().right;  // encodingName->FT_Encoding
  const auto itEncName = encMapR.find(encName);
  if (encMapR.end() == itEncName) {
    cerr << "Unknown encoding " << encName << endl;
    return false;
  }

  const auto& uniqueEncsL = uniqueEncs.left;
  const auto itEnc =
      uniqueEncsL.find(itEncName->second);  // FT_Encoding->uniqueIndices
  if (uniqueEncsL.end() == itEnc) {
    cerr << "Current font doesn't contain encoding " << encName << endl;
    return false;
  }

  encodingIndex = UINT_MAX;
  return setNthUniqueEncoding(itEnc->second);
}
#pragma warning(default : WARN_THROWS_ALTHOUGH_NOEXCEPT)

#pragma warning(disable : WARN_THROWS_ALTHOUGH_NOEXCEPT)
void FontEngine::setFace(FT_Face face_,
                         const string& /*fontFile_ = ""*/) noexcept(!UT) {
  EXPECTS_OR_REPORT_AND_THROW_CONST_MSG(
      face_, invalid_argument,
      "Cannot provide a NULL face argument to "s + HERE.function_name());

  if (face) {
    if (!strcmp(face->family_name, face_->family_name) &&
        !strcmp(face->style_name, face_->style_name))
      return;  // same face

    FT_Done_Face(face);
  }

  tinySyms.clear();
  symsCont->reset();
  symsCount = 0U;
  symsUnableToLoad.clear();
  uniqueEncs.clear();
  face = face_;

  cout << "Using " << quoted(face->family_name, '\'') << ' ' << face->style_name
       << endl;

  for (int i{}, charmapsCount{face->num_charmaps}; i < charmapsCount; ++i)
    uniqueEncs.insert(bimap<FT_Encoding, unsigned>::value_type(
        face->charmaps[(size_t)i]->encoding, (unsigned)i));

  cout << "The available encodings are:";
  for (const auto& enc : uniqueEncs.right)
    cout << ' ' << quoted(encodingsMap().left.find(enc.second)->second, '\'');
  cout << endl;

  encodingIndex = UINT_MAX;
  setNthUniqueEncoding(0U);
}
#pragma warning(default : WARN_THROWS_ALTHOUGH_NOEXCEPT)

bool FontEngine::newFont(const string& fontFile_) noexcept {
  FT_Face face_;
  const path fontPath{absolute(fontFile_)};
  if (!checkFontFile(fontPath, face_))
    return false;

  setFace(face_, fontPath.string());

  symSettingsUpdater->newFontFile(fontFile_);

  return true;
}

#pragma warning(disable : WARN_THROWS_ALTHOUGH_NOEXCEPT)
void FontEngine::adjustScaling(unsigned sz,
                               FT_BBox& bb,
                               double& factorH,
                               double& factorV) noexcept(!UT) {
  vector<double> vTop, vBottom, vLeft, vRight, vHeight, vWidth;
  FT_Size_RequestRec req;
  req.type = FT_SIZE_REQUEST_TYPE_REAL_DIM;  // FT_SIZE_REQUEST_TYPE_BBOX, ...
  req.height = FT_Long(sz << 6);             // 26.6 format
  req.width = req.height;  // initial check for square drawing board
  req.horiResolution = req.vertResolution =
      72U;  // 72dpi is set by default by higher-level methods
  FT_Error error{FT_Err_Ok};
  if (error = FT_Request_Size(face, &req); error != FT_Err_Ok)
    reportAndThrow<invalid_argument>(
        "Couldn't set font size: " + to_string(sz) +
        "  Error: " + FtErrors[(size_t)error]);
  symsUnableToLoad.clear();
  FT_UInt idx;
  for (FT_ULong c{FT_Get_First_Char(face, &idx)}; idx;
       c = FT_Get_Next_Char(face, c, &idx)) {
    if (error = FT_Load_Char(face, c, FT_LOAD_RENDER); error != FT_Err_Ok) {
      cerr << "Couldn't load glyph " << c
           << " before resizing. Error: " << FtErrors[(size_t)error] << endl;
      symsUnableToLoad.insert(c);
      continue;
    }
    not_null<const FT_GlyphSlotRec_* const> g{face->glyph};
    const FT_Bitmap b{g->bitmap};

    const unsigned height{b.rows};
    const unsigned width{b.width};
    vHeight.push_back(height);
    vWidth.push_back(width);

    const int left{g->bitmap_left};
    const int right{left + (int)width - 1};
    const int top{g->bitmap_top};
    const int bottom{top - (int)height + 1};
    vLeft.push_back(left);
    vRight.push_back(right);
    vTop.push_back(top);
    vBottom.push_back(bottom);
  }
  symsCount = narrow_cast<unsigned>(size(vTop));

  // Compute some means and standard deviations
  Vec<double, 1> avgTop, sdTop, avgBottom, sdBottom, avgLeft, sdLeft, avgRight,
      sdRight;
  Scalar avgHeight, avgWidth;
#pragma warning(disable : WARN_CODE_ANALYSIS_IGNORES_OPENMP)
#pragma omp parallel if (ParallelizePixMapStatistics)
#pragma omp sections nowait
  {
#pragma omp section
    {
      OMP_PRINTF(ParallelizePixMapStatistics, "height");
      avgHeight = mean(Mat{1, (int)symsCount, CV_64FC1, vHeight.data()});
    }
#pragma omp section
    {
      OMP_PRINTF(ParallelizePixMapStatistics, "width");
      avgWidth = mean(Mat{1, (int)symsCount, CV_64FC1, vWidth.data()});
    }
#pragma omp section
    {
      OMP_PRINTF(ParallelizePixMapStatistics, "top");
      meanStdDev(Mat{1, (int)symsCount, CV_64FC1, vTop.data()}, avgTop, sdTop);
    }
#pragma omp section
    {
      OMP_PRINTF(ParallelizePixMapStatistics, "bottom");
      meanStdDev(Mat{1, (int)symsCount, CV_64FC1, vBottom.data()}, avgBottom,
                 sdBottom);
    }
#pragma omp section
    {
      OMP_PRINTF(ParallelizePixMapStatistics, "left");
      meanStdDev(Mat{1, (int)symsCount, CV_64FC1, vLeft.data()}, avgLeft,
                 sdLeft);
    }
#pragma omp section
    {
      OMP_PRINTF(ParallelizePixMapStatistics, "right");
      meanStdDev(Mat{1, (int)symsCount, CV_64FC1, vRight.data()}, avgRight,
                 sdRight);
    }
  }
#pragma warning(default : WARN_CODE_ANALYSIS_IGNORES_OPENMP)

  // 1. means a single standard deviation => ~68% of the data
  static constexpr double kv{1.};
  static constexpr double kh{1.};

  // Enlarge factors, forcing the average width + lateral std. devs
  // to fit the width of the drawing square.
  factorH = sz / (*avgWidth.val + kh * (*sdLeft.val + *sdRight.val));
  factorV = sz / (*avgHeight.val + kv * (*sdTop.val + *sdBottom.val));

  // Computing new height & width
  req.height = (FT_Long)floor(factorV * req.height);
  req.width = (FT_Long)floor(factorH * req.width);

  // Reshaping the fonts to better fill the drawing square
  if (error = FT_Request_Size(face, &req); error != FT_Err_Ok)
    reportAndThrow<invalid_argument>(
        "Couldn't set font size: " + to_string(sz) +
        "  Error: " + FtErrors[(size_t)error]);

  // Positioning the Bounding box to best cover the estimated future position &
  // size of the symbols

  // current bottom scaled by factorV
  double yMin{factorV * (*avgBottom.val - *sdBottom.val)};
  double yMax{factorV * (*avgTop.val + *sdTop.val)};  // top

  // the difference to divide equally between top & bottom
  double yDiff2{(yMax - yMin + 1 - sz) / 2.};

  // current left scaled by factorH
  double xMin{factorH * (*avgLeft.val - *sdLeft.val)};
  double xMax{factorH * (*avgRight.val + *sdRight.val)};  // right

  // the difference to divide equally between left & right
  const double xDiff2{(xMax - xMin + 1 - sz) / 2.};

  // distributing the differences
  yMin += yDiff2;
  yMax -= yDiff2;
  xMin += xDiff2;
  xMax -= xDiff2;

  // ensure yMin <= 0 (should be at most the baseline y coord, which is 0)
  if (yMin > 0) {
    yMax -= yMin;
    yMin = 0;
  }

  bb.xMin = (FT_Pos)round(xMin);
  bb.xMax = (FT_Pos)round(xMax);
  bb.yMin = (FT_Pos)round(yMin);
  bb.yMax = (FT_Pos)round(yMax);
}
#pragma warning(default : WARN_THROWS_ALTHOUGH_NOEXCEPT)

void FontEngine::setFontSz(unsigned fontSz_) {
  EXPECTS_OR_REPORT_AND_THROW_CONST_MSG(
      face, logic_error,
      "Call FontEngine::newFont() before calling "s + HERE.function_name());
  EXPECTS_OR_REPORT_AND_THROW_CONST_MSG(
      symsMonitor, logic_error,
      "Call FontEngine::setSymsMonitor() before calling "s +
          HERE.function_name());
  EXPECTS_OR_REPORT_AND_THROW(p2s::cfg::ISettings::isFontSizeOk(fontSz_),
                              invalid_argument,
                              "Invalid fontSz " + to_string(fontSz_));

  if (symsCont->isReady() && symsCont->getFontSz() == fontSz_)
    return;  // same font size

  cout << "Setting font size " << fontSz_ << endl;

  const double sz{narrow_cast<double>(fontSz_)};
  vector<DataForSymToResize> toResize;
  double factorH{};
  double factorV{};
  FT_BBox bb{0LL, 0LL, 0LL, 0LL};
  FT_UInt idx{};
  FT_Size_RequestRec req;
  req.type = FT_SIZE_REQUEST_TYPE_REAL_DIM;
  req.horiResolution = req.vertResolution = 72U;

  using p2s::ui::TaskMonitor;

  static TaskMonitor determineOptimalSquareFittingSymbols{
      "determine optimal square-fitting "
      "for the symbols",
      *symsMonitor};

  adjustScaling(fontSz_, bb, factorH, factorV);

  determineOptimalSquareFittingSymbols.taskDone();

  static TaskMonitor loadFitSymbols{"load & filter symbols that fit the square",
                                    *symsMonitor};

  loadFitSymbols.setTotalSteps((size_t)symsCount);
  symsCont->reset(fontSz_, symsCount);

  cmapPresenter->showUnofficialSymDetails(symsCount);

  FT_Error error{FT_Err_Ok};
  SymFilterCache sfc;
  sfc.setFontSz(fontSz_);
  // Store the pixmaps of the symbols that fit the bounding box already or by
  // shifting. Preserve the symbols that don't fit, in order to resize them
  // first, then add them too to pixmaps.
  size_t i{};
  size_t countOfSymsUnableToLoad{};
  for (FT_ULong c{FT_Get_First_Char(face, &idx)}; idx;
       c = FT_Get_Next_Char(face, c, &idx), loadFitSymbols.taskAdvanced(++i)) {
    if (error = FT_Load_Char(face, c, FT_LOAD_RENDER); error != FT_Err_Ok) {
      if (symsUnableToLoad.contains(c)) {  // known glyph
        ++countOfSymsUnableToLoad;
        continue;
      } else  // unexpected glyph
        reportAndThrow<runtime_error>(
            "Couldn't load an unexpected glyph (" + to_string(c) +
            ") during initial resizing. Error: " + FtErrors[(size_t)error]);
    }
    const FT_GlyphSlot g{face->glyph};
    const FT_Bitmap b{g->bitmap};
    const unsigned height{b.rows};
    const unsigned width{b.width};
    if (width > fontSz_ || height > fontSz_)
      toResize.push_back({.symCode = c,
                          .symIdx = i,
                          .hRatio = max(1., width / sz),
                          .vRatio = max(1., height / sz)});
    else
      symsCont->appendSym(c, i, g, bb, sfc);
  }

  if (countOfSymsUnableToLoad < size(symsUnableToLoad))
    reportAndThrow<runtime_error>(
        "Initial resizing of the glyphs found only " +
        to_string(countOfSymsUnableToLoad) +
        " symbols that couldn't be loaded when expecting " +
        to_string(size(symsUnableToLoad)));

  loadFitSymbols.taskDone();

  // Resize symbols which didn't fit initially
  static TaskMonitor loadExtraSqueezedSymbols{
      "load & filter extra-squeezed symbols", *symsMonitor};

  loadExtraSqueezedSymbols.setTotalSteps(size(toResize));

  const FT_Long fontSzMul64{(FT_Long)(fontSz_) << 6};
  const double numeratorV{factorV * fontSzMul64};
  const double numeratorH{factorH * fontSzMul64};
  i = 0U;
  for (const DataForSymToResize& item : toResize) {
    req.height = (FT_Long)floor(numeratorV / item.vRatio);
    req.width = (FT_Long)floor(numeratorH / item.hRatio);
    if (error = FT_Request_Size(face, &req); error != FT_Err_Ok)
      reportAndThrow<invalid_argument>(
          "Couldn't set font size: " +
          to_string(factorV * fontSz_ / item.vRatio) + " x " +
          to_string(factorH * fontSz_ / item.hRatio) +
          "  Error: " + FtErrors[(size_t)error]);

    if (error = FT_Load_Char(face, item.symCode, FT_LOAD_RENDER);
        error != FT_Err_Ok)
      reportAndThrow<runtime_error>(
          "Couldn't load glyph " + to_string(item.symCode) +
          " which needed resizing twice. Error: " + FtErrors[(size_t)error]);
    symsCont->appendSym(item.symCode, item.symIdx, face->glyph, bb, sfc);

    loadExtraSqueezedSymbols.taskAdvanced(++i);
  }

  // Determine coverageOfSmallGlyphs
  static TaskMonitor determineCoverageOfSmallGlyphs{
      "determine coverageOfSmallGlyphs", *symsMonitor};

  symsCont->setAsReady();
  determineCoverageOfSmallGlyphs.taskDone();  // mark it as already finished

  /**
  Original provided fonts are typically not square, so they need to be reshaped
  sometimes even twice, to fit within a square of a desired size - symbol's
  size.

  VIEW_CONCLUSIONS_FROM_RESHAPING_LOADED_FONTS should be defined when interested
  in the details about a set of reshaped fonts.
  */
  //#define VIEW_CONCLUSIONS_FROM_RESHAPING_LOADED_FONTS
#if defined(VIEW_CONCLUSIONS_FROM_RESHAPING_LOADED_FONTS) && \
    !defined(UNIT_TESTING)
  cout << "\nResulted Bounding box: " << bb.yMin << "," << bb.xMin << " -> "
       << bb.yMax << "," << bb.xMax
       << "\nSymbols considered small cover at most " << fixed
       << setprecision(2) << 100. * symsCont->getCoverageOfSmallGlyphs()
       << "% of the box\n";

  if (!toResize.empty()) {
    cout << size(toResize) << " symbols were resized twice: ";
    for (const DataForSymToResize& item : toResize)
      cout << item.symCode << ", ";
    cout << '\n';
  }

  cout << '\n';
#endif  // VIEW_CONCLUSIONS_FROM_RESHAPING_LOADED_FONTS && !UNIT_TESTING
  cout << '\n';
  if (symsCont->getBlanksCount())
    cout << "Removed " << symsCont->getBlanksCount()
         << " Space characters from symsSet!\n";
  if (symsCont->getDuplicatesCount())
    cout << "Removed " << symsCont->getDuplicatesCount()
         << " duplicates from symsSet!\n";

  const auto& removableSymsByCateg = symsCont->getRemovableSymsByCateg();
  for (const auto& [categIdx, categCount] : removableSymsByCateg)
    cout << "Detected " << categCount << ' ' << SymFilter::filterName(categIdx)
         << " in the symsSet!\n";

  cout << "Count of remaining symbols is " << symsCont->getSyms().size()
       << endl;
}
#pragma warning(default : WARN_THROWS_ALTHOUGH_NOEXCEPT)

#pragma warning(disable : WARN_THROWS_ALTHOUGH_NOEXCEPT)
const string& FontEngine::getEncoding(
    unsigned* pEncodingIndex /* = nullptr*/) const noexcept(!UT) {
  EXPECTS_OR_REPORT_AND_THROW_CONST_MSG(
      face, logic_error,
      "Call FontEngine::newFont() before calling "s + HERE.function_name());

  if (pEncodingIndex)
    *pEncodingIndex = encodingIndex;

  return ss->getEncoding();
}
#pragma warning(default : WARN_THROWS_ALTHOUGH_NOEXCEPT)

#pragma warning(disable : WARN_THROWS_ALTHOUGH_NOEXCEPT)
unsigned FontEngine::uniqueEncodings() const noexcept(!UT) {
  EXPECTS_OR_REPORT_AND_THROW_CONST_MSG(
      face, logic_error,
      "Call FontEngine::newFont() before calling "s + HERE.function_name());

  return narrow_cast<unsigned>(size(uniqueEncs));
}
#pragma warning(default : WARN_THROWS_ALTHOUGH_NOEXCEPT)

#pragma warning(disable : WARN_THROWS_ALTHOUGH_NOEXCEPT)
unsigned FontEngine::upperSymsCount() const noexcept(!UT) {
  EXPECTS_OR_REPORT_AND_THROW_CONST_MSG(
      face, logic_error,
      "Call FontEngine::newFont() before calling "s + HERE.function_name());

  return symsCount;
}
#pragma warning(default : WARN_THROWS_ALTHOUGH_NOEXCEPT)

#pragma warning(disable : WARN_THROWS_ALTHOUGH_NOEXCEPT)
const VPixMapSym& FontEngine::symsSet() const noexcept(!UT) {
  EXPECTS_OR_REPORT_AND_THROW_CONST_MSG(
      (face && symsCont->isReady()), logic_error,
      "Select a font and add all its glyphs before calling "s +
          HERE.function_name());

  return symsCont->getSyms();
}
#pragma warning(default : WARN_THROWS_ALTHOUGH_NOEXCEPT)

#pragma warning(disable : WARN_THROWS_ALTHOUGH_NOEXCEPT)
double FontEngine::smallGlyphsCoverage() const noexcept(!UT) {
  EXPECTS_OR_REPORT_AND_THROW_CONST_MSG(
      (face && symsCont->isReady()), logic_error,
      "Select a font and add all its glyphs before calling "s +
          HERE.function_name());

  return symsCont->getCoverageOfSmallGlyphs();
}
#pragma warning(default : WARN_THROWS_ALTHOUGH_NOEXCEPT)

const string& FontEngine::fontFileName() const noexcept {
  // Don't throw if empty; simply denote that the user didn't select a font yet
  return ss->getFontFile();
}

FT_String* FontEngine::getFamily() const noexcept {
  if (face)
    return face->family_name;

  return const_cast<FT_String*>("");
}

FT_String* FontEngine::getStyle() const noexcept {
  if (face)
    return face->style_name;

  return const_cast<FT_String*>("");
}

FontEngine& FontEngine::useSymsMonitor(
    p2s::ui::AbsJobMonitor& symsMonitor_) noexcept {
  symsMonitor = &symsMonitor_;
  return *this;
}

}  // namespace syms
}  // namespace pic2sym
