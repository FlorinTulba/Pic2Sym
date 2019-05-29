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

#include "controllerBase.h"
#include "fontEngine.h"
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
#include "tinySym.h"
#include "updateSymSettingsBase.h"

#pragma warning(push, 0)

#include <algorithm>
#include <sstream>

#include <filesystem>
#include FT_TRUETYPE_IDS_H

#pragma warning(pop)

using namespace std;
using namespace cv;
using namespace std::filesystem;
using namespace boost::bimaps;

extern const bool ParallelizePixMapStatistics;

namespace {
/// Creates a bimap using initializer_list. Needed in 'encodingsMap' below
template <typename L, typename R>
bimap<L, R> make_bimap(
    initializer_list<typename bimap<L, R>::value_type> il) noexcept {
  return bimap<L, R>(CBOUNDS(il));
}

/**
@return mapping between encodings codes and their corresponding names.
It's non-const just to allow accessing the map with operator[].
*/
const bimap<FT_Encoding, string>& encodingsMap() noexcept {
  // Defines pairs like { FT_ENCODING_ADOBE_STANDARD, "ADOBE_STANDARD" }
#define enc(encValue) \
  { encValue, string(#encValue).substr(12) }

  static const bimap<FT_Encoding, string> encMap(
      make_bimap<FT_Encoding, string>(
          {// known encodings
           enc(FT_ENCODING_NONE), enc(FT_ENCODING_UNICODE),
           enc(FT_ENCODING_MS_SYMBOL), enc(FT_ENCODING_ADOBE_LATIN_1),
           enc(FT_ENCODING_OLD_LATIN_2), enc(FT_ENCODING_SJIS),
           enc(FT_ENCODING_GB2312), enc(FT_ENCODING_BIG5),
           enc(FT_ENCODING_WANSUNG), enc(FT_ENCODING_JOHAB),
           enc(FT_ENCODING_ADOBE_STANDARD), enc(FT_ENCODING_ADOBE_EXPERT),
           enc(FT_ENCODING_ADOBE_CUSTOM), enc(FT_ENCODING_APPLE_ROMAN)}));
#undef enc

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
FontEngine::FontEngine(IController& ctrler_, const ISymSettings& ss_) noexcept
    : symSettingsUpdater(ctrler_.getUpdateSymSettings()),
      cmapPresenter(ctrler_.getPresentCmap()),
      ss(ss_),
      symsCont(make_unique<PmsCont>(ctrler_)) {
  if (const FT_Error error = FT_Init_FreeType(&library); error != FT_Err_Ok)
    THROW_WITH_VAR_MSG(
        "Couldn't initialize FreeType! Error: " + FtErrors[(size_t)error],
        runtime_error);
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

bool FontEngine::checkFontFile(const path& fontPath, FT_Face& face_) const
    noexcept {
  if (!exists(fontPath)) {
    cerr << "No such file: " << fontPath << endl;
    return false;
  }

  if (const FT_Error error =
          FT_New_Face(library, fontPath.string().c_str(), 0, &face_);
      error != FT_Err_Ok) {
    cerr << "Invalid font file: " << fontPath
         << "  Error: " << FtErrors[(size_t)error] << endl;
    return false;
  }
  /*
  // Some faces not providing this flag 'squeeze' basic ASCII characters to the
  // left of the square

  if (!FT_IS_FIXED_WIDTH(face_)) {
    cerr << "The font file " << fontPath
         << " isn't a fixed-width (monospace) font! Flags: 0x" << hex
         << face_->face_flags << dec << endl;
    return false;
  }
  */

  if (!FT_IS_SCALABLE(face_)) {
    cerr << "The font file " << fontPath << " isn't a scalable font!" << endl;
    return false;
  }

  return true;
}

#pragma warning(disable : WARN_THROWS_ALTHOUGH_NOEXCEPT)
bool FontEngine::setNthUniqueEncoding(unsigned idx) noexcept(!UT) {
  if (face == nullptr)
    THROW_WITH_CONST_MSG(
        "No Font yet! Please select one first and then call " __FUNCTION__ "!",
        logic_error);

  if (idx == encodingIndex)
    return true;  // same encoding

  if (idx >= uniqueEncodings())
    return false;

  if (const FT_Error error = FT_Set_Charmap(
          face, face->charmaps[next(uniqueEncs.right.begin(), idx)->first]);
      error != FT_Err_Ok) {
    cerr << "Couldn't set new cmap! Error: " << FtErrors[(size_t)error] << endl;
    return false;
  }

  encodingIndex = idx;
  const string& encName =
      encodingsMap().left.find(face->charmap->encoding)->second;
  cout << "Using encoding " << encName << " (index " << encodingIndex << ')'
       << endl;

  tinySyms.clear();
  symsCont->reset();
  symsCount = 0U;
  symsUnableToLoad.clear();

  symSettingsUpdater.newFontEncoding(encName);

  return true;
}
#pragma warning(default : WARN_THROWS_ALTHOUGH_NOEXCEPT)

#pragma warning(disable : WARN_THROWS_ALTHOUGH_NOEXCEPT)
bool FontEngine::setEncoding(const string& encName,
                             bool forceUpdate /* = false*/) noexcept(!UT) {
  if (face == nullptr)
    THROW_WITH_CONST_MSG(
        "No Font yet! Please select one first and then call " __FUNCTION__ "!",
        logic_error);

  if (encName == ss.getEncoding() && !forceUpdate)
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
  if (face_ == nullptr)
    THROW_WITH_CONST_MSG(
        "Can't provide a NULL face_ as parameter in " __FUNCTION__ "!",
        invalid_argument);

  if (face != nullptr) {
    if (strcmp(face->family_name, face_->family_name) == 0 &&
        strcmp(face->style_name, face_->style_name) == 0)
      return;  // same face

    FT_Done_Face(face);
  }

  tinySyms.clear();
  symsCont->reset();
  symsCount = 0U;
  symsUnableToLoad.clear();
  uniqueEncs.clear();
  face = face_;

  cout << "Using " << face->family_name << ' ' << face->style_name << endl;

  for (int i = 0, charmapsCount = face->num_charmaps; i < charmapsCount; ++i)
    uniqueEncs.insert(bimap<FT_Encoding, unsigned>::value_type(
        face->charmaps[(size_t)i]->encoding, (unsigned)i));

  cout << "The available encodings are:";
  for (const auto& enc : uniqueEncs.right)
    cout << ' ' << encodingsMap().left.find(enc.second)->second;
  cout << endl;

  encodingIndex = UINT_MAX;
  setNthUniqueEncoding(0U);
}
#pragma warning(default : WARN_THROWS_ALTHOUGH_NOEXCEPT)

bool FontEngine::newFont(const string& fontFile_) noexcept {
  FT_Face face_;
  const path fontPath(absolute(fontFile_));
  if (!checkFontFile(fontPath, face_))
    return false;

  setFace(face_, fontPath.string());

  symSettingsUpdater.newFontFile(fontFile_);

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
  FT_Error error = FT_Err_Ok;
  if (error = FT_Request_Size(face, &req); error != FT_Err_Ok)
    THROW_WITH_VAR_MSG("Couldn't set font size: " + to_string(sz) +
                           "  Error: " + FtErrors[(size_t)error],
                       invalid_argument);
  symsUnableToLoad.clear();
  FT_UInt idx;
  for (FT_ULong c = FT_Get_First_Char(face, &idx); idx != 0;
       c = FT_Get_Next_Char(face, c, &idx)) {
    if (error = FT_Load_Char(face, c, FT_LOAD_RENDER); error != FT_Err_Ok) {
      cerr << "Couldn't load glyph " << c
           << " before resizing. Error: " << FtErrors[(size_t)error] << endl;
      symsUnableToLoad.insert(c);
      continue;
    }
    const FT_GlyphSlotRec_* const g = face->glyph;
    const FT_Bitmap b = g->bitmap;

    const unsigned height = b.rows, width = b.width;
    vHeight.push_back(height);
    vWidth.push_back(width);

    const int left = g->bitmap_left, right = left + (int)width - 1,
              top = g->bitmap_top, bottom = top - (int)height + 1;
    vLeft.push_back(left);
    vRight.push_back(right);
    vTop.push_back(top);
    vBottom.push_back(bottom);
  }
  symsCount = (unsigned)vTop.size();

  // Compute some means and standard deviations
  Vec<double, 1> avgTop, sdTop, avgBottom, sdBottom, avgLeft, sdLeft, avgRight,
      sdRight;
  Scalar avgHeight, avgWidth;
#pragma omp parallel if (ParallelizePixMapStatistics)
#pragma omp sections nowait
  {
#pragma omp section
    {
      ompPrintf(ParallelizePixMapStatistics, "height");
      avgHeight = mean(Mat(1, (int)symsCount, CV_64FC1, vHeight.data()));
    }
#pragma omp section
    {
      ompPrintf(ParallelizePixMapStatistics, "width");
      avgWidth = mean(Mat(1, (int)symsCount, CV_64FC1, vWidth.data()));
    }
#pragma omp section
    {
      ompPrintf(ParallelizePixMapStatistics, "top");
      meanStdDev(Mat(1, (int)symsCount, CV_64FC1, vTop.data()), avgTop, sdTop);
    }
#pragma omp section
    {
      ompPrintf(ParallelizePixMapStatistics, "bottom");
      meanStdDev(Mat(1, (int)symsCount, CV_64FC1, vBottom.data()), avgBottom,
                 sdBottom);
    }
#pragma omp section
    {
      ompPrintf(ParallelizePixMapStatistics, "left");
      meanStdDev(Mat(1, (int)symsCount, CV_64FC1, vLeft.data()), avgLeft,
                 sdLeft);
    }
#pragma omp section
    {
      ompPrintf(ParallelizePixMapStatistics, "right");
      meanStdDev(Mat(1, (int)symsCount, CV_64FC1, vRight.data()), avgRight,
                 sdRight);
    }
  }

  static constexpr double
      kv = 1.,
      kh = 1.;  // 1. means a single standard deviation => ~68% of the data

  // Enlarge factors, forcing the average width + lateral std. devs
  // to fit the width of the drawing square.
  factorH = sz / (*avgWidth.val + kh * (*sdLeft.val + *sdRight.val));
  factorV = sz / (*avgHeight.val + kv * (*sdTop.val + *sdBottom.val));

  // Computing new height & width
  req.height = (FT_Long)floor(factorV * req.height);
  req.width = (FT_Long)floor(factorH * req.width);

  // Reshaping the fonts to better fill the drawing square
  if (error = FT_Request_Size(face, &req); error != FT_Err_Ok)
    THROW_WITH_VAR_MSG("Couldn't set font size: " + to_string(sz) +
                           "  Error: " + FtErrors[(size_t)error],
                       invalid_argument);

  // Positioning the Bounding box to best cover the estimated future position &
  // size of the symbols
  double yMin = factorV * (*avgBottom.val -
                           *sdBottom.val),  // current bottom scaled by factorV
      yMax = factorV * (*avgTop.val + *sdTop.val),  // top
      yDiff2 = (yMax - yMin + 1 - sz) /
               2.,  // the difference to divide equally between top & bottom
      xMin = factorH *
             (*avgLeft.val - *sdLeft.val),  // current left scaled by factorH
      xMax = factorH * (*avgRight.val + *sdRight.val);  // right
  const double xDiff2 =
      (xMax - xMin + 1 - sz) /
      2.;  // the difference to divide equally between left & right

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

#pragma warning(disable : WARN_THROWS_ALTHOUGH_NOEXCEPT)
void FontEngine::setFontSz(unsigned fontSz_) noexcept(!UT) {
  if (face == nullptr)
    THROW_WITH_CONST_MSG(
        "Please use FontEngine::newFont before calling " __FUNCTION__ "!",
        logic_error);

  if (nullptr == symsMonitor)
    THROW_WITH_CONST_MSG(
        "Please use FontEngine::setSymsMonitor before calling " __FUNCTION__ "!",
        logic_error);

  if (!ISettings::isFontSizeOk(fontSz_))
    THROW_WITH_VAR_MSG(
        "Invalid font size (" + to_string(fontSz_) + ") for " __FUNCTION__ "!",
        invalid_argument);

  if (symsCont->isReady() && symsCont->getFontSz() == fontSz_)
    return;  // same font size

  cout << "Setting font size " << fontSz_ << endl;

  const double sz = fontSz_;
  vector<DataForSymToResize> toResize;
  double factorH = 0., factorV = 0.;
  FT_BBox bb{0LL, 0LL, 0LL, 0LL};
  FT_UInt idx = 0U;
  FT_Size_RequestRec req;
  req.type = FT_SIZE_REQUEST_TYPE_REAL_DIM;
  req.horiResolution = req.vertResolution = 72U;

  static TaskMonitor determineOptimalSquareFittingSymbols(
      "determine optimal square-fitting "
      "for the symbols",
      *symsMonitor);

  adjustScaling(fontSz_, bb, factorH, factorV);
  determineOptimalSquareFittingSymbols
      .taskDone();  // mark it as already finished

  static TaskMonitor loadFitSymbols("load & filter symbols that fit the square",
                                    *symsMonitor);

  loadFitSymbols.setTotalSteps((size_t)symsCount);
  symsCont->reset(fontSz_, symsCount);

  cmapPresenter->showUnofficialSymDetails(symsCount);

  FT_Error error = FT_Err_Ok;
  SymFilterCache sfc;
  sfc.setFontSz(fontSz_);
  // Store the pixmaps of the symbols that fit the bounding box already or by
  // shifting. Preserve the symbols that don't fit, in order to resize them
  // first, then add them too to pixmaps.
  size_t i = 0ULL, countOfSymsUnableToLoad = 0ULL;
  for (FT_ULong c = FT_Get_First_Char(face, &idx); idx != 0;
       c = FT_Get_Next_Char(face, c, &idx), loadFitSymbols.taskAdvanced(++i)) {
    if (error = FT_Load_Char(face, c, FT_LOAD_RENDER); error != FT_Err_Ok) {
      if (symsUnableToLoad.find(c) != symsUnableToLoad.end()) {  // known glyph
        ++countOfSymsUnableToLoad;
        continue;
      } else  // unexpected glyph
        THROW_WITH_VAR_MSG(
            "Couldn't load an unexpected glyph (" + to_string(c) +
                ") during initial resizing. Error: " + FtErrors[(size_t)error],
            runtime_error);
    }
    const FT_GlyphSlot g = face->glyph;
    const FT_Bitmap b = g->bitmap;
    const unsigned height = b.rows, width = b.width;
    if (width > fontSz_ || height > fontSz_)
      toResize.push_back({c, i, max(1., width / sz), max(1., height / sz)});
    else
      symsCont->appendSym(c, i, g, bb, sfc);
  }

  if (countOfSymsUnableToLoad < symsUnableToLoad.size())
    THROW_WITH_VAR_MSG("Initial resizing of the glyphs found only " +
                           to_string(countOfSymsUnableToLoad) +
                           " symbols that couldn't be loaded when expecting " +
                           to_string(symsUnableToLoad.size()),
                       runtime_error);

  loadFitSymbols.taskDone();

  // Resize symbols which didn't fit initially
  static TaskMonitor loadExtraSqueezedSymbols(
      "load & filter extra-squeezed symbols", *symsMonitor);

  loadExtraSqueezedSymbols.setTotalSteps(toResize.size());

  const FT_Long fontSzMul64 = (FT_Long)(fontSz_) << 6;
  const double numeratorV = factorV * fontSzMul64,
               numeratorH = factorH * fontSzMul64;
  i = 0U;
  for (const DataForSymToResize& item : toResize) {
    req.height = (FT_Long)floor(numeratorV / item.vRatio);
    req.width = (FT_Long)floor(numeratorH / item.hRatio);
    if (error = FT_Request_Size(face, &req); error != FT_Err_Ok)
      THROW_WITH_VAR_MSG("Couldn't set font size: " +
                             to_string(factorV * fontSz_ / item.vRatio) +
                             " x " +
                             to_string(factorH * fontSz_ / item.hRatio) +
                             "  Error: " + FtErrors[(size_t)error],
                         invalid_argument);

    if (error = FT_Load_Char(face, item.symCode, FT_LOAD_RENDER);
        error != FT_Err_Ok)
      THROW_WITH_VAR_MSG(
          "Couldn't load glyph " + to_string(item.symCode) +
              " which needed resizing twice. Error: " + FtErrors[(size_t)error],
          runtime_error);
    symsCont->appendSym(item.symCode, item.symIdx, face->glyph, bb, sfc);

    loadExtraSqueezedSymbols.taskAdvanced(++i);
  }

  // Determine coverageOfSmallGlyphs
  static TaskMonitor determineCoverageOfSmallGlyphs(
      "determine coverageOfSmallGlyphs", *symsMonitor);

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
    cout << toResize.size() << " symbols were resized twice: ";
    for (const DataForSymToResize& item : toResize)
      cout << item.symCode << ", ";
    cout << '\n';
  }

  cout << '\n';
#endif  // VIEW_CONCLUSIONS_FROM_RESHAPING_LOADED_FONTS && !UNIT_TESTING
  cout << '\n';
  if (symsCont->getBlanksCount() != 0U)
    cout << "Removed " << symsCont->getBlanksCount()
         << " Space characters from symsSet!\n";
  if (symsCont->getDuplicatesCount() != 0U)
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
  if (face == nullptr)
    THROW_WITH_CONST_MSG(
        __FUNCTION__ " called before font configuration completion.",
        logic_error);

  if (pEncodingIndex != nullptr)
    *pEncodingIndex = encodingIndex;

  return ss.getEncoding();
}
#pragma warning(default : WARN_THROWS_ALTHOUGH_NOEXCEPT)

#pragma warning(disable : WARN_THROWS_ALTHOUGH_NOEXCEPT)
unsigned FontEngine::uniqueEncodings() const noexcept(!UT) {
  if (face == nullptr)
    THROW_WITH_CONST_MSG(__FUNCTION__ " called before selecting a font.",
                         logic_error);

  return (unsigned)uniqueEncs.size();
}
#pragma warning(default : WARN_THROWS_ALTHOUGH_NOEXCEPT)

#pragma warning(disable : WARN_THROWS_ALTHOUGH_NOEXCEPT)
unsigned FontEngine::upperSymsCount() const noexcept(!UT) {
  if (face == nullptr)
    THROW_WITH_CONST_MSG(__FUNCTION__ " called before selecting a font.",
                         logic_error);

  return symsCount;
}
#pragma warning(default : WARN_THROWS_ALTHOUGH_NOEXCEPT)

#pragma warning(disable : WARN_THROWS_ALTHOUGH_NOEXCEPT)
const VPixMapSym& FontEngine::symsSet() const noexcept(!UT) {
  if (face == nullptr || !symsCont->isReady())
    THROW_WITH_CONST_MSG(__FUNCTION__ " called before selecting a font "
                         "or before finishing adding all its glyphs.",
                         logic_error);

  return symsCont->getSyms();
}
#pragma warning(default : WARN_THROWS_ALTHOUGH_NOEXCEPT)

#pragma warning(disable : WARN_THROWS_ALTHOUGH_NOEXCEPT)
double FontEngine::smallGlyphsCoverage() const noexcept(!UT) {
  if (face == nullptr || !symsCont->isReady())
    THROW_WITH_CONST_MSG(__FUNCTION__ " called before selecting a font "
                         "or before finishing adding all its glyphs.",
                         logic_error);

  return symsCont->getCoverageOfSmallGlyphs();
}
#pragma warning(default : WARN_THROWS_ALTHOUGH_NOEXCEPT)

const string& FontEngine::fontFileName() const noexcept {
  // Don't throw if empty; simply denote that the user didn't select a font yet
  return ss.getFontFile();
}

FT_String* FontEngine::getFamily() const noexcept {
  if (face != nullptr)
    return face->family_name;

  return "";
}

FT_String* FontEngine::getStyle() const noexcept {
  if (face != nullptr)
    return face->style_name;

  return "";
}

FontEngine& FontEngine::useSymsMonitor(AbsJobMonitor& symsMonitor_) noexcept {
  symsMonitor = &symsMonitor_;
  return *this;
}
