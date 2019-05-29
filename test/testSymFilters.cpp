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

#include "bulkySymsFilter.h"
#include "filledRectanglesFilter.h"
#include "gridBarsFilter.h"
#include "misc.h"
#include "pixMapSym.h"
#include "selectBranch.h"
#include "sievesSymsFilter.h"
#include "symFilterCache.h"
#include "symsSerialization.h"
#include "testMain.h"
#include "unreadableSymsFilter.h"

#pragma warning(push, 0)

#include <fstream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include <boost/test/unit_test.hpp>
#include <filesystem>
#include <opencv2/imgproc/imgproc.hpp>

#pragma warning(pop)

using namespace std;
using namespace std::filesystem;
using namespace boost;
using namespace cv;

extern template class vector<Mat>;

/*
For each symbol filter type, count:
- True Positives and False Negatives on
res/TestSymFilters/<filterType>Positives.txt
- True Negatives and False Positives on
res/TestSymFilters/<filterType>Negatives.txt

and report large values of False Negatives/Positives.
*/

extern const bool BulkySymsFilterEnabled;
extern const bool UnreadableSymsFilterEnabled;
extern const bool SievesSymsFilterEnabled;
extern const bool FilledRectanglesFilterEnabled;
extern const bool GridBarsFilterEnabled;

namespace ut {
/**
When detecting misfiltered symbols during Unit Testing, it displays a window
with them.

@param testTitle the name of the test producing mismatches.
It's appended with a unique id to distinguish among homonym tests
from different unit testing sessions.
@param misfiltered vector of pointers to misfiltered PixMapSym objects
*/
void showMisfiltered(
    const string& testTitle,
    const vector<std::unique_ptr<const IPixMapSym>>& misfiltered);

/**
TestSymFilter performs unit testing for symbol filter of class SymFilterType.
The tests contain 2 types of symbols:
- positives, which are supposed to be captured by the given filter
- negatives, which shouldn't be captured by the filter

Each group is placed in a separated file, but the names of both files share a
common prefix, while the suffixes are 'Positives.txt' and 'Negatives.txt'.

The test is successful if only less than 10% of the symbols from each group get
miscategorized.
*/
template <class SymFilterType>
class TestSymFilter {
 public:
  /**
  Tests SymFilterType on a single group of symbols.

  @param disposableCateg group type:
    - true for disposable symbols (filtered out positives)
    - false for negatives
  @param pathForCateg full path of one group of symbols (either positives or
  negatives)
  @param prefixStem prefix of the name of the file, without the folder parts

  @return true (successful) if only less than 10% of the symbols from the group
  get miscategorized
  */
  static bool testCategory(const bool disposableCateg,
                           const string& pathForCateg,
                           const string& prefixStem) noexcept {
    // the 10% threshold mentioned above
    static constexpr double WrongCategThreshold = .1;
    const bool filteringEnabled = SymFilterType::isEnabled();

    vector<Mat> symsToTest;
    loadSymsSelection(pathForCateg, symsToTest);

    const unsigned symsCount = (unsigned)symsToTest.size();

    // Keep pointers to miscategorized symbols
    vector<std::unique_ptr<const IPixMapSym>> wrongCateg;

    // compute max-span consec and revConsec needed below by SymFilterCache
    // objects
    extern const unsigned Settings_MAX_FONT_SIZE;
    Mat consec(1, Settings_MAX_FONT_SIZE, CV_64FC1), revConsec;
    iota(BOUNDS_FOR_ITEM_TYPE(consec, double), (double)0.);
    flip(consec, revConsec, 1);
    revConsec = revConsec.t();

    for (unsigned symIdx = 0U; symIdx < symsCount; ++symIdx) {
      const Mat& symMat = symsToTest[symIdx];
      const unsigned rows = (unsigned)symMat.rows, cols = (unsigned)symMat.cols;
      SymFilterCache sfc;
      sfc.setFontSz(rows);
      sfc.setBoundingBox(rows, rows);

      const vector<unsigned char> symData(
          BOUNDS_FOR_ITEM_TYPE(symMat, unsigned char));
      std::unique_ptr<const PixMapSym> pms = std::make_unique<const PixMapSym>(
          symData,

          // Take only rows values from [rev]consec
          consec.colRange(Range(0, rows)),
          revConsec.rowRange(
              Range(Settings_MAX_FONT_SIZE - rows, Settings_MAX_FONT_SIZE)));

      const bool isDisposable =
          filteringEnabled && SymFilterType::isDisposable(*pms, sfc);
      if (disposableCateg != isDisposable)
        // Found new miscategorized symbol
        wrongCateg.push_back(std::move(pms));
    }

    if (wrongCateg.empty())
      return true;

    ostringstream oss;
    oss << prefixStem << " false "
        << (disposableCateg ? "negatives" : "positives") << ':'
        << wrongCateg.size();
    const string misfilteredText = oss.str();
    cerr << misfilteredText << endl;

    if (wrongCateg.size() > symsCount * WrongCategThreshold) {
      showMisfiltered(misfilteredText, wrongCateg);
      return false;  // too many miscategorized symbols in group
    }

    return true;  // less than 10% miscategorized symbols within current group
  }

  /**
  Tests SymFilterType on a both group of symbols (positives and negatives)

  @param pathPrefix shared path part for both groups of symbols

  @return true (successful) if only less than 10% of the symbols from each group
  get miscategorized
  */
  static bool against(const path& pathPrefix) {
    const path positives(path(pathPrefix).concat("Positives.txt"));
    const path negatives(path(pathPrefix).concat("Negatives.txt"));
    if (!exists(positives) || !exists(negatives))
      THROW_WITH_VAR_MSG(string("Couldn't find ") + positives.string() +
                             " or " + negatives.string(),
                         invalid_argument);

    const string prefixStem = pathPrefix.stem().string();

    const bool testedPositives =
        testCategory(true, positives.string(), prefixStem);
    const bool testedNegatives =
        testCategory(false, negatives.string(), prefixStem);

    return testedPositives && testedNegatives;
  }
};

/// Ensuring the desired enabled/disabled state for all the filters while
/// testing them
template <bool desiredEnabledState>
class SymFiltersFixt {
 public:
  SymFiltersFixt() noexcept
      : pms({128U}, Mat::ones(1, 1, CV_64FC1), Mat::ones(1, 1, CV_64FC1)),
        sfc() {
#define ENSURE_ENABLED(FilterType)                \
  if (FilterType##Enabled != desiredEnabledState) \
  const_cast<bool&>(FilterType##Enabled) = desiredEnabledState

    ENSURE_ENABLED(BulkySymsFilter);
    ENSURE_ENABLED(UnreadableSymsFilter);
    ENSURE_ENABLED(SievesSymsFilter);
    ENSURE_ENABLED(FilledRectanglesFilter);
    ENSURE_ENABLED(GridBarsFilter);

#undef ENSURE_ENABLED
  }

  ~SymFiltersFixt() noexcept {
#define RESTORE_PREV_ENABLED_STATE(FilterType)         \
  if (old##FilterType##Enabled != FilterType##Enabled) \
  const_cast<bool&>(FilterType##Enabled) = old##FilterType##Enabled

    RESTORE_PREV_ENABLED_STATE(BulkySymsFilter);
    RESTORE_PREV_ENABLED_STATE(UnreadableSymsFilter);
    RESTORE_PREV_ENABLED_STATE(SievesSymsFilter);
    RESTORE_PREV_ENABLED_STATE(FilledRectanglesFilter);
    RESTORE_PREV_ENABLED_STATE(GridBarsFilter);

#undef RESTORE_PREV_ENABLED_STATE
  }

  PixMapSym pms;       ///< dummy PixMapSym
  SymFilterCache sfc;  ///< dummy SymFilterCache

#define SAVE_PREV_ENABLED_STATE(FilterType) \
  const bool old##FilterType##Enabled = FilterType##Enabled;

  SAVE_PREV_ENABLED_STATE(BulkySymsFilter);
  SAVE_PREV_ENABLED_STATE(UnreadableSymsFilter);
  SAVE_PREV_ENABLED_STATE(SievesSymsFilter);
  SAVE_PREV_ENABLED_STATE(FilledRectanglesFilter);
  SAVE_PREV_ENABLED_STATE(GridBarsFilter);

#undef SAVE_PREV_ENABLED_STATE
};

/// folder containing the test files for symbol filters
const path testSymFiltersDir("res\\TestSymFilters");
}  // namespace ut

using namespace ut;

BOOST_FIXTURE_TEST_SUITE(EnabledSymFilters, SymFiltersFixt<true>)
TITLED_AUTO_TEST_CASE(
    CheckFilledRectanglesSymFilter_SeveralPositivesAndNegatives_MinFalseLabeling) {
  BOOST_REQUIRE(FilledRectanglesFilter::isEnabled());
  BOOST_REQUIRE(TestSymFilter<FilledRectanglesFilter>::against(
      path(testSymFiltersDir).append("filledRectangles")));
  TITLED_AUTO_TEST_CASE_END
}

TITLED_AUTO_TEST_CASE(
    CheckGridBarsSymFilter_SeveralPositivesAndNegatives_MinFalseLabeling) {
  BOOST_REQUIRE(GridBarsFilter::isEnabled());
  BOOST_REQUIRE(TestSymFilter<GridBarsFilter>::against(
      path(testSymFiltersDir).append("gridBars")));
  TITLED_AUTO_TEST_CASE_END
}

TITLED_AUTO_TEST_CASE(
    CheckUnreadableSymFilter_SeveralPositivesAndNegatives_MinFalseLabeling) {
  BOOST_REQUIRE(UnreadableSymsFilter::isEnabled());
  BOOST_REQUIRE(TestSymFilter<UnreadableSymsFilter>::against(
      path(testSymFiltersDir).append("unreadable")));
  TITLED_AUTO_TEST_CASE_END
}

TITLED_AUTO_TEST_CASE(
    CheckBulkiesSymFilter_SeveralPositivesAndNegatives_MinFalseLabeling) {
  BOOST_REQUIRE(BulkySymsFilter::isEnabled());
  BOOST_REQUIRE(TestSymFilter<BulkySymsFilter>::against(
      path(testSymFiltersDir).append("bulky")));
  TITLED_AUTO_TEST_CASE_END
}

TITLED_AUTO_TEST_CASE(
    CheckSievesSymFilter_SeveralPositivesAndNegatives_MinFalseLabeling) {
  BOOST_REQUIRE(SievesSymsFilter::isEnabled());
  BOOST_REQUIRE(TestSymFilter<SievesSymsFilter>::against(
      path(testSymFiltersDir).append("sieves")));
  TITLED_AUTO_TEST_CASE_END
}
BOOST_AUTO_TEST_SUITE_END()  // EnabledSymFilters

BOOST_FIXTURE_TEST_SUITE(DisabledSymFilters, SymFiltersFixt<false>)
TITLED_AUTO_TEST_CASE(CheckFilledRectanglesSymFilter_Disabled_NoFiltering) {
  BOOST_REQUIRE(!FilledRectanglesFilter::isEnabled());
  BOOST_REQUIRE(!FilledRectanglesFilter::isDisposable(pms, sfc));
  TITLED_AUTO_TEST_CASE_END
}

TITLED_AUTO_TEST_CASE(CheckGridBarsSymFilter_Disabled_NoFiltering) {
  BOOST_REQUIRE(!GridBarsFilter::isEnabled());
  BOOST_REQUIRE(!GridBarsFilter::isDisposable(pms, sfc));
  TITLED_AUTO_TEST_CASE_END
}

TITLED_AUTO_TEST_CASE(CheckUnreadableSymFilter_Disabled_NoFiltering) {
  BOOST_REQUIRE(!UnreadableSymsFilter::isEnabled());
  BOOST_REQUIRE(!UnreadableSymsFilter::isDisposable(pms, sfc));
  TITLED_AUTO_TEST_CASE_END
}

TITLED_AUTO_TEST_CASE(CheckBulkiesSymFilter_Disabled_NoFiltering) {
  BOOST_REQUIRE(!BulkySymsFilter::isEnabled());
  BOOST_REQUIRE(!BulkySymsFilter::isDisposable(pms, sfc));
  TITLED_AUTO_TEST_CASE_END
}

TITLED_AUTO_TEST_CASE(CheckSievesSymFilter_Disabled_NoFiltering) {
  BOOST_REQUIRE(!SievesSymsFilter::isEnabled());
  BOOST_REQUIRE(!SievesSymsFilter::isDisposable(pms, sfc));
  TITLED_AUTO_TEST_CASE_END
}
BOOST_AUTO_TEST_SUITE_END()  // DisabledSymFilters
