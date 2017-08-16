/************************************************************************************************
 The application Pic2Sym approximates images by a
 grid of colored symbols with colored backgrounds.

 Copyrights from the libraries used by the program:
 - (c) 2016 Boost (www.boost.org)
		License: <http://www.boost.org/LICENSE_1_0.txt>
			or doc/licenses/Boost.lic
 - (c) 2015 OpenCV (www.opencv.org)
		License: <http://opencv.org/license.html>
            or doc/licenses/OpenCV.lic
 - (c) 2015 The FreeType Project (www.freetype.org)
		License: <http://git.savannah.gnu.org/cgit/freetype/freetype2.git/plain/docs/FTL.TXT>
	        or doc/licenses/FTL.txt
 - (c) 1997-2002 OpenMP Architecture Review Board (www.openmp.org)
   (c) Microsoft Corporation (Visual C++ implementation for OpenMP C/C++ Version 2.0 March 2002)
		See: <https://msdn.microsoft.com/en-us/library/8y6825x5(v=vs.140).aspx>
 - (c) 1995-2013 zlib software (Jean-loup Gailly and Mark Adler - see: www.zlib.net)
		License: <http://www.zlib.net/zlib_license.html>
            or doc/licenses/zlib.lic
 
 (c) 2016, 2017 Florin Tulba <florintulba@yahoo.com>

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
 ***********************************************************************************************/

#include "testMain.h"
#include "selectBranch.h"
#include "filledRectanglesFilter.h"
#include "gridBarsFilter.h"
#include "bulkySymsFilter.h"
#include "unreadableSymsFilter.h"
#include "sievesSymsFilter.h"
#include "symFilterCache.h"
#include "pixMapSym.h"
#include "symsSerialization.h"
#include "misc.h"

#pragma warning ( push, 0 )

#include <string>
#include <vector>
#include <numeric>
#include <fstream>
#include <sstream>

#include <boost/test/unit_test.hpp>
#include <boost/filesystem/path.hpp>
#include <boost/filesystem/operations.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#pragma warning ( pop )

using namespace std;
using namespace boost::filesystem;
using namespace boost;
using namespace cv;

/*
For each symbol filter type, count:
- True Positives and False Negatives on res/TestSymFilters/<filterType>Positives.txt
- True Negatives and False Positives on res/TestSymFilters/<filterType>Negatives.txt

and report large values of False Negatives/Positives.
*/

extern const bool BulkySymsFilterEnabled;
extern const bool UnreadableSymsFilterEnabled;
extern const bool SievesSymsFilterEnabled;
extern const bool FilledRectanglesFilterEnabled;
extern const bool GridBarsFilterEnabled;

namespace ut {
	/**
	When detecting misfiltered symbols during Unit Testing, it displays a window with them.

	@param testTitle the name of the test producing mismatches.
	It's appended with a unique id to distinguish among homonym tests
	from different unit testing sessions.
	@param misfiltered vector of pointers to misfiltered PixMapSym objects
	*/
	void showMisfiltered(const string &testTitle,
						 const vector<std::shared_ptr<PixMapSym>> &misfiltered);

	/**
	TestSymFilter performs unit testing for symbol filter of class SymFilterType.
	The tests contain 2 types of symbols:
	- positives, which are supposed to be captured by the given filter
	- negatives, which shouldn't be captured by the filter

	Each group is placed in a separated file, but the names of both files share a common prefix,
	while the suffixes are 'Positives.txt' and 'Negatives.txt'.

	The test is successful if only less than 10% of the symbols from each group get miscategorized.
	*/
	template<class SymFilterType>
	struct TestSymFilter {
		/**
		Tests SymFilterType on a single group of symbols.

		@param disposableCateg group type:
			- true for disposable symbols (filtered out positives)
			- false for negatives
		@param pathForCateg full path of one group of symbols (either positives or negatives)
		@param prefixStem prefix of the name of the file, without the folder parts

		@return true (successful) if only less than 10% of the symbols from the group get miscategorized
		*/
		static bool testCategory(const bool disposableCateg,
								 const string &pathForCateg, const string &prefixStem) {
			static const double WrongCategThreshold = .1; // the 10% threshold mentioned above
			const bool filteringEnabled = SymFilterType::isEnabled();

			vector<const Mat> symsToTest;
			loadSymsSelection(pathForCateg, symsToTest);

			const unsigned symsCount = (unsigned)symsToTest.size();
			vector<std::shared_ptr<PixMapSym>> wrongCateg; // keep pointers to miscategorized symbols

			// compute max-span consec and revConsec needed below by SymFilterCache objects
			extern const unsigned Settings_MAX_FONT_SIZE;
			Mat consec(1, Settings_MAX_FONT_SIZE, CV_64FC1), revConsec;
			iota(BOUNDS_FOR_ITEM_TYPE(consec, double), (double)0.);
			flip(consec, revConsec, 1); revConsec = revConsec.t();

			for(unsigned symIdx = 0U; symIdx < symsCount; ++symIdx) {
				const Mat &symMat = symsToTest[symIdx];
				const unsigned rows = (unsigned)symMat.rows, cols = (unsigned)symMat.cols;
				SymFilterCache sfc; sfc.setFontSz(rows); sfc.setBoundingBox(rows, rows);

				const vector<unsigned char> symData(BOUNDS_FOR_ITEM_TYPE(symMat, unsigned char));
				std::shared_ptr<PixMapSym> pms =
					std::make_shared<PixMapSym>(symData,
						consec.colRange(Range(0, rows)), // take only rows values from [rev]consec
						revConsec.rowRange(Range(Settings_MAX_FONT_SIZE-rows, Settings_MAX_FONT_SIZE)));

				const bool isDisposable = filteringEnabled && SymFilterType::isDisposable(*pms, sfc);
				if(disposableCateg != isDisposable)
					wrongCateg.push_back(pms); // found new miscategorized symbol
			}

			if(wrongCateg.empty())
				return true;

			ostringstream oss;
			oss<<prefixStem<<" false "
				<<(disposableCateg ? "negatives" : "positives")
				<<':'<<wrongCateg.size();
			const string misfilteredText = oss.str();
			cerr<<misfilteredText<<endl;

			if(wrongCateg.size() > symsCount * WrongCategThreshold) {
				showMisfiltered(misfilteredText, wrongCateg);
				return false; // too many miscategorized symbols in group
			}

			return true; // less than 10% miscategorized symbols within current group
		}

		/**
		Tests SymFilterType on a both group of symbols (positives and negatives)

		@param pathPrefix shared path part for both groups of symbols

		@return true (successful) if only less than 10% of the symbols from each group get miscategorized
		*/
		static bool against(const path &pathPrefix) {
			const path positives(path(pathPrefix).concat("Positives.txt"));
			const path negatives(path(pathPrefix).concat("Negatives.txt"));
			if(!exists(positives) || !exists(negatives))
				THROW_WITH_VAR_MSG(string("Couldn't find ") + positives.string() + " or "
										+ negatives.string(), invalid_argument);

			const string prefixStem = pathPrefix.stem().string();

			const bool testedPositives = testCategory(true, positives.string(), prefixStem);
			const bool testedNegatives = testCategory(false, negatives.string(), prefixStem);

			return testedPositives && testedNegatives;
		}
	};

	/// Ensuring the desired enabled/disabled state for all the filters while testing them
	template<bool desiredEnabledState>
	struct SymFiltersFixt {
		PixMapSym pms;		///< dummy PixMapSym
		SymFilterCache sfc;	///< dummy SymFilterCache

#define SAVE_PREV_ENABLED_STATE(FilterType) \
		const bool old##FilterType##Enabled = FilterType##Enabled;

		SAVE_PREV_ENABLED_STATE(BulkySymsFilter);
		SAVE_PREV_ENABLED_STATE(UnreadableSymsFilter);
		SAVE_PREV_ENABLED_STATE(SievesSymsFilter);
		SAVE_PREV_ENABLED_STATE(FilledRectanglesFilter);
		SAVE_PREV_ENABLED_STATE(GridBarsFilter);

#undef SAVE_PREV_ENABLED_STATE

		SymFiltersFixt() : 
				pms({ 128U }, Mat::ones(1, 1, CV_64FC1), Mat::ones(1, 1, CV_64FC1)),
				sfc() {

#define ENSURE_ENABLED(FilterType) \
			if(FilterType##Enabled != desiredEnabledState) \
				const_cast<bool&>(FilterType##Enabled) = desiredEnabledState

			ENSURE_ENABLED(BulkySymsFilter);
			ENSURE_ENABLED(UnreadableSymsFilter);
			ENSURE_ENABLED(SievesSymsFilter);
			ENSURE_ENABLED(FilledRectanglesFilter);
			ENSURE_ENABLED(GridBarsFilter);

#undef ENSURE_ENABLED
		}

		~SymFiltersFixt() {
#define RESTORE_PREV_ENABLED_STATE(FilterType) \
			if(old##FilterType##Enabled != FilterType##Enabled) \
				const_cast<bool&>(FilterType##Enabled) = old##FilterType##Enabled

			RESTORE_PREV_ENABLED_STATE(BulkySymsFilter);
			RESTORE_PREV_ENABLED_STATE(UnreadableSymsFilter);
			RESTORE_PREV_ENABLED_STATE(SievesSymsFilter);
			RESTORE_PREV_ENABLED_STATE(FilledRectanglesFilter);
			RESTORE_PREV_ENABLED_STATE(GridBarsFilter);

#undef RESTORE_PREV_ENABLED_STATE
		}
	};

	/// folder containing the test files for symbol filters
	const path testSymFiltersDir("res\\TestSymFilters");
}

using namespace ut;

BOOST_FIXTURE_TEST_SUITE(EnabledSymFilters, SymFiltersFixt<true>)
	AutoTestCase(CheckFilledRectanglesSymFilter_SeveralPositivesAndNegatives_MinFalseLabeling);
		BOOST_REQUIRE(FilledRectanglesFilter::isEnabled());
		BOOST_REQUIRE(TestSymFilter<FilledRectanglesFilter>::against(
			path(testSymFiltersDir).append("filledRectangles")));
	}

	AutoTestCase(CheckGridBarsSymFilter_SeveralPositivesAndNegatives_MinFalseLabeling);
		BOOST_REQUIRE(GridBarsFilter::isEnabled());
		BOOST_REQUIRE(TestSymFilter<GridBarsFilter>::against(
			path(testSymFiltersDir).append("gridBars")));
	}

	AutoTestCase(CheckUnreadableSymFilter_SeveralPositivesAndNegatives_MinFalseLabeling);
		BOOST_REQUIRE(UnreadableSymsFilter::isEnabled());
		BOOST_REQUIRE(TestSymFilter<UnreadableSymsFilter>::against(
			path(testSymFiltersDir).append("unreadable")));
	}

	AutoTestCase(CheckBulkiesSymFilter_SeveralPositivesAndNegatives_MinFalseLabeling);
		BOOST_REQUIRE(BulkySymsFilter::isEnabled());
		BOOST_REQUIRE(TestSymFilter<BulkySymsFilter>::against(
			path(testSymFiltersDir).append("bulky")));
	}

	AutoTestCase(CheckSievesSymFilter_SeveralPositivesAndNegatives_MinFalseLabeling);
		BOOST_REQUIRE(SievesSymsFilter::isEnabled());
		BOOST_REQUIRE(TestSymFilter<SievesSymsFilter>::against(
			path(testSymFiltersDir).append("sieves")));
	}
BOOST_AUTO_TEST_SUITE_END() // EnabledSymFilters

BOOST_FIXTURE_TEST_SUITE(DisabledSymFilters, SymFiltersFixt<false>)
	AutoTestCase(CheckFilledRectanglesSymFilter_Disabled_NoFiltering);
		BOOST_REQUIRE(!FilledRectanglesFilter::isEnabled());
		BOOST_REQUIRE_THROW(FilledRectanglesFilter::isDisposable(pms, sfc), logic_error);
	}

	AutoTestCase(CheckGridBarsSymFilter_Disabled_NoFiltering);
		BOOST_REQUIRE(!GridBarsFilter::isEnabled());
		BOOST_REQUIRE_THROW(GridBarsFilter::isDisposable(pms, sfc), logic_error);
	}

	AutoTestCase(CheckUnreadableSymFilter_Disabled_NoFiltering);
		BOOST_REQUIRE(!UnreadableSymsFilter::isEnabled());
		BOOST_REQUIRE_THROW(UnreadableSymsFilter::isDisposable(pms, sfc), logic_error);
	}

	AutoTestCase(CheckBulkiesSymFilter_Disabled_NoFiltering);
		BOOST_REQUIRE(!BulkySymsFilter::isEnabled());
		BOOST_REQUIRE_THROW(BulkySymsFilter::isDisposable(pms, sfc), logic_error);
	}

	AutoTestCase(CheckSievesSymFilter_Disabled_NoFiltering);
		BOOST_REQUIRE(!SievesSymsFilter::isEnabled());
		BOOST_REQUIRE_THROW(SievesSymsFilter::isDisposable(pms, sfc), logic_error);
	}
BOOST_AUTO_TEST_SUITE_END() // DisabledSymFilters
