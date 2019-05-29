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

#ifndef H_TEST_MAIN
#define H_TEST_MAIN

#include "bestMatch.h"
#include "match.h"
#include "matchParams.h"

#pragma warning(push, 0)

#include <boost/preprocessor/cat.hpp>
#include <boost/test/unit_test.hpp>

#pragma warning(pop)

/// Defines test case named Name and ensures it will show its name when launched
#define TITLED_AUTO_TEST_CASE(Name) \
  BOOST_AUTO_TEST_CASE(Name) {      \
    BOOST_TEST_MESSAGE("Running " BOOST_PP_STRINGIZE(Name));

/// Allows formatting TITLED_AUTO_TEST_CASE_ content as a block
#define TITLED_AUTO_TEST_CASE_END }

/// unit testing namespace
namespace ut {

/// Generates an uniformly-distributed random unsigned
unsigned randUnifUint();

/**
Generates an uniformly-distributed random unsigned char.

@param minIncl fist possible random value
@param maxIncl last possible random value
@return the random value
*/
unsigned char randUnsignedChar(unsigned char minIncl = 0U,
                               unsigned char maxIncl = 255U);

/// Used for a global fixture to reinitialize Controller's fields for each test
class Controller {
 public:
  /*
  Which Controller's fields to reinitialize.
  The global fixture sets them to true.
  After initialization each is set to false.
  */
  static inline bool initImg{false}, initFontEngine{false},
      initMatchEngine{false}, initTransformer{false}, initPreselManager{false},
      initComparator{false}, initControlPanel{false};
};

/// Mock MatchEngine
class IMatchEngine /*abstract*/ {
 public:
  virtual ~IMatchEngine() noexcept {}

  // Slicing prevention
  IMatchEngine(const IMatchEngine&) = delete;
  IMatchEngine(IMatchEngine&&) = delete;
  IMatchEngine& operator=(const IMatchEngine&) = delete;
  IMatchEngine& operator=(IMatchEngine&&) = delete;

 protected:
  constexpr IMatchEngine() noexcept {}
};
class MatchEngine : public IMatchEngine {};

/// Fixture to be used before every test
class Fixt /*abstract*/ {
 public:
  // If slicing is observed and becomes a severe problem, use `= delete` for all
  Fixt(const Fixt&) noexcept = default;
  Fixt(Fixt&&) noexcept = default;
  Fixt& operator=(const Fixt&) noexcept = default;
  Fixt& operator=(Fixt&&) noexcept = default;

 protected:
  Fixt() noexcept;                     ///< set up
  virtual ~Fixt() noexcept = default;  ///< tear down
};

/**
When detecting mismatches during Unit Testing, it displays a comparator window
with them.

@param testTitle the name of the test producing mismatches.
It's appended with a unique id to distinguish among homonym tests
from different unit testing sessions.
@param mismatches vector of BestMatch objects
*/
void showMismatches(const std::string& testTitle,
                    const std::vector<std::unique_ptr<BestMatch>>& mismatches);
}  // namespace ut

#endif  // H_TEST_MAIN
