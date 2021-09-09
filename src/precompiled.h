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

// For platforms without precompiled header support
// or to temporarily disable this support,
// just define H_PRECOMPILED among the project defined values
#ifndef H_PRECOMPILED
#define H_PRECOMPILED

#pragma warning(push, 0)

#include <algorithm>
#include <array>
#include <atomic>
#include <bitset>
#include <cassert>
#include <chrono>
#include <concepts>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <functional>
#include <future>
#include <iomanip>
#include <iostream>
#include <limits>
#include <list>
#include <map>
#include <memory>
#include <numbers>
#include <numeric>
#include <optional>
#include <queue>
#include <ranges>
#include <regex>
#include <set>
#include <source_location>
#include <span>
#include <sstream>
#include <stack>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <gsl/gsl>

/*
<Windows.h> is needed in several compilation units.

Defining WIN32_LEAN_AND_MEAN below will strip parts from <Windows.h>.

That is ok for the UnitTesting project, but Pic2Sym needs more from <Windows.h>,
for example the OPENFILENAME type in `dlgs.h`.
*/
#ifdef UNIT_TESTING
#define WIN32_LEAN_AND_MEAN
#endif  // UNIT_TESTING defined
#define NOMINMAX
#include <Windows.h>
#include <tchar.h>

#include <omp.h>

#include <boost/algorithm/string/replace.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/bimap/bimap.hpp>
#include <boost/lockfree/queue.hpp>
#include <boost/property_tree/info_parser.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/serialization/array.hpp>
#include <boost/serialization/base_object.hpp>
#include <boost/serialization/split_member.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/version.hpp>

#ifdef UNIT_TESTING
#include <boost/preprocessor/cat.hpp>
#include <boost/preprocessor/iteration/iterate.hpp>

/*
Don't include <boost/test/...> here, because the unit tests depend on the define
BOOST_TEST_MODULE, which needs to be defined:
- before including any <boost/test/...>
- in exactly one unit test source (*.cpp), so not everywhere

This is not possible, since the precompiled header inclusion is mandatory for
every source (*.cpp) and ignores anything preceding its inclusion.

For details, see:
https://www.boost.org/doc/libs/1_70_0/libs/test/doc/html/boost_test/utf_reference/link_references/link_boost_test_module_macro.html
*/
#endif  // UNIT_TESTING defined

#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#ifndef UNIT_TESTING
#include <opencv2/highgui/highgui.hpp>
#endif  // UNIT_TESTING not defined

#include <ft2build.h>
#include FT_FREETYPE_H

#pragma warning(pop)

#include "misc.h"

#ifndef UNIT_TESTING
// 'main' referred from namespaces for projects with precompiled headers
// should be forward declared also in the precompiled header in Visual Studio.
extern int main(int, gsl::zstring<>*);
#endif  // UNIT_TESTING not defined

#endif  // H_PRECOMPILED not defined
