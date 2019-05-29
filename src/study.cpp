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

#ifndef UNIT_TESTING

#include "misc.h"
#include "study.h"

#pragma warning(push, 0)

#include <chrono>
#include <iostream>
#include <string>

#pragma warning(pop)

using namespace std;
using namespace std::chrono;

extern template class time_point<high_resolution_clock>;

namespace {

/// Simple timer class to measure the performance of some task
class Timer {
 public:
  // No intention to copy / move
  Timer(const Timer&) = delete;
  Timer(Timer&&) = delete;
  void operator=(const Timer&) = delete;
  void operator=(Timer&&) = delete;

  Timer(const string& taskName_ = "") noexcept
      : taskName(taskName_), startedAt(high_resolution_clock::now()) {}
  virtual ~Timer() noexcept {
    const double elapsedS = elapsed();
    cout << "Task " << taskName << " required: " << elapsedS << "s!" << endl;
  }

  /// Returns the time elapse since the Timer was started (created)
  double elapsed() const noexcept {
    const duration<double> elapsedS = high_resolution_clock::now() - startedAt;
    return elapsedS.count();
  }

 private:
  const string taskName;                        ///< name of the monitored task
  time_point<high_resolution_clock> startedAt;  ///< starting moment
};

}  // anonymous namespace

bool prompt(const string& question, const string& context) noexcept {
  cout << question << " in context " << context << "? ([y]/n) ";
  string line;
  getline(cin, line);
  return line.empty() || line == "y" || line == "Y";
}

bool studying() noexcept {
  return false;
}

int study(int argc, char* argv[]) noexcept {
  Timer timer("studying");
  UNREFERENCED_PARAMETER(argc);
  UNREFERENCED_PARAMETER(argv);
  return 0;
}

#endif  // UNIT_TESTING not defined
