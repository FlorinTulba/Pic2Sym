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

// This file is used to instantiate common template classes

#include "precompiled.h"

#include "preselCandidates.h"

#pragma warning(push, 0)

#include <set>
#include <stack>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <ft2build.h>
#include <opencv2/core/core.hpp>
#include FT_FREETYPE_H

#include <chrono>
#include <optional>

#pragma warning(pop)

template class std::vector<int>;
template class std::vector<unsigned>;
template class std::vector<size_t>;
template class std::vector<double>;
template class std::vector<std::string>;
template class std::vector<std::vector<unsigned>>;
template class std::vector<cv::Mat>;

template class std::set<unsigned>;
template class std::set<double>;

template class std::unordered_set<unsigned>;
template class std::unordered_set<FT_ULong>;
template class std::unordered_set<std::string>;
template class std::unordered_set<std::string, std::hash<std::string>>;
template class std::unordered_set<const cv::String*>;
template class std::unordered_set<cv::String, std::hash<std::string>>;

template class std::unordered_map<unsigned, unsigned>;
template class std::unordered_map<std::string, std::string>;
template class std::unordered_map<const cv::String*, bool>;

template class std::stack<CandidateId, std::vector<CandidateId>>;

template class std::optional<unsigned>;
template class std::optional<unsigned long>;
template class std::optional<double>;

template class std::chrono::time_point<std::chrono::high_resolution_clock>;
