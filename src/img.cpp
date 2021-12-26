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

#include "img.h"

#include "imgSettings.h"
#include "warnings.h"

#pragma warning(push, 0)

#include <filesystem>
#include <iostream>

#include <opencv2/imgcodecs/imgcodecs.hpp>

#pragma warning(pop)

using namespace std;
using namespace std::filesystem;
using namespace cv;

namespace pic2sym {
namespace input {

bool Img::reset(const Mat& source_) noexcept {
  if (source_.empty())
    return false;

  source = source_;
  color = source.channels() > 1;
  return true;
}

bool Img::reset(const string& picName) noexcept {
  {
    path newPic{absolute(picName)};
    if (imgPath == newPic)
      return true;  // image already in use

    const Mat source_{imread(picName, ImreadModes::IMREAD_UNCHANGED)};
    if (!reset(source_)) {
      cerr << "Couldn't read image " << picName << endl;
      return false;
    }

    imgPath = move(newPic);
  }

  imgName = imgPath.stem().string();

  cout << "The image to process is '" << imgPath.string() << "' (";
  if (color)
    cout << "Color";
  else
    cout << "Grayscale";
  cout << " w=" << source.cols << "; h=" << source.rows << ")\n" << endl;
  return true;
}

}  // namespace input

namespace cfg {

void ImgSettings::setMaxHSyms(unsigned syms) noexcept {
  if (syms == hMaxSyms)
    return;
  cout << "hMaxSyms"
       << " : " << hMaxSyms << " -> " << syms << endl;
  hMaxSyms = syms;
}

void ImgSettings::setMaxVSyms(unsigned syms) noexcept {
  if (syms == vMaxSyms)
    return;
  cout << "vMaxSyms"
       << " : " << vMaxSyms << " -> " << syms << endl;
  vMaxSyms = syms;
}

unique_ptr<IfImgSettings> ImgSettings::clone() const noexcept {
  return make_unique<ImgSettings>(*this);
}

#pragma warning(disable : WARN_EXPR_ALWAYS_FALSE)
bool ImgSettings::olderVersionDuringLastIO() noexcept {
  return VersionFromLast_IO_op < Version;
}
#pragma warning(default : WARN_EXPR_ALWAYS_FALSE)

}  // namespace cfg
}  // namespace pic2sym
