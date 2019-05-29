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

#ifndef H_TRANSFORM_BASE
#define H_TRANSFORM_BASE

#include "transformCompletion.h"

class AbsJobMonitor;  // forward declaration

/// Transformer allows images to be approximated as a table of colored symbols
/// from font files.
class ITransformer /*abstract*/ : public ITransformCompletion {
 public:
  /// Transformation duration in seconds
  virtual double duration() const noexcept = 0;

  /**
  Updates symsBatchSz.
  @param symsBatchSz_ the value to set. If 0 is provided, batching symbols
  gets disabled for the rest of the transformation, ignoring any new slider
  positions.
  */
  virtual void setSymsBatchSize(int symsBatchSz_) noexcept = 0;

  /**
  Applies the configured transformation onto current/new image
  @throw logic_error, domain_error only in UnitTesting for incomplete
  configuration

  Exceptions to be caught only in UnitTesting
  */
  virtual void run() noexcept(!UT) = 0;

  /// Setting the transformation monitor
  virtual ITransformer& useTransformMonitor(
      AbsJobMonitor& transformMonitor_) noexcept = 0;

 protected:
  ITransformer() noexcept {}
};

#endif  // H_TRANSFORM_BASE
