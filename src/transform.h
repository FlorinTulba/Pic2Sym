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

#ifndef H_TRANSFORM
#define H_TRANSFORM

#include "transformBase.h"

#pragma warning(push, 0)

#include <atomic>
#include <memory>
#include <string>
#include <vector>

#include <opencv2/core/core.hpp>

#pragma warning(pop)

#ifndef UNIT_TESTING

extern const unsigned SymsBatch_defaultSz;

#endif  // UNIT_TESTING not defined

// Forward declarations
class ISettings;                     // global settings
class IPicTransformProgressTracker;  // data & views manager
class AbsTaskMonitor;
class IBestMatch;
class IBasicImgData;
class ITransformSupport;
class IMatchEngine;
class IController;

/// Transformer allows images to be approximated as a table of colored symbols
/// from font files.
class Transformer : public ITransformer {
 public:
  Transformer(IController& ctrler_,
              const ISettings& cfg_,
              IMatchEngine& me_,
              IBasicImgData& img_) noexcept;

  const cv::Mat& getResult() const noexcept final { return result; }

  void setDuration(double durationS_) noexcept override {
    durationS = durationS_;
  }

  double duration() const noexcept final { return durationS; }

  /**
  Updates symsBatchSz.
  @param symsBatchSz_ the value to set. If 0 is provided, batching symbols
    gets disabled for the rest of the transformation, ignoring any new slider
  positions.
  */
  void setSymsBatchSize(int symsBatchSz_) noexcept override;

  /**
  Applies the configured transformation onto current/new image
  @throw logic_error, domain_error only in UnitTesting for incomplete
  configuration

  Exceptions to be caught only in UnitTesting
  */
  void run() noexcept(!UT) override;

  /// Setting the transformation monitor
  Transformer& useTransformMonitor(
      AbsJobMonitor& transformMonitor_) noexcept override;

 protected:
  /// Updates the unique id for the studied case
  void updateStudiedCase(int rows, int cols) noexcept;

  /// Makes sure draftMatches will be computed for correct resized img
  void initDraftMatches(bool newResizedImg,
                        const cv::Mat& resizedVersion,
                        unsigned patchesPerCol,
                        unsigned patchesPerRow) noexcept;

  /// Improves the result by analyzing the symbols in range [fromIdx, upperIdx)
  /// under the supervision of imgTransformTaskMonitor
  void considerSymsBatch(unsigned fromIdx,
                         unsigned upperIdx,
                         AbsTaskMonitor& imgTransformTaskMonitor) noexcept;

 private:
  IController& ctrler;  ///< controller

  /// Image transformation management from the controller
  IPicTransformProgressTracker& ptpt;

  /// Observer of the transformation process who reports its progress
  AbsJobMonitor* transformMonitor = nullptr;

  const ISettings& cfg;  ///< general configuration
  IMatchEngine& me;      ///< approximating patches
  IBasicImgData& img;  ///< basic information about the current image to process

  cv::Mat result;  ///< the result of the transformation

  std::string studiedCase;  ///< unique id for the studied case
  cv::Mat resized;          ///< resized version of the original
  cv::Mat resizedBlurred;   ///< blurred version of the resized original

  /// temporary best matches
  std::vector<std::vector<std::unique_ptr<IBestMatch>>> draftMatches;

  // Keep this after previous fields, as it depends on them
  /// Initializes and updates draft matches
  const std::unique_ptr<ITransformSupport> transformSupport;

  double durationS = 0.;  ///< transformation duration in seconds

  int w = 0;                ///< width of the resized image
  int h = 0;                ///< height of the resized image
  unsigned sz = 0U;         ///< font size used during transformation
  unsigned symsCount = 0U;  ///< symbols count within the used cmap

#ifndef UNIT_TESTING  // Start with batching SymsBatch_defaultSz symbols
  /// Runtime control of how large next symbol batches are
  std::atomic_uint symsBatchSz{SymsBatch_defaultSz};

#else  // No batching in Unit Testing mode
  /// Runtime control of how large next symbol batches are
  std::atomic_uint symsBatchSz{UINT_MAX};

#endif  // UNIT_TESTING

  std::atomic_bool isCanceled{false};  ///< has the process been canceled?
};

#endif  // H_TRANSFORM
