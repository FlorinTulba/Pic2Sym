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

#ifndef H_TRANSFORM
#define H_TRANSFORM

#include "transformBase.h"

#include "bestMatchBase.h"
#include "controllerBase.h"
#include "imgBasicData.h"
#include "matchEngineBase.h"
#include "picTransformProgressTrackerBase.h"
#include "settingsBase.h"
#include "taskMonitorBase.h"
#include "transformSupportBase.h"

#pragma warning(push, 0)

#include <atomic>
#include <future>
#include <memory>
#include <string>
#include <vector>

#include <opencv2/core/core.hpp>

#pragma warning(pop)

namespace pic2sym {

#ifndef UNIT_TESTING

extern const unsigned SymsBatch_defaultSz;

#endif  // UNIT_TESTING not defined

namespace transform {

/// Transformer allows images to be approximated as a table of colored symbols
/// from font files.
class Transformer : public ITransformer {
 public:
  Transformer(IController& ctrler_,
              const cfg::ISettings& cfg_,
              match::IMatchEngine& me_,
              input::IBasicImgData& img_) noexcept;

  // Slicing prevention
  Transformer(const Transformer&) = delete;
  Transformer(Transformer&&) = delete;
  void operator=(const Transformer&) = delete;
  void operator=(Transformer&&) = delete;

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

  Exceptions from above to be caught only in UnitTesting

  @throw AbortedJob if the user aborts the operation.
  This exception needs to be handled by the caller.
  */
  void run() override;

  /// Setting the transformation monitor
  Transformer& useTransformMonitor(
      ui::AbsJobMonitor& transformMonitor_) noexcept override;

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
                         ui::AbsTaskMonitor& imgTransformTaskMonitor,
                         std::future<void>& abortHandler) noexcept;

 private:
  gsl::not_null<IController*> ctrler;  ///< controller

  /// Image transformation management from the controller
  gsl::not_null<IPicTransformProgressTracker*> ptpt;

  /// Observer of the transformation process who reports its progress
  ui::AbsJobMonitor* transformMonitor = nullptr;

  gsl::not_null<const cfg::ISettings*> cfg;  ///< general configuration
  gsl::not_null<match::IMatchEngine*> me;    ///< approximating patches
  gsl::not_null<input::IBasicImgData*>
      img;  ///< basic information about the current image to process

  cv::Mat result;  ///< the result of the transformation

  std::string studiedCase;  ///< unique id for the studied case
  cv::Mat resized;          ///< resized version of the original
  cv::Mat resizedBlurred;   ///< blurred version of the resized original

  /// temporary best matches
  std::vector<std::vector<std::unique_ptr<match::IBestMatch>>> draftMatches;

  // Keep this after previous fields, as it depends on them
  /// Initializes and updates draft matches
  std::unique_ptr<ITransformSupport> transformSupport;

  double durationS{};  ///< transformation duration in seconds

  int w{};               ///< width of the resized image
  int h{};               ///< height of the resized image
  unsigned sz{};         ///< font size used during transformation
  unsigned symsCount{};  ///< symbols count within the used cmap

#ifndef UNIT_TESTING  // Start with batching SymsBatch_defaultSz symbols
  /// Runtime control of how large next symbol batches are
  std::atomic_uint symsBatchSz{SymsBatch_defaultSz};

#else  // No batching in Unit Testing mode
  /// Runtime control of how large next symbol batches are
  std::atomic_uint symsBatchSz{UINT_MAX};

#endif  // UNIT_TESTING

  std::atomic_flag isCanceled{};  ///< has the process been canceled?
};

}  // namespace transform
}  // namespace pic2sym

#endif  // H_TRANSFORM
