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

#include "controlPanelActions.h"
#include "controllerBase.h"
#include "picTransformProgressTracker.h"

using namespace std;

namespace {  // Anonymous namespace
/// Actions for start & stop chronometer while timing the approximation of the
/// picture
class TimerActions : public ITimerActions {
 public:
  explicit TimerActions(IPicTransformProgressTracker& ptpt_) noexcept
      : ptpt(ptpt_) {}

  /// Action to be performed when the timer is started
  void onStart() noexcept override { ptpt.reportTransformationProgress(0.); }

  /// action to be performed when the timer is released/deleted
  /// @param elapsedS total elapsed time in seconds
  void onRelease(double elapsedS) noexcept override {
    ptpt.reportTransformationProgress(1.);
    ptpt.presentTransformationResults(elapsedS);
  }

  /// action to be performed when the timer is canceled
  /// @param reason explanation for cancellation
  void onCancel(const std::string& reason = "") noexcept override {
    ptpt.reportTransformationProgress(1., true);
    infoMsg(reason);
  }

 private:
  IPicTransformProgressTracker& ptpt;
};
}  // Anonymous namespace

PicTransformProgressTracker::PicTransformProgressTracker(
    IController& ctrler_) noexcept
    : ctrler(ctrler_) {}

Timer PicTransformProgressTracker::createTimerForImgTransform() const noexcept {
  return Timer(
      std::make_shared<TimerActions>(ctrler.getPicTransformProgressTracker()));
}

#ifndef UNIT_TESTING

void PicTransformProgressTracker::transformFailedToStart() noexcept {
  ctrler.getControlPanelActions().invalidateFont();
}

void PicTransformProgressTracker::reportTransformationProgress(
    double progress,
    bool showDraft /* = false*/) const noexcept {
  extern const string Controller_PREFIX_TRANSFORMATION_PROGRESS;
  ctrler.hourGlass(progress, Controller_PREFIX_TRANSFORMATION_PROGRESS);

  if (showDraft)
    presentTransformationResults();
}

void PicTransformProgressTracker::presentTransformationResults(
    double completionDurationS /* = -1.*/) const noexcept {
  ctrler.showResultedImage(completionDurationS);
}

#endif  // UNIT_TESTING
