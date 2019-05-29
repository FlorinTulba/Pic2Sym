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

#ifndef H_PIC_TRANSFORM_PROGRESS_TRACKER_BASE
#define H_PIC_TRANSFORM_PROGRESS_TRACKER_BASE

#include "timing.h"

/// Interface to monitor the progress of transforming an image.
class IPicTransformProgressTracker /*abstract*/ {
 public:
  /// Called when unable to load the symbols right when attempting to transform
  /// an image
  virtual void transformFailedToStart() noexcept = 0;

  /**
  An hourglass window displays the progress [0..1] of the transformation in %.
  If showDraft is true, and a draft is available, it will be presented within
  Comparator window.
  */
  virtual void reportTransformationProgress(double progress,
                                            bool showDraft = false) const
      noexcept = 0;

  /**
  Present the partial / final result after the transformation has been canceled
  / has finished. When the transformation completes, there'll be a report about
  the duration of the process. Otherwise, completionDurationS will have its
  default negative value and no duration report will be issued.

  @param completionDurationS the duration of the transformation in seconds or a
  negative value for aborted transformations
  */
  virtual void presentTransformationResults(
      double completionDurationS = -1.) const noexcept = 0;

  /// Creates the monitor to time the picture approximation process
  virtual Timer createTimerForImgTransform() const noexcept = 0;

  virtual ~IPicTransformProgressTracker() noexcept {}

  // No intention to copy / move such trackers
  IPicTransformProgressTracker(const IPicTransformProgressTracker&) = delete;
  IPicTransformProgressTracker(IPicTransformProgressTracker&&) = delete;
  IPicTransformProgressTracker& operator=(const IPicTransformProgressTracker&) =
      delete;
  IPicTransformProgressTracker& operator=(IPicTransformProgressTracker&&) =
      delete;

 protected:
  constexpr IPicTransformProgressTracker() noexcept {}
};

#endif  // H_PIC_TRANSFORM_PROGRESS_TRACKER_BASE
