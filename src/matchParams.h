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

#ifndef H_MATCH_PARAMS
#define H_MATCH_PARAMS

#include "matchParamsBase.h"

#include "misc.h"

namespace pic2sym::match {

/// Holds relevant data during patch&glyph matching
class MatchParams : public IMatchParamsRW {
 public:
  // These params are computed only once, if necessary, when approximating the
  // patch
  /// Mass center for the patch (range 0..1 x 0..1)
  const std::optional<cv::Point2d>& getMcPatch() const noexcept final;
#ifdef UNIT_TESTING
  /// Sum of the values of the pixels of the patch
  const std::optional<double>& getPatchSum() const noexcept final;

  /// Element-wise product patch * patch
  const std::optional<cv::Mat>& getPatchSq() const noexcept final;

  /// Norm L2 of (patch - miu patch)
  const std::optional<double>& getNormPatchMinMiu() const noexcept final;

  /// Blurred version of the patch
  const std::optional<cv::Mat>& getBlurredPatch() const noexcept final;

  /// BlurredPatch element-wise squared
  const std::optional<cv::Mat>& getBlurredPatchSq() const noexcept final;

  /// Blur(patch^2) - blurredPatchSq
  const std::optional<cv::Mat>& getVariancePatch() const noexcept final;

  // These params are evaluated for each symbol compared to the patch
  /// Patch approximated by a given symbol
  const std::optional<cv::Mat>& getPatchApprox() const noexcept final;
#endif  // UNIT_TESTING defined

  /// Mass center for the approximation of the patch (range 0..1 x 0..1)
  const std::optional<cv::Point2d>& getMcPatchApprox() const noexcept final;

  /// Distance between the 2 mass centers (range 0..sqrt(2))
  const std::optional<double>& getMcsOffset() const noexcept final;

  /// % of the box covered by the glyph (0..1)
  const std::optional<double>& getSymDensity() const noexcept final;

#if defined(_DEBUG) || defined(UNIT_TESTING)
  /// Color for fg (range 0..255)
  const std::optional<double>& getFg() const noexcept final;

  /// Provides a representation of the parameters
  std::wstring toWstring() const noexcept override;
#endif  // defined(_DEBUG) || defined(UNIT_TESTING)

  /// Color for bg (range 0..255)
  const std::optional<double>& getBg() const noexcept final;

  /// fg - bg (range -255..255)
  const std::optional<double>& getContrast() const noexcept final;

  /// Structural similarity (-1..1)
  const std::optional<double>& getSsim() const noexcept final;

  /// Absolute value of correlation (0..1)
  const std::optional<double>& getAbsCorr() const noexcept final;

  // ideal value for the standard deviations below is 0
  /// Standard deviation for fg (0..127.5)
  const std::optional<double>& getSdevFg() const noexcept final;

  /// Standard deviation for bg  (0..127.5)
  const std::optional<double>& getSdevBg() const noexcept final;

  /// Standard deviation for contour (0..255)
  const std::optional<double>& getSdevEdge() const noexcept final;

#ifdef UNIT_TESTING
  /// @return a copy of itself
  std::unique_ptr<IMatchParamsRW> clone() const noexcept override;
#endif  // UNIT_TESTING defined

  /**
  Prepares for next symbol to match against patch.

  When skipPatchInvariantParts = true resets everything except:
  mcPatch, patchSum, normPatchMinMiu, patchSq, blurredPatch, blurredPatchSq and
  variancePatch.
  */
  MatchParams& reset(bool skipPatchInvariantParts = true) noexcept override;

  // Methods for computing each field
  void computeFg(const cv::Mat& patch,
                 const syms::ISymData& symData) noexcept override;
  void computeBg(const cv::Mat& patch,
                 const syms::ISymData& symData) noexcept override;
  void computeContrast(const cv::Mat& patch,
                       const syms::ISymData& symData) noexcept override;
  void computeSdevFg(const cv::Mat& patch,
                     const syms::ISymData& symData) noexcept override;
  void computeSdevBg(const cv::Mat& patch,
                     const syms::ISymData& symData) noexcept override;
  void computeSdevEdge(const cv::Mat& patch,
                       const syms::ISymData& symData) noexcept override;
  void computeSymDensity(const syms::ISymData& symData) noexcept override;
  void computePatchSum(const cv::Mat& patch) noexcept override;
  void computePatchSq(const cv::Mat& patch) noexcept override;
  void computeMcPatch(
      const cv::Mat& patch,
      const transform::CachedData& cachedData) noexcept override;
  void computeMcPatchApprox(
      const cv::Mat& patch,
      const syms::ISymData& symData,
      const transform::CachedData& cachedData) noexcept override;
  void computeMcsOffset(
      const cv::Mat& patch,
      const syms::ISymData& symData,
      const transform::CachedData& cachedData) noexcept override;
  void computePatchApprox(const cv::Mat& patch,
                          const syms::ISymData& symData) noexcept override;
  void computeBlurredPatch(
      const cv::Mat& patch,
      const transform::CachedData& cachedData) noexcept override;
  void computeBlurredPatchSq(
      const cv::Mat& patch,
      const transform::CachedData& cachedData) noexcept override;
  void computeVariancePatch(
      const cv::Mat& patch,
      const transform::CachedData& cachedData) noexcept override;
  void computeSsim(const cv::Mat& patch,
                   const syms::ISymData& symData,
                   const transform::CachedData& cachedData) noexcept override;
  void computeNormPatchMinMiu(
      const cv::Mat& patch,
      const transform::CachedData& cachedData) noexcept override;
  void computeAbsCorr(
      const cv::Mat& patch,
      const syms::ISymData& symData,
      const transform::CachedData& cachedData) noexcept override;

  /**
  Returns an instance as for an ideal match between a symbol and a patch.
  Also avoids:
  - either cluttering the interface IMatchParamsRW with setters for creating the
    MatchParams of the perfect match.
  - or introducing a special constructor / Builder just for the perfect match
  */
  static const MatchParams& perfectMatch() noexcept;

  // public method for UnitTesting project
  PROTECTED :

      /// Both computeFg and computeBg simply call this
      static void
      computeMean(const cv::Mat& patch,
                  const cv::Mat& mask,
                  std::optional<double>& miu) noexcept;

  /// Both computeSdevFg and computeSdevBg simply call this
  static void computeSdev(const cv::Mat& patch,
                          const cv::Mat& mask,
                          std::optional<double>& miu,
                          std::optional<double>& sdev) noexcept;

 private:
  // These params are computed only once, if necessary, when approximating the
  // patch
  /// Sum of the values of the pixels of the patch
  std::optional<double> patchSum;

  std::optional<cv::Mat> patchSq;  ///< element-wise product patch * patch
  std::optional<double> normPatchMinMiu;  ///< norm L2 of (patch - miu patch)

  /// Mass center for the patch (range 0..1 x 0..1)
  std::optional<cv::Point2d> mcPatch;

  std::optional<cv::Mat> blurredPatch;    ///< blurred version of the patch
  std::optional<cv::Mat> blurredPatchSq;  ///< blurredPatch element-wise squared
  std::optional<cv::Mat> variancePatch;   ///< blur(patch^2) - blurredPatchSq

  // These params are evaluated for each symbol compared to the patch
  std::optional<cv::Mat> patchApprox;  ///< patch approximated by a given symbol

  /// Mass center for the approximation of the patch (range 0..1 x 0..1)
  std::optional<cv::Point2d> mcPatchApprox;

  /// Distance between the 2 mass centers (range 0..sqrt(2))
  std::optional<double> mcsOffset;

  /// % of the box covered by the glyph (0..1)
  std::optional<double> symDensity;

  std::optional<double> fg;        ///< color for fg (range 0..255)
  std::optional<double> bg;        ///< color for bg (range 0..255)
  std::optional<double> contrast;  ///< fg - bg (range -255..255)
  std::optional<double> ssim;      ///< structural similarity (-1..1)
  std::optional<double> absCorr;   ///< absolute value of correlation (0..1)

  // ideal value for the standard deviations below is 0
  std::optional<double> sdevFg;    ///< standard deviation for fg (0..127.5)
  std::optional<double> sdevBg;    ///< standard deviation for bg  (0..127.5)
  std::optional<double> sdevEdge;  ///< standard deviation for contour (0..255)
};

}  // namespace pic2sym::match

#endif  // H_MATCH_PARAMS
