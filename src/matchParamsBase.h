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

#ifndef H_MATCH_PARAMS_BASE
#define H_MATCH_PARAMS_BASE

#pragma warning(push, 0)

#if defined _DEBUG || defined UNIT_TESTING
#include <string>
#endif  // _DEBUG || UNIT_TESTING

#include <memory>

#include <optional>

#include <opencv2/core/core.hpp>

#pragma warning(pop)

// Forward declarations
class ISymData;
class CachedData;
class IMatchParamsRW;

extern template class std::optional<double>;

/**
Base class (Read-only version) for the relevant parameters during patch&glyph
matching.

AI Reviewer detected segregation & split personality in this interface.

Indeed, the fields & methods concerning mass centers (getMc*) are used only in
GravitationalSmoothness and DirectionalSmoothness, apart from the Unit tests.
However, those references of IMatchParams appear within the parameter list of
several methods overriding corresponding methods from MatchAspect.
So, after introducing a new interface for getMc* methods derived from
IMatchParams, the interface would be returned by a
dynamic_cast<new_interface&>(param_of_type_IMatchParams).

This makes the interface separation less appealing.
*/
class IMatchParams /*abstract*/ {
 public:
  // These params are computed only once, if necessary, when approximating the
  // patch
  /// Mass center for the patch (range 0..1 x 0..1)
  virtual const std::optional<cv::Point2d>& getMcPatch() const noexcept = 0;
#ifdef UNIT_TESTING
  /// Sum of the values of the pixels of the patch
  virtual const std::optional<double>& getPatchSum() const noexcept = 0;

  /// Element-wise product patch * patch
  virtual const std::optional<cv::Mat>& getPatchSq() const noexcept = 0;

  /// Norm L2 of (patch - miu patch)
  virtual const std::optional<double>& getNormPatchMinMiu() const noexcept = 0;

  /// Blurred version of the patch
  virtual const std::optional<cv::Mat>& getBlurredPatch() const noexcept = 0;

  /// BlurredPatch element-wise squared
  virtual const std::optional<cv::Mat>& getBlurredPatchSq() const = 0;

  /// Blur(patch^2) - blurredPatchSq
  virtual const std::optional<cv::Mat>& getVariancePatch() const noexcept = 0;

  // These params are evaluated for each symbol compared to the patch
  /// Patch approximated by a given symbol
  virtual const std::optional<cv::Mat>& getPatchApprox() const noexcept = 0;
#endif  // UNIT_TESTING defined

  /// Mass center for the approximation of the patch (range 0..1 x 0..1)
  virtual const std::optional<cv::Point2d>& getMcPatchApprox() const
      noexcept = 0;

  /// Distance between the 2 mass centers (range 0..sqrt(2))
  virtual const std::optional<double>& getMcsOffset() const noexcept = 0;

  /// % of the box covered by the glyph (0..1)
  virtual const std::optional<double>& getSymDensity() const noexcept = 0;

#if defined(_DEBUG) || defined(UNIT_TESTING)
  /// Color for fg (range 0..255)
  virtual const std::optional<double>& getFg() const noexcept = 0;
#endif  // defined(_DEBUG) || defined(UNIT_TESTING)

  /// Color for bg (range 0..255)
  virtual const std::optional<double>& getBg() const noexcept = 0;

  /// fg - bg (range -255..255)
  virtual const std::optional<double>& getContrast() const noexcept = 0;

  /// Structural similarity (-1..1)
  virtual const std::optional<double>& getSsim() const noexcept = 0;

  /// Absolute value of correlation (0..1)
  virtual const std::optional<double>& getAbsCorr() const noexcept = 0;

  // ideal value for the standard deviations below is 0
  /// Standard deviation for fg (0..127.5)
  virtual const std::optional<double>& getSdevFg() const noexcept = 0;

  /// Standard deviation for bg  (0..127.5)
  virtual const std::optional<double>& getSdevBg() const noexcept = 0;

  /// Standard deviation for contour (0..255)
  virtual const std::optional<double>& getSdevEdge() const noexcept = 0;

  virtual ~IMatchParams() noexcept {}

#ifdef UNIT_TESTING
  /// @return a configurable copy of itself
  virtual std::unique_ptr<IMatchParamsRW> clone() const noexcept = 0;
#endif  // UNIT_TESTING defined

#if defined _DEBUG || defined UNIT_TESTING
  /// Provides a representation of the parameters
  virtual const std::wstring toWstring() const noexcept = 0;
#endif  // _DEBUG || UNIT_TESTING

  // If slicing is observed and becomes a severe problem, use `= delete` for all
  IMatchParams(const IMatchParams&) noexcept = default;
  IMatchParams(IMatchParams&&) noexcept = default;
  IMatchParams& operator=(const IMatchParams&) noexcept = default;
  IMatchParams& operator=(IMatchParams&&) noexcept = default;

 protected:
  constexpr IMatchParams() noexcept {}
};

#if defined _DEBUG || defined UNIT_TESTING
std::wostream& operator<<(std::wostream& wos, const IMatchParams& mp) noexcept;
#endif  // _DEBUG || UNIT_TESTING

/// Base class (Read-Write version) for the relevant parameters during
/// patch&glyph matching
class IMatchParamsRW /*abstract*/ : public IMatchParams {
 public:
  /**
  Prepares for next symbol to match against patch.

  When skipPatchInvariantParts = true resets everything except:
  mcPatch, blurredPatch, blurredPatchSq and variancePatch.
  */
  virtual IMatchParamsRW& reset(
      bool skipPatchInvariantParts = true) noexcept = 0;

  // Methods for computing each field
  virtual void computeFg(const cv::Mat& patch, const ISymData&) noexcept = 0;
  virtual void computeBg(const cv::Mat& patch, const ISymData&) noexcept = 0;
  virtual void computeContrast(const cv::Mat& patch,
                               const ISymData&) noexcept = 0;
  virtual void computeSdevFg(const cv::Mat& patch,
                             const ISymData&) noexcept = 0;
  virtual void computeSdevBg(const cv::Mat& patch,
                             const ISymData&) noexcept = 0;
  virtual void computeSdevEdge(const cv::Mat& patch,
                               const ISymData&) noexcept = 0;
  virtual void computeSymDensity(const ISymData&) noexcept = 0;
  virtual void computePatchSum(const cv::Mat& patch) noexcept = 0;
  virtual void computePatchSq(const cv::Mat& patch) noexcept = 0;
  virtual void computeMcPatch(const cv::Mat& patch,
                              const CachedData&) noexcept = 0;
  virtual void computeMcPatchApprox(const cv::Mat& patch,
                                    const ISymData&,
                                    const CachedData&) noexcept = 0;
  virtual void computeMcsOffset(const cv::Mat& patch,
                                const ISymData&,
                                const CachedData&) noexcept = 0;
  virtual void computePatchApprox(const cv::Mat& patch,
                                  const ISymData&) noexcept = 0;
  virtual void computeBlurredPatch(const cv::Mat& patch,
                                   const CachedData&) noexcept = 0;
  virtual void computeBlurredPatchSq(const cv::Mat& patch,
                                     const CachedData&) noexcept = 0;
  virtual void computeVariancePatch(const cv::Mat& patch,
                                    const CachedData&) noexcept = 0;
  virtual void computeSsim(const cv::Mat& patch,
                           const ISymData&,
                           const CachedData&) noexcept = 0;
  virtual void computeNormPatchMinMiu(const cv::Mat& patch,
                                      const CachedData&) noexcept = 0;
  virtual void computeAbsCorr(const cv::Mat& patch,
                              const ISymData&,
                              const CachedData&) noexcept = 0;

 protected:
  constexpr IMatchParamsRW() noexcept {}
};

#endif  // H_MATCH_PARAMS_BASE
