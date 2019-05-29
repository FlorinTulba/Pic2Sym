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

#ifndef H_BEST_MATCH
#define H_BEST_MATCH

#include "bestMatchBase.h"
#include "matchParamsBase.h"

/// Holds the best match found at a given time
class BestMatch : public IBestMatch {
 public:
  /// Constructor setting only 'patch'. The other fields need the setters or the
  /// 'update' method.
  BestMatch(const IPatch& patch_) noexcept;

  /// The patch to approximate, together with some other details. No setter
  /// available
  const IPatch& getPatch() const noexcept final;

  /// The approximation of the patch
  const cv::Mat& getApprox() const noexcept final;

  /// Parameters of the match (none for blur-only approximations)
  const std::optional<const IMatchParams*> getParams() const noexcept final;

  /// Parameters of the match
  const std::unique_ptr<IMatchParamsRW>& refParams() const noexcept final;

  /// Index within vector&lt;DerivedFrom_ISymData&gt;. none if patch
  /// approximation is blur-based only.
  const std::optional<unsigned>& getSymIdx() const noexcept final;

  /// Index of last cluster that was worth investigating thoroughly
  const std::optional<unsigned>& getLastPromisingNontrivialCluster() const
      noexcept final;

  /// Set the index of last cluster that was worth investigating thoroughly;
  /// @return itself
  BestMatch& setLastPromisingNontrivialCluster(
      unsigned clustIdx) noexcept final;

  /// Glyph code. none if patch approximation is blur-based only
  const std::optional<unsigned long>& getSymCode() const noexcept final;

  /// Score of the best match. If patch approximation is blur-based only, score
  /// will remain 0.
  double getScore() const noexcept final;

  /// Set the score of the new best match; @return itself
  BestMatch& setScore(double score_) noexcept final;

  /// Resets everything apart the patch and the patch-invariant parameters
  BestMatch& reset() noexcept override;

  /// Called when finding a better match. Returns itself
  BestMatch& update(double score_,
                    unsigned long symCode_,
                    unsigned symIdx_,
                    const ISymData& sd) noexcept override;

  /**
  It generates the approximation of the patch based on the rest of the fields.

  When pSymData is null, it approximates the patch with patch.blurredPatch.

  Otherwise it adapts the foreground and background of the original glyph to
  be as close as possible to the patch.
  For color patches, it does that for every channel.

  For low-contrast images, it generates the average of the patch.

  @param ms additional match settings that might be needed

  @return reference to the updated object

  @throw logic_error when called for uniform patches - not to be handled
  */
  BestMatch& updatePatchApprox(const IMatchSettings& ms) noexcept(!UT) override;

#if defined _DEBUG || \
    defined UNIT_TESTING  // Next members are necessary for logging
  /**
  Unicode symbols will be logged in symbol format, while other encodings will
  log their code
  @return is Unicode the charmap's encoding
  */
  bool isUnicode() const noexcept final;

  /// Updates unicode field and returns the updated BestMatch object
  BestMatch& setUnicode(bool unicode_) noexcept final;

  /// Provides a representation of the match
  const std::wstring toWstring() const noexcept override;

#endif  // defined _DEBUG || defined UNIT_TESTING

 private:
  /// The patch to approximate, together with some other details
  const std::unique_ptr<const IPatch> patch;

  cv::Mat approx;  ///< the approximation of the patch

  /// Parameters of the match (none for blur-only approximations)
  const std::unique_ptr<IMatchParamsRW> params;

  /// Index within vector&lt;DerivedFrom_ISymData&gt; none if patch
  /// approximation is blur-based only.
  std::optional<unsigned> symIdx;

  /// Index of last cluster that was worth investigating thoroughly
  std::optional<unsigned> lastPromisingNontrivialCluster;

  /// pointer to vector&lt;DerivedFrom_ISymData&gt;[symIdx] or null when patch
  /// approximation is blur-based only.
  const ISymData* pSymData = nullptr;

  /// glyph code. none if patch approximation is blur-based only
  std::optional<unsigned long> symCode;

  /// score of the best match. If patch approximation is blur-based only, score
  /// will remain 0.
  double score = 0.;

#if defined _DEBUG || defined UNIT_TESTING  // Field necessary for logging
  // Unicode symbols are logged in symbol format, while other encodings log
  // their code
  bool unicode = false;  ///< is Unicode the charmap's encoding

#endif  // defined _DEBUG || defined UNIT_TESTING
};

#endif  // H_BEST_MATCH
