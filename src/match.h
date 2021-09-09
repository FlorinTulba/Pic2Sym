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

#ifndef H_MATCH
#define H_MATCH

#include "cachedData.h"
#include "matchParamsBase.h"
#include "matchSettingsBase.h"
#include "symbolsSupportBase.h"

#pragma warning(push, 0)

#include <vector>

#include <gsl/gsl>

#include <opencv2/core/core.hpp>

#pragma warning(pop)

extern template class std::vector<std::string>;

namespace pic2sym::match {

/**
Base class for all considered aspects of matching.

Derived classes should have protected constructors and
objects should be created only by the MatchAspectsFactory class.

UNIT_TESTING should still have the constructors of the derived classes as
public.
*/
class MatchAspect /*abstract*/ {
 public:
  virtual ~MatchAspect() noexcept {}

  MatchAspect(const MatchAspect&) noexcept = default;

  MatchAspect(MatchAspect&&) noexcept = delete;
  void operator=(const MatchAspect&) = delete;
  void operator=(MatchAspect&&) = delete;

  /// Scores the match between a gray patch and a symbol based on current aspect
  /// (Template method)
  double assessMatch(const cv::Mat& patch,
                     const syms::ISymData& symData,
                     const transform::CachedData& cachedData,
                     IMatchParamsRW& mp) const noexcept;

  /// Computing max score of a this MatchAspect
  double maxScore(const transform::CachedData& cachedData) const noexcept;

  /**
  Providing a clue about how complex is this MatchAspect compared to the others.

  @return 1000 for Structural Similarity (the slowest one) and proportionally
  lower values for the rest of the aspects when timed individually using
  DengXian regular Unicode size 10, no preselection, no symbol batching and no
  hybrid result
  */
  virtual double relativeComplexity() const noexcept = 0;

  /// Provides aspect's name
  virtual const std::string& name() const noexcept = 0;

  /// All aspects that are configured with coefficients > 0 are enabled; those
  /// with 0 are disabled
  bool enabled() const noexcept;

  /// Provides the list of names of all registered aspects
  static const std::vector<std::string>& aspectNames() noexcept;

 protected:
  /// Base class constructor
  explicit MatchAspect(const double& k_) noexcept;

  /// Provides a list of names of the already registered aspects
  static std::vector<std::string>& registeredAspects() noexcept;

  /// Defines the scoring rule, based on all required fields computed already in
  /// MatchParams mp
  virtual double score(
      const IMatchParams& mp,
      const transform::CachedData& cachedData) const noexcept = 0;

  /// Prepares required fields from MatchParams mp to be able to assess the
  /// match
  virtual void fillRequiredMatchParams(const cv::Mat& patch,
                                       const syms::ISymData& symData,
                                       const transform::CachedData& cachedData,
                                       IMatchParamsRW& mp) const noexcept = 0;

  /**
  Helper class to populate registeredAspects.
  Define a static private field of this type in each subclass using
  REGISTER_MATCH_ASPECT defined below
  */
  class NameRegistrator {
   public:
    /// adds a new aspect name to registeredAspects
    explicit NameRegistrator(const std::string& aspectType) noexcept;
  };

  /// Cached coefficient from IMatchSettings, corresponding to current aspect
  gsl::not_null<const double*> k;
};

/// Place this call at the end of an aspect class to register (HEADER file).
#define REGISTER_MATCH_ASPECT(AspectName)                            \
 public:                                                             \
  /** provides aspect's name */                                      \
  const std::string& name() const noexcept override { return Name; } \
                                                                     \
 private:                                                            \
  /** static method 'MatchAspectsFactory::create(...)` */            \
  /** creates unique_ptr<AspectName> */                              \
  friend class MatchAspectsFactory;                                  \
                                                                     \
  /** aspect's name */                                               \
  static inline const std::string Name{#AspectName};                 \
                                                                     \
  /** Instance that registers this Aspect */                         \
  static inline const NameRegistrator nameRegistrator{#AspectName};

/*
STEPS TO CREATE A NEW 'MatchAspect' (<NewAspect>):
==================================================

(1) Create a class for it using the template:

  /// Class Details
  class <NewAspect> : public MatchAspect {
  protected:
    double score(const IMatchParams &mp,
                 const transform::CachedData &cachedData) const noexcept
override;

    void fillRequiredMatchParams(const cv::Mat &patch,
                                 const syms::ISymData &symData,
                                 const transform::CachedData &cachedData,
                                 IMatchParamsRW &mp) const noexcept override;

  public:
    ~<NewAspect>() noexcept = default;

    <NewAspect>(const <NewAspect>&) noexcept = default;
    <NewAspect>(<NewAspect>&&) noexcept = default;

    // 'k' reference is supposed not to change for the original / copy
    void operator=(const <NewAspect>&) = delete;
    void operator=(<NewAspect>&&) = delete;

    double relativeComplexity() const noexcept override;

  PROTECTED: // UNIT_TESTING needs the constructors as public

    /// Constructor Details
    explicit <NewAspect>(const cfg::IMatchSettings &ms) noexcept;

    REGISTER_MATCH_ASPECT(<NewAspect>);
  };

(2) Include the file declaring <NewAspect> in 'matchAspectsFactory.cpp' and
  add the line below to 'MatchAspectsFactory::create()'.

  HANDLE_MATCH_ASPECT(<NewAspect>);
*/

}  // namespace pic2sym::match

#endif  // H_MATCH
