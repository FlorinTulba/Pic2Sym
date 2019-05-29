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

#include "matchSettings.h"
#include "warnings.h"

unsigned MatchSettings::VERSION_FROM_LAST_IO_OP = UINT_MAX;

#pragma warning(push, 0)

#include <iostream>

#pragma warning(pop)

#ifndef UNIT_TESTING

#include "appStart.h"
#include "propsReader.h"
#include "settingsBase.h"

#pragma warning(push, 0)

#include <filesystem>

#pragma warning(pop)

using namespace boost::archive;

using namespace std;
using namespace std::filesystem;

namespace {

/// Validator for a threshold for blanks-like patches (really poor contrast)
class BlanksLimit : public ConfigItemValidator<unsigned> {
 public:
  constexpr BlanksLimit() noexcept {}

  /// @return false when itemVal is wrong for itemName
  bool examine(const string& itemName, const unsigned& itemVal) const
      noexcept override {
    return ISettings::isBlanksThresholdOk(itemVal, itemName);
  }
};

/// Instance of the blanks validator
const BlanksLimit blanksLimit;

/// Ensures several settings are >= 0
VALIDATOR(positiveD, IsGreaterThan, double, 0., true);

}  // anonymous namespace

#pragma warning(disable : WARN_THROWS_ALTHOUGH_NOEXCEPT)
void MatchSettings::configurePaths() noexcept(!UT) {
  if (defCfgPath.empty()) {
    defCfgPath = cfgPath = AppStart::dir();

    if (!exists(defCfgPath.append("res").append("defaultMatchSettings.txt")))
      THROW_WITH_VAR_MSG(__FUNCTION__ " : There's no " + defCfgPath.string(),
                         runtime_error);

    cfgPath.append("initMatchSettings.cfg");
  }
}
#pragma warning(default : WARN_THROWS_ALTHOUGH_NOEXCEPT)

void MatchSettings::replaceByUserDefaults() {
  ifstream ifs(cfgPath.string(), ios::binary);
  binary_iarchive ia(ifs);

  // Throws domain_error if loading from an archive with an unsupported
  // version(more recent) or for an obsolete 'initMatchSettings.cfg'
  ia >> *this;
}

void MatchSettings::saveAsUserDefaults() const noexcept {
  ofstream ofs(cfgPath.string(), ios::binary);
  binary_oarchive oa(ofs);
  oa << *this;
}

#pragma warning(disable : WARN_THROWS_ALTHOUGH_NOEXCEPT)
void MatchSettings::createUserDefaults() noexcept(!UT) {
  if (!parseCfg())
    THROW_WITH_CONST_MSG(__FUNCTION__ " : Invalid Configuration!",
                         runtime_error);

  saveAsUserDefaults();
}
#pragma warning(default : WARN_THROWS_ALTHOUGH_NOEXCEPT)

bool MatchSettings::parseCfg() noexcept(!UT) {
  static PropsReader parser(defCfgPath);
  // Might trigger info_parser_error if unable to find/parse the file
  // The exception terminates the program, unless in UnitTesting, where it
  // propagates

  const bool newResultMode = parser.read<bool>("HYBRID_RESULT").value_or(false);

  const double
      new_kSsim = parser.read<double>("STRUCTURAL_SIMILARITY", positiveD())
                      .value_or(0.),
      new_kCorrel = parser.read<double>("CORRELATION_CORRECTNESS", positiveD())
                        .value_or(0.),
      new_kSdevFg = parser.read<double>("UNDER_SYM_CORRECTNESS", positiveD())
                        .value_or(0.),
      new_kSdevEdge =
          parser.read<double>("SYM_EDGE_CORRECTNESS", positiveD()).value_or(0.),
      new_kSdevBg = parser.read<double>("ASIDE_SYM_CORRECTNESS", positiveD())
                        .value_or(0.),
      new_kMCsOffset =
          parser.read<double>("MORE_CONTRAST_PREF", positiveD()).value_or(0.),
      new_kCosAngleMCs =
          parser.read<double>("GRAVITATIONAL_SMOOTHNESS", positiveD())
              .value_or(0.),
      new_kContrast = parser.read<double>("DIRECTIONAL_SMOOTHNESS", positiveD())
                          .value_or(0.),
      new_kSymDensity =
          parser.read<double>("LARGER_SYM_PREF", positiveD()).value_or(0.);

  const unsigned newThreshold4Blank =
      parser.read<unsigned>("THRESHOLD_FOR_BLANK", blanksLimit).value_or(0U);

  if (parser.anyError())
    return false;

  setResultMode(newResultMode);
  set_kSsim(new_kSsim);
  set_kCorrel(new_kCorrel);
  set_kSdevFg(new_kSdevFg);
  set_kSdevEdge(new_kSdevEdge);
  set_kSdevBg(new_kSdevBg);
  set_kContrast(new_kContrast);
  set_kMCsOffset(new_kMCsOffset);
  set_kCosAngleMCs(new_kCosAngleMCs);
  set_kSymDensity(new_kSymDensity);
  setBlankThreshold(newThreshold4Blank);

  cout << "Initial config values:\n" << *this << endl;

  return true;
}

MatchSettings::MatchSettings() noexcept(!UT) {
  configurePaths();

  ScopeExitAction scopeExitAction([this] {
    if (!initialized)
      initialized = true;
  });

  if (exists(cfgPath)) {
    if (last_write_time(cfgPath) > last_write_time(defCfgPath)) {  // newer
#pragma warning(disable : WARN_SEH_NOT_CAUGHT)
      try {
        replaceByUserDefaults();  // throws invalid files or older versions

        return;

      } catch (const domain_error&) {
      }  // invalid files or older versions
#pragma warning(default : WARN_SEH_NOT_CAUGHT)
    }

    // Renaming the obsolete file
    rename(cfgPath, std::filesystem::path(cfgPath)
                        .concat(".")
                        .concat(to_string(time(nullptr)))
                        .concat(".bak"));
  }

  // Create a fresh 'initMatchSettings.cfg' with data from
  // 'res/defaultMatchSettings.txt'
  createUserDefaults();
}

#endif  // UNIT_TESTING not defined

using namespace std;

MatchSettings& MatchSettings::setResultMode(bool hybridResultMode_) noexcept {
  if (hybridResultMode != hybridResultMode_) {
    cout << "hybridResultMode"
         << " : " << hybridResultMode << " -> " << hybridResultMode_ << endl;
    hybridResultMode = hybridResultMode_;
  }
  return *this;
}

MatchSettings& MatchSettings::set_kSsim(double kSsim_) noexcept {
  if (kSsim != kSsim_) {
    cout << "kSsim"
         << " : " << kSsim << " -> " << kSsim_ << endl;
    kSsim = kSsim_;
  }
  return *this;
}

MatchSettings& MatchSettings::set_kCorrel(double kCorrel_) noexcept {
  if (kCorrel != kCorrel_) {
    cout << "kCorrel"
         << " : " << kCorrel << " -> " << kCorrel_ << endl;
    kCorrel = kCorrel_;
  }
  return *this;
}

MatchSettings& MatchSettings::set_kSdevFg(double kSdevFg_) noexcept {
  if (kSdevFg != kSdevFg_) {
    cout << "kSdevFg"
         << " : " << kSdevFg << " -> " << kSdevFg_ << endl;
    kSdevFg = kSdevFg_;
  }
  return *this;
}

MatchSettings& MatchSettings::set_kSdevEdge(double kSdevEdge_) noexcept {
  if (kSdevEdge != kSdevEdge_) {
    cout << "kSdevEdge"
         << " : " << kSdevEdge << " -> " << kSdevEdge_ << endl;
    kSdevEdge = kSdevEdge_;
  }
  return *this;
}

MatchSettings& MatchSettings::set_kSdevBg(double kSdevBg_) noexcept {
  if (kSdevBg != kSdevBg_) {
    cout << "kSdevBg"
         << " : " << kSdevBg << " -> " << kSdevBg_ << endl;
    kSdevBg = kSdevBg_;
  }
  return *this;
}

MatchSettings& MatchSettings::set_kContrast(double kContrast_) noexcept {
  if (kContrast != kContrast_) {
    cout << "kContrast"
         << " : " << kContrast << " -> " << kContrast_ << endl;
    kContrast = kContrast_;
  }
  return *this;
}

MatchSettings& MatchSettings::set_kCosAngleMCs(double kCosAngleMCs_) noexcept {
  if (kCosAngleMCs != kCosAngleMCs_) {
    cout << "kCosAngleMCs"
         << " : " << kCosAngleMCs << " -> " << kCosAngleMCs_ << endl;
    kCosAngleMCs = kCosAngleMCs_;
  }
  return *this;
}

MatchSettings& MatchSettings::set_kMCsOffset(double kMCsOffset_) noexcept {
  if (kMCsOffset != kMCsOffset_) {
    cout << "kMCsOffset"
         << " : " << kMCsOffset << " -> " << kMCsOffset_ << endl;
    kMCsOffset = kMCsOffset_;
  }
  return *this;
}

MatchSettings& MatchSettings::set_kSymDensity(double kSymDensity_) noexcept {
  if (kSymDensity != kSymDensity_) {
    cout << "kSymDensity"
         << " : " << kSymDensity << " -> " << kSymDensity_ << endl;
    kSymDensity = kSymDensity_;
  }
  return *this;
}

MatchSettings& MatchSettings::setBlankThreshold(
    unsigned threshold4Blank_) noexcept {
  if (threshold4Blank != threshold4Blank_) {
    cout << "threshold4Blank"
         << " : " << threshold4Blank << " -> " << threshold4Blank_ << endl;
    threshold4Blank = threshold4Blank_;
  }
  return *this;
}

unique_ptr<IMatchSettings> MatchSettings::clone() const noexcept {
  return make_unique<MatchSettings>(*this);
}

#pragma warning(disable : WARN_EXPR_ALWAYS_FALSE)
bool MatchSettings::olderVersionDuringLastIO() noexcept {
  return VERSION_FROM_LAST_IO_OP < VERSION;
}
#pragma warning(default : WARN_EXPR_ALWAYS_FALSE)
