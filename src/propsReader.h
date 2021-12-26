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

#ifndef H_PROPS_READER
#define H_PROPS_READER

#include "misc.h"

#include "warnings.h"

#pragma warning(push, 0)

#include <algorithm>
#include <array>
#include <concepts>
#include <filesystem>
#include <iomanip>
#include <map>
#include <optional>
#include <ranges>
#include <string>
#include <type_traits>
#include <unordered_set>

#include <boost/property_tree/info_parser.hpp>
#include <boost/property_tree/ptree.hpp>

#include <gsl/gsl>

#pragma warning(pop)

namespace pic2sym {

/// Base class for validators of the configuration items from
/// 'res/varConfig.txt'
template <class Type>
class ConfigItemValidator /*abstract*/ {
 public:
  /// @return false when itemVal is wrong for itemName
  virtual bool examine(const std::string& itemName,
                       const Type& itemVal) const noexcept = 0;

  virtual ~ConfigItemValidator() noexcept = 0 {}
};

/// Base class for IsOdd and IsEven from below
template <std::integral Type>
class IsOddOrEven /*abstract*/ : public ConfigItemValidator<Type> {
 public:
  /// @return false when itemVal is wrong for itemName
  bool examine(const std::string& itemName,
               const Type& itemVal) const noexcept final {
    const Type requiredMod2Remainder{gsl::narrow_cast<Type>(isOdd ? 1 : 0)};
    if (requiredMod2Remainder != (itemVal & (Type)1)) {
      std::cerr << "Configuration item " << std::quoted(itemName, '\'') << " ("
                << itemVal << ") needs to be "
                << parityType.at((size_t)requiredMod2Remainder) << "!"
                << std::endl;
      return false;
    }

    return true;
  }

 protected:
  explicit IsOddOrEven(bool isOdd_) noexcept : isOdd(isOdd_) {}

 private:
  /// even for Mod 2 -> 0, odd for Mod 2 -> 1
  static constexpr std::array<const char*, 2ULL> parityType{"even", "odd"};

  const bool isOdd;  ///< true for IsOdd, false for IsEven
};

/// Signals non-odd configuration items
template <class Type>
class IsOdd final : public IsOddOrEven<Type> {
 public:
  IsOdd() noexcept : IsOddOrEven<Type>{true} {}
};

/// Signals non-even configuration items
template <class Type>
class IsEven final : public IsOddOrEven<Type> {
 public:
  IsEven() noexcept : IsOddOrEven<Type>{false} {}
};

/// Base class for IsLessThan and IsGreaterThan from below
template <class Type>
requires std::is_arithmetic_v<Type>
class IsLessOrGreaterThan /*abstract*/ : public ConfigItemValidator<Type> {
 public:
  /// @return false when itemVal is wrong for itemName
  bool examine(const std::string& itemName,
               const Type& itemVal) const noexcept final {
    if (const bool ok = (isLess ? (itemVal < refVal) : (itemVal > refVal)) ||
                        (orEqual && (itemVal == refVal));
        !ok) {
      std::cerr << "Configuration item " << std::quoted(itemName, '\'') << " ("
                << itemVal << ") needs to be " << PrefixCompare.at(isLess)
                << SuffixCompare.at(orEqual) << ' ' << refVal << '!'
                << std::endl;
      return false;
    }

    return true;
  }

 protected:
  IsLessOrGreaterThan(bool isLess_,
                      const Type& refVal_,
                      bool orEqual_ = false) noexcept
      : refVal(refVal_), isLess(isLess_), orEqual(orEqual_) {}

 private:
  // '<' for isLess = true; '>' for isLess = false;
  static const inline std::map<bool, const char*> PrefixCompare{{true, "<"},
                                                                {false, ">"}};

  // '=' for orEqual = true; '' for orEqual = false;
  static const inline std::map<bool, const char*> SuffixCompare{{true, "="},
                                                                {false, ""}};

  Type refVal;   ///< the value to compare against
  bool isLess;   ///< true for IsLessThan, false for IsGreaterThan
  bool orEqual;  ///< compare strictly or not
};

/// Signals values > or >= than refVal_
template <class Type>
class IsLessThan final : public IsLessOrGreaterThan<Type> {
 public:
  IsLessThan(const Type& refVal_, bool orEqual_ = false) noexcept
      : IsLessOrGreaterThan<Type>{true, refVal_, orEqual_} {}
};

/// Signals values < or <= than refVal_
template <class Type>
class IsGreaterThan final : public IsLessOrGreaterThan<Type> {
 public:
  IsGreaterThan(const Type& refVal_, bool orEqual_ = false) noexcept
      : IsLessOrGreaterThan<Type>{false, refVal_, orEqual_} {}
};

/// Checks that the provided value for a configuration item is within a given
/// set of accepted values.
template <class Type>
class IsOneOf final : public ConfigItemValidator<Type> {
 public:
#pragma warning(disable : WARN_THROWS_ALTHOUGH_NOEXCEPT)
  /**
  Creates a 'belongs to' allowedSet_ validator.
  @throw invalid_argument if allowedSet_ is empty

  Exception to be reported only, not handled
  */
  explicit IsOneOf(const std::unordered_set<Type>& allowedSet_) noexcept(!UT)
      : ConfigItemValidator<Type>(),
        allowedSet(allowedSet_),
        allowedSetStr(setAsString(allowedSet_)) {
    EXPECTS_OR_REPORT_AND_THROW_CONST_MSG(
        !allowedSet_.empty(), std::invalid_argument,
        HERE.function_name() +
            " should get a non-empty set of allowed values!"s);
  }
#pragma warning(default : WARN_THROWS_ALTHOUGH_NOEXCEPT)

  /// @return false when itemVal is wrong for itemName
  bool examine(const std::string& itemName,
               const Type& itemVal) const noexcept final {
    if (!allowedSet.contains(itemVal)) {
      std::cerr << "Configuration item " << std::quoted(itemName, '\'') << " ("
                << itemVal
                << ") needs to be among these values: " << allowedSetStr << "!"
                << std::endl;
      return false;
    }

    return true;
  }

 private:
  /// Helper to initialize allowedSetStr in initialization list
  static std::string setAsString(
      const std::unordered_set<Type>& allowedSet_) noexcept {
    std::ostringstream oss;
    std::ranges::copy(allowedSet_, std::ostream_iterator<Type>(oss, ", "));
    oss << "\b\b ";
    return oss.str();
  }

  std::unordered_set<Type> allowedSet;  ///< allowed set of values
  std::string allowedSetStr;            ///< same set in string format
};

/// Defining validators
#define VALIDATOR(Name, Kind, Type, ...)            \
  const Kind<Type>& Name() noexcept {               \
    static const Kind<Type> validator{__VA_ARGS__}; \
    return validator;                               \
  }

/**
Parser for reading mandatory properties from a configuration file.
Checks against:
- missing properties
- conversion errors
- invalid properties values
*/
class PropsReader {
 public:
  // No intention to copy / move such data providers
  PropsReader(const PropsReader&) = delete;
  PropsReader(PropsReader&&) = delete;
  void operator=(const PropsReader&) = delete;
  void operator=(PropsReader&&) = delete;

  /**
  Builds the parser.

  @param propsFile_ the path to the configuration file

  @throw info_parser_error when the file doesn't exist or cannot be parsed

  Exception to be only reported, not handled
  */
  explicit PropsReader(const std::filesystem::path& propsFile_) noexcept(!UT);

  virtual ~PropsReader() noexcept = default;

  /**
  Reads a certain property assuming it has type T.
  Optionally, several validators (derived from ConfigItemValidator<T>) can be
  provided. All violated validators are reported.

  @param prop the name of the property to be read

  @return the read property or nullopt if unable to read / convert the property
  or its value is invalid
  */
  template <typename T, class... ValidatorTypes>
  requires std::conjunction_v<
      std::is_base_of<ConfigItemValidator<T>, ValidatorTypes>...>
  const std::optional<T> read(
      const std::string& prop,
      const ValidatorTypes&... validators) const noexcept {
    try {
      T value{props.get<T>(prop)};

      // Non-empty array holding the validators' results
      bool validRes[]{
          true,  // a first element to ensure the array won't be empty

          // calling 'examine' for each validator
          validators.examine(prop, value)...};

      // If all validator.examine() returned true, then the value is valid
      if (std::ranges::all_of(validRes, [](bool b) noexcept { return b; }))
        return value;

      foundErrors = true;
      return std::nullopt;

    } catch (const boost::property_tree::ptree_bad_path&) {
      std::cerr << "Property " << std::quoted(prop, '\'')
                << " is missing from '" << propsFile.string() << "' !"
                << std::endl;
    } catch (const boost::property_tree::ptree_bad_data&) {
      std::cerr << "Property " << std::quoted(prop, '\'')
                << " cannot be converted to its required type!" << std::endl;
    }

    foundErrors = true;
    return std::nullopt;
  }

  /// @return true if any property failed to read / convert or its value is
  /// invalid
  bool anyError() const noexcept { return foundErrors; }

  PRIVATE :

      /// Path to the configuration file
      std::filesystem::path propsFile;

  /// The property tree built from the configuration
  boost::property_tree::ptree props;

  /// Any invalid property or failing to be read/convert sets this on true
  mutable bool foundErrors{false};  // mutable to be set from read<T>() const
};

}  // namespace pic2sym

#endif  // H_PROPS_READER
