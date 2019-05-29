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

#ifndef H_PROPS_READER
#define H_PROPS_READER

#include "misc.h"
#include "warnings.h"

#pragma warning(push, 0)

#include <algorithm>
#include <filesystem>
#include <optional>
#include <string>
#include <type_traits>
#include <unordered_set>

#include <boost/property_tree/info_parser.hpp>
#include <boost/property_tree/ptree.hpp>

#pragma warning(pop)

/// Base class for validators of the configuration items from
/// 'res/varConfig.txt'
template <class Type>
class ConfigItemValidator /*abstract*/ {
 public:
  /// @return false when itemVal is wrong for itemName
  virtual bool examine(const std::string& itemName, const Type& itemVal) const
      noexcept = 0;

  virtual ~ConfigItemValidator() noexcept {}

  // Slicing prevention
  ConfigItemValidator(const ConfigItemValidator&) = delete;
  ConfigItemValidator(ConfigItemValidator&&) = delete;
  ConfigItemValidator& operator=(const ConfigItemValidator&) = delete;
  ConfigItemValidator& operator=(ConfigItemValidator&&) = delete;

 protected:
  constexpr ConfigItemValidator() noexcept {}
};

/// Base class for IsOdd and IsEven from below
template <class Type, typename = std::enable_if_t<std::is_integral_v<Type>>>
class IsOddOrEven /*abstract*/ : public ConfigItemValidator<Type> {
 public:
  /// @return false when itemVal is wrong for itemName
  bool examine(const std::string& itemName, const Type& itemVal) const
      noexcept final {
#define REQUIRE_PARITY(ParityType, Mod2Remainder)                       \
  if ((Type)Mod2Remainder != (itemVal & (Type)1)) {                     \
    std::cerr << "Configuration item '" << itemName << "' (" << itemVal \
              << ") needs to be " ParityType "!" << std::endl;          \
    return false;                                                       \
  }

    if (isOdd) {
      REQUIRE_PARITY("odd", 1);
    } else {
      REQUIRE_PARITY("even", 0);
    }

#undef REQUIRE_PARITY

    return true;
  }

 protected:
  explicit IsOddOrEven(bool isOdd_) noexcept : isOdd(isOdd_) {}

 private:
  const bool isOdd;  ///< true for IsOdd, false for IsEven
};

/// Signals non-odd configuration items
template <class Type>
class IsOdd final : public IsOddOrEven<Type> {
 public:
  IsOdd() noexcept : IsOddOrEven(true) {}
};

/// Signals non-even configuration items
template <class Type>
class IsEven final : public IsOddOrEven<Type> {
 public:
  IsEven() noexcept : IsOddOrEven(false) {}
};

/// Base class for IsLessThan and IsGreaterThan from below
template <class Type, typename = std::enable_if_t<std::is_arithmetic_v<Type>>>
class IsLessOrGreaterThan /*abstract*/ : public ConfigItemValidator<Type> {
 public:
  /// @return false when itemVal is wrong for itemName
  bool examine(const std::string& itemName, const Type& itemVal) const
      noexcept final {
#define REQUIRE_REL(RelationType)                                       \
  if (!(itemVal RelationType refVal)) {                                 \
    std::cerr << "Configuration item '" << itemName << "' (" << itemVal \
              << ") needs to be " #RelationType " " << refVal << "!"    \
              << std::endl;                                             \
    return false;                                                       \
  }

    if (isLess) {
      if (orEqual) {
        REQUIRE_REL(<=);
      } else {
        REQUIRE_REL(<);
      }

    } else {  // isGreater
      if (orEqual) {
        REQUIRE_REL(>=);
      } else {
        REQUIRE_REL(>);
      }
    }

#undef REQUIRE_REL

    return true;
  }

 protected:
  IsLessOrGreaterThan(bool isLess_,
                      const Type& refVal_,
                      bool orEqual_ = false) noexcept
      : refVal(refVal_), isLess(isLess_), orEqual(orEqual_) {}

 private:
  const Type refVal;   ///< the value to compare against
  const bool isLess;   ///< true for IsLessThan, false for IsGreaterThan
  const bool orEqual;  ///< compare strictly or not
};

/// Signals values > or >= than refVal_
template <class Type>
class IsLessThan final : public IsLessOrGreaterThan<Type> {
 public:
  IsLessThan(const Type& refVal_, bool orEqual_ = false) noexcept
      : IsLessOrGreaterThan(true, refVal_, orEqual_) {}
};

/// Signals values < or <= than refVal_
template <class Type>
class IsGreaterThan final : public IsLessOrGreaterThan<Type> {
 public:
  IsGreaterThan(const Type& refVal_, bool orEqual_ = false) noexcept
      : IsLessOrGreaterThan(false, refVal_, orEqual_) {}
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
      : ConfigItemValidator(),
        allowedSet(allowedSet_),
        allowedSetStr(setAsString(allowedSet_)) {
    if (allowedSet_.empty())
      THROW_WITH_CONST_MSG(
          __FUNCTION__ " should get a non-empty set of allowed values!",
          std::invalid_argument);
  }
#pragma warning(default : WARN_THROWS_ALTHOUGH_NOEXCEPT)

  /// @return false when itemVal is wrong for itemName
  bool examine(const std::string& itemName, const Type& itemVal) const
      noexcept final {
    if (allowedSet.cend() == allowedSet.find(itemVal)) {
      std::cerr << "Configuration item '" << itemName << "' (" << itemVal
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
    std::copy(CBOUNDS(allowedSet_), std::ostream_iterator<Type>(oss, ", "));
    oss << "\b\b ";
    return oss.str();
  }

  const std::unordered_set<Type> allowedSet;  ///< allowed set of values
  const std::string allowedSetStr;            ///< same set in string format
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
  template <typename T,
            class... ValidatorTypes,
            typename = std::enable_if_t<std::conjunction_v<
                std::is_base_of<ConfigItemValidator<T>, ValidatorTypes>...>>>
  const std::optional<T> read(const std::string& prop,
                              const ValidatorTypes&... validators) const
      noexcept {
    try {
      T value = props.get<T>(prop);

      // Non-empty array holding the validators' results
      bool validRes[]{
          true,  // a first element to ensure the array won't be empty

          // calling 'examine' for each validator
          validators.examine(prop, value)...};

      // If all validator.examine() returned true, then the value is valid
      if (std::all_of(CBOUNDS(validRes), [](bool b) noexcept { return b; }))
        return value;

      foundErrors = true;
      return std::nullopt;

    } catch (const boost::property_tree::ptree_bad_path&) {
      std::cerr << "Property '" << prop << "' is missing from '" << propsFile
                << "' !" << std::endl;
    } catch (const boost::property_tree::ptree_bad_data&) {
      std::cerr << "Property '" << prop
                << "' cannot be converted to its required type!" << std::endl;
    }

    foundErrors = true;
    return std::nullopt;
  }

  /// @return true if any property failed to read / convert or its value is
  /// invalid
  bool anyError() const noexcept { return foundErrors; }

  PRIVATE :

      /// Path to the configuration file
      const std::filesystem::path propsFile;

  /// The property tree built from the configuration
  boost::property_tree::ptree props;

  /// Any invalid property or failing to be read/convert sets this on true
  mutable bool foundErrors = false;  // mutable to be set from read<T>() const
};

#endif  // H_PROPS_READER
