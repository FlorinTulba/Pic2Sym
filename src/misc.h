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

#ifndef H_MISC
#define H_MISC

#pragma warning(push, 0)

#include <functional>
#include <iostream>
#include <optional>
#include <ostream>
#include <source_location>
#include <string>
#include <string_view>
#include <type_traits>

#include <gsl/pointers>

#pragma warning(pop)

using namespace std::literals;

/*
Several helper macros to simplify expressing differences between the Pic2Sym and
UnitTesting projects:
- UT bool which is true only when compiling the UnitTesting project
- PROTECTED remains protected in Pic2Sym, while in UnitTesting becomes public
- PRIVATE remains private in Pic2Sym, while in UnitTesting becomes public
*/
#ifdef UNIT_TESTING

#define UT true
#define PROTECTED public
#define PRIVATE public

#else  // UNIT_TESTING not defined

#define UT false
#define PROTECTED protected
#define PRIVATE private

#endif  // UNIT_TESTING

namespace pic2sym {

// Various constants.
// Initialization-order fiasco avoided by directly including this file,
// thus no need for extern globals, nor lazy evaluation and function calls
constexpr double Eps{1e-6};  ///< Error margin
constexpr double EpsPlus1{1. + Eps};
constexpr double OneMinusEps{1. - Eps};
constexpr double Inv255{1. / 255.};

constexpr int EscKeyCode{27};

}  // namespace pic2sym

// Display an expression and its value
#define PRINT(expr) std::cout << #expr " : " << (expr)
#define PRINTLN(expr) PRINT(expr) << std::endl

namespace std {

// Display values of optionals or `--` if they are not initialized
template <typename CharType, typename T>
std::basic_ostream<CharType, std::char_traits<CharType>>& operator<<(
    std::basic_ostream<CharType, std::char_traits<CharType>>& os,
    const std::optional<T>& v) noexcept {
  if (v) [[likely]]
    return os << *v;

  return os << "--";
}

}  // namespace std

namespace gsl {

// Display values of not_null by getting the wrapped pointer.
// Useful for not_null<char*> or not_null<wchar_t*>
template <typename CharType, typename T>
std::basic_ostream<CharType, std::char_traits<CharType>>& operator<<(
    std::basic_ostream<CharType, std::char_traits<CharType>>& os,
    const gsl::not_null<T>& v) noexcept {
  return os << v.get();
}

}  // namespace gsl

// Oftentimes functions operating on ranges need the full range.
// Example: copy(x.begin(), x.end(), ..) => copy(CBOUNDS(x), ..)
#define BOUNDS(iterable) std::begin(iterable), std::end(iterable)
#define CBOUNDS(iterable) std::cbegin(iterable), std::cend(iterable)
#define BOUNDS_FOR_ITEM_TYPE(iterable, type) \
  iterable.begin<type>(), iterable.end<type>()

// Access to source location information(file, line, col, function)
#define HERE std::source_location::current()

namespace pic2sym {

// string <-> wstring conversions
std::wstring str2wstr(std::string_view str) noexcept;
std::string wstr2str(std::wstring_view wstr) noexcept;

// Notifying the user
void infoMsg(std::string_view text, std::string_view title = "") noexcept;
void warnMsg(std::string_view text, std::string_view title = "") noexcept;
void errMsg(std::string_view text, std::string_view title = "") noexcept;

// Throwing exceptions while displaying the exception message to the console
/**
First version should be used when the exception message is constant.
Declaring a method-static variable makes sense only when the exception is
caught, so particularly for Unit Testing for this application. Otherwise, the
program just leaves, letting no chance for reusing the method-static variable.
The errMsg parameter might be a concatenating expression, so it is better to
evaluate it only once by storing it into constErrMsgForConsoleAndThrow
*/
#define REPORT_AND_THROW_CONST_MSG(excType, errMsg, ...)                 \
  {                                                                      \
    static const std::string constErrMsgForConsoleAndThrow{errMsg};      \
    reportAndThrow<excType>(constErrMsgForConsoleAndThrow, __VA_ARGS__); \
  }

/**
Second version should be used when the exception message is variable (reports
specific values).
The errMsg parameter might be a concatenating expression, so it is better to
evaluate it only once by storing it into varErrMsgForConsoleAndThrow
*/
template <class ExcType, typename... OtherExcArgsTypes>
requires std::is_base_of_v<std::exception, ExcType>
void reportAndThrow(const std::string& errMsg,
                    OtherExcArgsTypes&&... otherExcArgs) {
  std::cerr << errMsg << std::endl;
  throw ExcType{errMsg, std::forward<OtherExcArgsTypes>(otherExcArgs)...};
}

/*
 * gsl::Expects and gsl::Ensures trigger terminate(), thus abort()
 * They don't provide an explicit failure mechanism, like exceptions.
 * Unit Testing needs the exception mechanism and at runtime the error
 * messages are helpful.
 *
 * EXPECTS_... and ENSURES_... from below were defined for such contexts.
 */

// Expressing preconditions with a constant error message
// if (!(cond)) does not allow conditions containing initialization part
#define EXPECTS_OR_REPORT_AND_THROW_CONST_MSG(cond, excType, errMsg, ...) \
  {                                                                       \
    if (cond) [[likely]] {                                                \
      ;                                                                   \
    } else [[unlikely]]                                                   \
      REPORT_AND_THROW_CONST_MSG(excType, "Precondition: "s + errMsg,     \
                                 __VA_ARGS__)                             \
  }

// Expressing preconditions with a variable error message
// if (!(cond)) does not allow conditions containing initialization part
#define EXPECTS_OR_REPORT_AND_THROW(cond, excType, errMsg, ...)         \
  {                                                                     \
    if (cond) [[likely]] {                                              \
      ;                                                                 \
    } else [[unlikely]]                                                 \
      reportAndThrow<excType>("Precondition: "s + errMsg, __VA_ARGS__); \
  }

// Expressing postconditions with a constant error message
// if (!(cond)) does not allow conditions containing initialization part
#define ENSURES_OR_REPORT_AND_THROW_CONST_MSG(cond, excType, errMsg, ...) \
  {                                                                       \
    if (cond) [[likely]] {                                                \
      ;                                                                   \
    } else [[unlikely]]                                                   \
      REPORT_AND_THROW_CONST_MSG(excType, "Postcondition: "s + errMsg,    \
                                 __VA_ARGS__)                             \
  }

// Expressing postconditions with a variable error message
// if (!(cond)) does not allow conditions containing initialization part
#define ENSURES_OR_REPORT_AND_THROW(cond, excType, errMsg, ...)          \
  {                                                                      \
    if (cond) [[likely]] {                                               \
      ;                                                                  \
    } else [[unlikely]]                                                  \
      reportAndThrow<excType>("Postcondition: "s + errMsg, __VA_ARGS__); \
  }

}  // namespace pic2sym

namespace p2s = pic2sym;

#endif  // H_MISC
