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

#ifndef H_MISC
#define H_MISC

#pragma warning(push, 0)

#include <functional>
#include <iostream>
#include <optional>
#include <ostream>
#include <string>
#include <type_traits>

#pragma warning(pop)

// Various constants
constexpr double EPS = 1e-6;  ///< Error margin
constexpr double EPSp1 = 1. + EPS;
constexpr double OneMinEPS = 1. - EPS;
constexpr double INV_255 = 1. / 255.;

// Prevent warning about unreferenced parameters
#define UNREFERENCED_PARAMETER(Par) (Par)

// Display an expression and its value
#define PRINT(expr) std::cout << #expr " : " << (expr)
#define PRINTLN(expr) PRINT(expr) << std::endl

// Display values of optionals or `--` if they are not initialized
template <typename CharType, typename T>
std::basic_ostream<CharType, std::char_traits<CharType>>& operator<<(
    std::basic_ostream<CharType, std::char_traits<CharType>>& os,
    const std::optional<T>& v) noexcept {
  if (v)
    return os << *v;

  return os << "--";
}

// Oftentimes functions operating on ranges need the full range.
// Example: copy(x.begin(), x.end(), ..) => copy(BOUNDS(x), ..)
#define BOUNDS(iterable) std::begin(iterable), std::end(iterable)
#define CBOUNDS(iterable) std::cbegin(iterable), std::cend(iterable)
#define BOUNDS_FOR_ITEM_TYPE(iterable, type) \
  iterable.begin<type>(), iterable.end<type>()

// string <-> wstring conversions
std::wstring str2wstr(const std::string& str) noexcept;
std::string wstr2str(const std::wstring& wstr) noexcept;

// Notifying the user
void infoMsg(const std::string& text, const std::string& title = "") noexcept;
void warnMsg(const std::string& text, const std::string& title = "") noexcept;
void errMsg(const std::string& text, const std::string& title = "") noexcept;

/**
Ensures that an action is performed when constructing the object and one when
leaving the scope of this object.
*/
class WrappingActions {
 public:
  /**
  Initializes the object that will perform an action when leaving the scope and
  launches fnCtor

  @param fnCtor is the action to be performed by the constructor
  @param fnDtor_ is the action to be performed by the destructor
  */
  WrappingActions(const std::function<void()>& fnCtor,
                  const std::function<void()>& fnDtor_) noexcept;

  virtual ~WrappingActions() noexcept;  ///< performs fnDtor

  WrappingActions(const WrappingActions&) = delete;
  WrappingActions(WrappingActions&&) = delete;
  void operator=(const WrappingActions&) = delete;
  void operator=(WrappingActions&&) = delete;

  // Inhibit dynamic allocations of this type of object and its derived classes
  static void* operator new(size_t) = delete;
  static void* operator new[](size_t) = delete;

 protected:
  /// Nothing to perform
  static inline const std::function<void()> NoAction = []() noexcept {};

 private:
  /// The function to be called by the destructor
  std::function<void()> fnDtor;
};

/**
Ensures that an action is performed when leaving the scope of a declared object
of this type.

Replacement of <boost/scope_exit.hpp> functionality
*/
class ScopeExitAction : public WrappingActions {
 public:
  /**
  Initializes the object that will perform an action when leaving the scope
  @param fn is the action to be performed by the destructor
  */
  explicit ScopeExitAction(const std::function<void()>& fn) noexcept;
};

// Throwing exceptions while displaying the exception message to the console
/**
First version should be used when the exception message is constant.
Declaring a method-static variable makes sense only when the exception is
caught, so particularly for Unit Testing for this application. Otherwise, the
program just leaves, letting no chance for reusing the method-static variable.
*/
#define THROW_WITH_CONST_MSG(excMsg, excType)                       \
  {                                                                 \
    static const std::string constErrMsgForConsoleAndThrow(excMsg); \
    std::cerr << constErrMsgForConsoleAndThrow << std::endl;        \
    throw excType(constErrMsgForConsoleAndThrow);                   \
  }

/**
Second version should be used when the exception message is variable (reports
specific values).
*/
#define THROW_WITH_VAR_MSG(msg, excType)                   \
  {                                                        \
    const std::string varErrMsgForConsoleAndThrow(msg);    \
    std::cerr << varErrMsgForConsoleAndThrow << std::endl; \
    throw excType(varErrMsgForConsoleAndThrow);            \
  }

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

#endif  // H_MISC
