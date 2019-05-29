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

#ifndef H_UPDATE_SYMS_ACTIONS
#define H_UPDATE_SYMS_ACTIONS

#pragma warning(push, 0)

#include <functional>

#include <boost/lockfree/queue.hpp>

#pragma warning(pop)

/// Allows separating the GUI actions related to updating the symbols
class IUpdateSymsAction /*abstract*/ {
 public:
  virtual void perform() = 0;  ///< executes the action; Might even throw

  virtual ~IUpdateSymsAction() noexcept {}

  // Slicing prevention
  IUpdateSymsAction(const IUpdateSymsAction&) = delete;
  IUpdateSymsAction(IUpdateSymsAction&&) = delete;
  IUpdateSymsAction& operator=(const IUpdateSymsAction&) = delete;
  IUpdateSymsAction& operator=(IUpdateSymsAction&&) = delete;

 protected:
  constexpr IUpdateSymsAction() noexcept {}
};

/// Common realization of IUpdateSymsAction
class BasicUpdateSymsAction : public IUpdateSymsAction {
 public:
  /// Creating an action object that performs the tasks described in fn_
  explicit BasicUpdateSymsAction(std::function<void()> fn_) noexcept
      : fn(fn_) {}

  /// Executes the action which might even throw
  void perform() override { fn(); }

 private:
  /// The function to be called by perform, that has access to private fields &
  /// methods
  std::function<void()> fn;
};

/**
Lock-free queue of size 103 (maximum 100 progress notifications + 2 cmap
update actions + 1 exception)

Cannot use unique_ptr<IUpdateSymsAction> because lockfree requires trivial
assign and destructor for the stored type and unique_ptr doesn't qualify.
*/
typedef boost::lockfree::queue<IUpdateSymsAction*,
                               boost::lockfree::capacity<103>>
    LockFreeQueue;

#endif  // H_UPDATE_SYMS_ACTIONS
