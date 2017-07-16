/************************************************************************************************
 The application Pic2Sym approximates images by a
 grid of colored symbols with colored backgrounds.

 Copyrights from the libraries used by the program:
 - (c) 2016 Boost (www.boost.org)
		License: <http://www.boost.org/LICENSE_1_0.txt>
			or doc/licenses/Boost.lic
 - (c) 2015 OpenCV (www.opencv.org)
		License: <http://opencv.org/license.html>
            or doc/licenses/OpenCV.lic
 - (c) 2015 The FreeType Project (www.freetype.org)
		License: <http://git.savannah.gnu.org/cgit/freetype/freetype2.git/plain/docs/FTL.TXT>
	        or doc/licenses/FTL.txt
 - (c) 1997-2002 OpenMP Architecture Review Board (www.openmp.org)
   (c) Microsoft Corporation (Visual C++ implementation for OpenMP C/C++ Version 2.0 March 2002)
		See: <https://msdn.microsoft.com/en-us/library/8y6825x5(v=vs.140).aspx>
 - (c) 1995-2013 zlib software (Jean-loup Gailly and Mark Adler - see: www.zlib.net)
		License: <http://www.zlib.net/zlib_license.html>
            or doc/licenses/zlib.lic
 
 (c) 2016, 2017 Florin Tulba <florintulba@yahoo.com>

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
 If not, see <http://www.gnu.org/licenses/agpl-3.0.txt>.
 ***********************************************************************************************/

#ifndef H_UPDATE_SYMS_ACTIONS
#define H_UPDATE_SYMS_ACTIONS

// Avoid using boost preprocessor when checking design of the project with AI Reviewer
#ifndef AI_REVIEWER_CHECK

#pragma warning ( push, 0 )

#include <functional>

#include <boost/lockfree/queue.hpp>

#pragma warning ( pop )

#else // AI_REVIEWER_CHECK defined

struct LockFreeQueue {
	void push(...) {}
	bool pop(...) { return true; }
};

#endif // AI_REVIEWER_CHECK

/// Allows separating the GUI actions related to updating the symbols
struct IUpdateSymsAction /*abstract*/ {
	virtual void perform() = 0; ///< executes the action

	virtual ~IUpdateSymsAction() = 0 {}
};

/// Common realization of IUpdateSymsAction
struct BasicUpdateSymsAction : IUpdateSymsAction {
protected:
	std::function<void()> fn; ///< the function to be called by perform, that has access to private fields & methods

public:
	/// Creating an action object that performs the tasks described in fn_
	BasicUpdateSymsAction(std::function<void()> fn_) : fn(fn_) {}

	void perform() override {
		fn();
	}
};

#ifndef AI_REVIEWER_CHECK
/// Lock-free queue of size 103 (maximum 100 progress notifications + 2 cmap update actions + 1 exception)
typedef boost::lockfree::queue<IUpdateSymsAction*, boost::lockfree::capacity<103>> LockFreeQueue;
#endif // AI_REVIEWER_CHECK

#endif // H_UPDATE_SYMS_ACTIONS
