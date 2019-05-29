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

#ifndef H_WARNINGS
#define H_WARNINGS

/*
Several warnings that need explicit suppression

If defined as below, the UnitTesting project won't parse the #pragma warning

  enum class Warn {
   ConstCondExpr = 4127,
   ....
  };

Example of guarded code that won't parse in UnitTesting (the 'Warn::...' part):

  #pragma warning(disable : Warn::ConstCondExpr)
  ... // Some code generating warning 4127
  #pragma warning(default : Warn::ConstCondExpr)

*/
#define WARN_LVALUE_CAST 4213
#define WARN_INHERITED_VIA_DOMINANCE 4250
#define WARN_INCONSISTENT_DLL_LINKAGE 4273
#define WARN_EXPR_ALWAYS_FALSE 4296
#define WARN_THROWS_ALTHOUGH_NOEXCEPT 4297 26447
#define WARN_BASE_INIT_USING_THIS 4355
#define WARN_DYNAMIC_CAST_MIGHT_FAIL 4437
#define WARN_INCLUDE_UNSAFE_PATH 4464
#define WARN_UNREFERENCED_FUNCTION_REMOVED 4505
#define WARN_CANNOT_GENERATE_ASSIGN_OP 4512
#define WARN_SEH_NOT_CAUGHT 4571
#define WARN_DEPRECATED 4996
#define WARN_EXPLICIT_NEW_OR_DELETE 26409
#define WARN_REF_TO_CONST_UNIQUE_PTR 26410

#endif  // H_WARNINGS
